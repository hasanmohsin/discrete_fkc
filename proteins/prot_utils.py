from transformers.models.esm.openfold_utils.protein import to_pdb, Protein as OFProtein
from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

from biotite.structure import AtomArray, sasa, annotate_sse
from biotite.structure.io.pdb import PDBFile

from scipy.spatial import ConvexHull
from typing import Dict, List, Tuple, Generator, Optional, Union
from io import StringIO

import torch
import copy
import numpy as np
import torch
from Bio import PDB
from tmtools import tm_align

_EPSILON = 1e-10
_HYDROPHILICS = {"ARG", "LYS", "ASP", "GLU", "HIS", "ASN", "GLN", "SER", "THR", "TYR"}
_HYDROPHOBICS = {"VAL", "ILE", "LEU", "PHE", "MET", "TRP"}

def secondary_structure(atom_array: AtomArray):
    # Get secondary structure elements ('H' for helix, 'E' for sheet, 'C' for coil)
    sse = annotate_sse(atom_array)

    # Compute the ratios for helix, sheet, and coil
    helix = sum(i == 'a' for i in sse) / len(sse)
    sheet = sum(i == 'b' for i in sse) / len(sse)
    coil = sum(i == 'c' for i in sse) / len(sse)

    return {"helix": helix, "sheet": sheet, "coil": coil}


def get_sheet_percent(atom_array: AtomArray):
    # Get secondary structure elements ('H' for helix, 'E' for sheet, 'C' for coil)
    sse = annotate_sse(atom_array)
    return sum(i == 'b' for i in sse) / len(sse)

def secondary_structure_diversity_score(
    atom_arrays: List[AtomArray],
    helix_weight: float = 1.0,
    strand_weight: float = 2.0,
    coil_weight: float = 0.5
):
    sse_results = [secondary_structure(atom_arr) for atom_arr in atom_arrays]
    struct_props = torch.cat([
        torch.tensor([x[ss_type] for x in sse_results]).unsqueeze(1)
        for ss_type in ['helix', 'sheet', 'coil']
    ], dim=1)

    sse_weights = torch.tensor([
        helix_weight, strand_weight, coil_weight
    ]).unsqueeze(0)

    base_scores = (struct_props * sse_weights).sum(dim=1)
    entropies = -(struct_props * (struct_props + _EPSILON).log()).sum(dim=1)

    return base_scores * entropies


def _calculate_bin_centers(boundaries: torch.Tensor) -> torch.Tensor:
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat([bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0)
    return bin_centers


def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(0, max_bin, steps=(no_bins - 1), device=logits.device)

    bin_centers = _calculate_bin_centers(boundaries)
    torch.sum(residue_weights)
    n = logits.shape[-2]
    clipped_n = max(n, 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers**2) / (d0**2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    normed_residue_mask = residue_weights / (eps + residue_weights.sum())
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights

    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]


def outputs_to_pdb_strs(outputs, idxs):
    final_atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs)
    final_atom_mask = outputs["atom37_atom_exists"]
    pdbs = []
    for i in idxs:
        aa = outputs["aatype"][i]
        pred_pos = final_atom_positions[i]
        mask = final_atom_mask[i]
        resid = outputs["residue_index"][i] + 1
        pred = OFProtein(
            aatype=aa.cpu().numpy(),
            atom_positions=pred_pos.cpu().numpy(),
            atom_mask=mask.cpu().numpy(),
            residue_index=resid.cpu().numpy(),
            b_factors=outputs["plddt"][i].cpu().float().numpy(),
            chain_index=outputs["chain_index"][i].cpu.numpy() if "chain_index" in outputs else None,
        )
        pdbs.append(to_pdb(pred))

    return pdbs


def _is_Nx3(array: np.ndarray) -> bool:
    return len(array.shape) == 2 and array.shape[1] == 3


def get_center_of_mass(coordinates: np.ndarray) -> np.ndarray:
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    return coordinates.mean(axis=0).reshape(1, 3)


def distances_to_centroid(coordinates: np.ndarray) -> np.ndarray:
    """
    Computes the distances from each of the coordinates to the
    centroid of all coordinates.
    """
    assert _is_Nx3(coordinates), "Coordinates must be Nx3."
    center_of_mass = get_center_of_mass(coordinates)
    m = coordinates - center_of_mass
    return np.linalg.norm(m, axis=-1)


def hydrophobic_hydrophilic_score(
    atom_array: AtomArray,
    start_residue_index: Optional[int] = None,
    end_residue_index: Optional[int] = None,
    threshold: float = 0.2  # SASA threshold to define "buried" residues
) -> float:
    """
    Computes ratio of hydrophobic atoms in a biotite AtomArray that are also surface
    exposed. Typically, lower is better.
    """

    hydrophobic_mask = np.array([aa in _HYDROPHOBICS for aa in atom_array.res_name])
    hydrophilic_mask = np.array([aa in _HYDROPHILICS for aa in atom_array.res_name])

    if (~hydrophobic_mask).all() and (~hydrophilic_mask).all():
        return 0.0, 0.0

    if start_residue_index is None and end_residue_index is None:
        selection_mask = np.ones_like(hydrophobic_mask)
    else:
        start_residue_index = 0 if start_residue_index is None else start_residue_index
        end_residue_index = (
            len(hydrophobic_mask) if end_residue_index is None else end_residue_index
        )
        selection_mask = np.array(
            [
                i >= start_residue_index and i < end_residue_index
                for i in range(len(hydrophobic_mask))
            ]
        )

    # TODO(scandido): Resolve the float/bool thing going on here.
    atom_array = copy.deepcopy(atom_array)

    sasa_result = sasa(atom_array)

    surface_mask = sasa_result >= threshold
    buried_mask = sasa_result < threshold

    hydrophobic_surf = np.logical_and(selection_mask * hydrophobic_mask, surface_mask)
    hydrophilic_core = np.logical_and(selection_mask * hydrophilic_mask, buried_mask)

    # TODO(brianhie): Figure out how to handle divide-by-zero.
    hydrophobic_denom = sum(selection_mask * hydrophobic_mask)
    hydrophilic_denom = sum(selection_mask * hydrophilic_mask)

    hydrophobic_score = 0.0
    if hydrophobic_mask.any() and hydrophobic_denom != 0:
        hydrophobic_score = sum(hydrophobic_surf) / hydrophobic_denom

    hydrophilic_score = 0.0
    if hydrophilic_mask.any() and hydrophilic_denom != 0:
        hydrophilic_score = sum(hydrophilic_core) / hydrophilic_denom

    return hydrophobic_score, hydrophilic_score


def get_backbone_atoms(atoms: AtomArray) -> AtomArray:
    return atoms[
        (atoms.atom_name == "CA") | (atoms.atom_name == "N") | (atoms.atom_name == "C")
    ]
def get_ca_atoms(atoms: AtomArray) -> AtomArray:
    return atoms[atoms.atom_name == "CA"]

def compute_geometric_properties(atom_coords: np.ndarray) -> dict:
    """
    Compute various geometric properties based on backbone atom coordinates.

    Args:
    - atom_coords (np.ndarray): Array of shape (N, 3) where N is the number of atoms and each row represents xyz coordinates.

    Returns:
    - properties (dict): Dictionary containing density within Rg, volume-to-surface ratio, and volumetric compactness.
    """
    # Check input shape
    assert atom_coords.shape[1] == 3, "Input array must have shape (N, 3)"

    # Compute the centroid (center of mass)
    centroid = np.mean(atom_coords, axis=0)

    # Calculate the radius of gyration (Rg)
    distances_from_centroid = np.linalg.norm(atom_coords - centroid, axis=1)
    Rg = np.sqrt(np.mean(distances_from_centroid**2))

    # Use all atoms to compute the convex hull
    try:
        hull = ConvexHull(atom_coords)
        volume = hull.volume
        surface_area = hull.area
    except Exception as e:
        return {"error": f"ConvexHull computation failed: {e}"}

    # Compute density using all atoms
    density = len(atom_coords) / volume if volume > 0 else np.nan

    # Volume-to-surface area ratio
    volume_to_surface_ratio = volume / surface_area if surface_area > 0 else np.nan

    # Volumetric compactness: volume of the hull / volume of the enclosing sphere
    max_dist = np.max(np.linalg.norm(atom_coords - centroid, axis=1))
    sphere_volume = (4/3) * np.pi * (max_dist**3)
    volumetric_compactness = volume / sphere_volume if sphere_volume > 0 else np.nan

    return {
        "Rg": Rg, # minimize
        "density": density, # maximize
        "volume_to_surface_ratio": volume_to_surface_ratio, # maximize
        "volumetric_compactness": volumetric_compactness, # maximize
    }


def pdb_file_to_atomarray(pdb_path: Union[str, StringIO]) -> AtomArray:
    return PDBFile.read(pdb_path).get_structure(model=1)


def get_ca_coordinates(pdb_path):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_coords.append(residue["CA"].coord)
    ca_tensor = torch.tensor(ca_coords, dtype=torch.float32)
    return ca_tensor


def calc_tm_score(pos_1, pos_2):
    seq_1 = 'A' * pos_1.shape[0]
    seq_2 = 'A' * pos_2.shape[0]
    tm_results = tm_align(pos_1, pos_2, seq_1, seq_2)
    return tm_results.tm_norm_chain1, tm_results.tm_norm_chain2 

