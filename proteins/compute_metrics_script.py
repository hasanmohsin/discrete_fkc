import torch 

import pickle
import json 
from pathlib import Path

from protein_reward import * 
from protein_esm2_reward import ESM2ProteinReward
from protein_esm2_llhd_reward import ESM2ProperLikelihoodProteinReward

import sequence_similarity


def read_fasta_file(filepath):
    """
    Read a FASTA file and return a list of sequences.
    
    Args:
        filepath (str): Path to the FASTA file
        
    Returns:
        list: List of protein sequences (strings)
    """
    sequences = []
    current_sequence = ""
    
    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                # If we have a current sequence, add it to the list
                if current_sequence:
                    sequences.append(current_sequence)
                    current_sequence = ""
            else:
                # Add sequence line to current sequence
                current_sequence += line
        
        # Add the last sequence if it exists
        if current_sequence:
            sequences.append(current_sequence)
    
    return sequences


def extract_base_filename(filepath):
    """
    Extract the base filename without extension.
    
    Args:
        filepath (Path or str): Path to the file
        
    Returns:
        str: Base filename without .fasta extension
    """
    return Path(filepath).stem


def process_fasta_directory(directory_path, reward_func, output_file=None, save_format='json'):
    """
    Process all FASTA files in a directory and create a dictionary of rewards.
    
    Args:
        directory_path (str): Path to directory containing FASTA files
        output_file (str, optional): Path to save the results
        save_format (str): Format to save results ('json' or 'pickle')
        
    Returns:
        dict: Dictionary with keys as modified filenames and values as rewards
    """
    # Initialize reward model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dictionary to store results
    reward_dict = {}
    
    # Get all FASTA files in directory
    fasta_files = list(Path(directory_path).glob("*.fasta"))
    print(f"Found {len(fasta_files)} FASTA files")
    
    for fasta_file in fasta_files:
        print(f"\nProcessing: {fasta_file.name}")
        
        # Extract base filename (without .fasta extension)
        base_key = extract_base_filename(fasta_file)
        
        # Read sequences from FASTA file
        sequences = read_fasta_file(fasta_file)
        print(f"  Found {len(sequences)} sequences")
        
        # Process each sequence
        for seq_idx, sequence in enumerate(sequences):
            if len(sequences) == 1:
                # Single sequence - use base key
                key = base_key
            else:
                # Multiple sequences - add sequence index
                key = f"{base_key}_seq_{seq_idx}"
            
            # tokenize sequence if needed by reward function
            tokenized_seq = reward_func.tokenizer(sequence, return_tensors='pt')['input_ids'].to(device)


            # Get reward for sequence
            reward = reward_func(tokenized_seq).item()
            reward_dict[key] = reward
            
            print(f"    {key}: {reward:.6f} (seq: {sequence[:20]}{'...' if len(sequence) > 20 else ''})")
    
    print(f"\nProcessed {len(reward_dict)} total sequences")
    
    # Save results if output file specified
    if output_file:
        if save_format.lower() == 'json':
            with open(output_file, 'w') as f:
                json.dump(reward_dict, f, indent=2)
            print(f"Results saved to {output_file} (JSON format)")
        elif save_format.lower() == 'pickle':
            with open(output_file, 'wb') as f:
                pickle.dump(reward_dict, f)
            print(f"Results saved to {output_file} (Pickle format)")
        else:
            raise ValueError("save_format must be 'json' or 'pickle'")
    
    return reward_dict


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    scratch_dir = os.getenv('SCRATCH')
    hf_cache_dir = os.path.join(scratch_dir, 'huggingface_cache')

    denoiser = DPLMDenoiser(device=device)
    tokenizer = denoiser.dplm.tokenizer  # Initialize your tokenizer here

    reward_beta = 1.0  # Adjust this for reward scaling


    reward_type = "thermo"

    if reward_type == "esm2_llhd":
        reward_fn = ESM2ProperLikelihoodProteinReward( tokenizer=tokenizer,
                                                       beta = reward_beta,  # Adjust this for reward scaling
                                                       hf_cache_dir=hf_cache_dir,
                                                       device="cuda"
                        )
    elif reward_type == "esm2":
        reward_fn = ESM2ProteinReward(
            tokenizer=tokenizer,
            beta = reward_beta,  # Adjust this for reward scaling
            hf_cache_dir=hf_cache_dir,
            device="cuda"
        )
    elif reward_type == "thermo":
        reward_fn = Thermostability(device=device, 
                                tokenizer = denoiser.dplm.tokenizer, 
                                beta=reward_beta, 
                                hf_cache_dir=hf_cache_dir)
        
    else:
        raise ValueError("Invalid reward type specified.")

    # Configuration
    directory_path = "./dplm_out_thermo/reward_guided_thermo_all_beta_10" #reward_guided_esm2_reference_15_mask_partial_cont_False_eos_bos_clamp"
    output_file = "protein_thermo_cond_new_settings.json"  # or "protein_rewards.pkl" for pickle
    
    # Process all FASTA files and get rewards
    reward_dict = process_fasta_directory(
        directory_path=directory_path,
        reward_func = reward_fn,
        output_file=output_file,
        save_format='json'  # or 'pickle'
    )
    
    # Print summary statistics
    if reward_dict:
        rewards = list(reward_dict.values())
        print(f"\n=== Summary Statistics ===")
        print(f"Total sequences processed: {len(rewards)}")
        print(f"Mean reward: {sum(rewards)/len(rewards):.6f}")
        print(f"Min reward: {min(rewards):.6f}")
        print(f"Max reward: {max(rewards):.6f}")
        
        # Show first few entries as example
        print(f"\n=== First 5 entries ===")
        for i, (key, reward) in enumerate(list(reward_dict.items())[:5]):
            print(f"{key}: {reward:.6f}")


if __name__ == "__main__":
    main()