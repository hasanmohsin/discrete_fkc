import torch 

import re 
import pickle
import json 
from pathlib import Path
import argparse
import os
import sys

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

def parse_filename_info(filename):
    """
    Parse information from filename to extract key parameters.
    
    Args:
        filename (str): Filename like "guided_iter_100_L_100_beta_400.0_num_particles_10_seed_1_seq_0"
        
    Returns:
        dict: Dictionary with parsed information
    """
    info = {}
    
    # Check if guided or unguided
    if filename.startswith('guided_'):
        info['guidance'] = 'guided'
    elif filename.startswith('unguided_'):
        info['guidance'] = 'unguided'
    else:
        info['guidance'] = 'unknown'
    
    # Extract length (L_XX)
    length_match = re.search(r'L_(\d+)', filename)
    if length_match:
        info['length'] = int(length_match.group(1))
    else:
        info['length'] = None
    
    # Extract iteration (iter_XX)
    iter_match = re.search(r'iter_(\d+)', filename)
    if iter_match:
        info['iteration'] = int(iter_match.group(1))
    else:
        info['iteration'] = None
    
    # Extract beta value
    beta_match = re.search(r'beta_([0-9.]+)', filename)
    if beta_match:
        info['beta'] = float(beta_match.group(1))
    else:
        info['beta'] = None
    
    # Extract seed
    seed_match = re.search(r'seed_(\d+)', filename)
    if seed_match:
        info['seed'] = int(seed_match.group(1))
    else:
        info['seed'] = None
    
    # Extract number of particles
    particles_match = re.search(r'num_particles_(\d+)', filename)
    if particles_match:
        info['num_particles'] = int(particles_match.group(1))
    else:
        info['num_particles'] = None
    
    return info

def process_fasta_directory(directory_path, reward_func, 
                            guided = True, 
                            num_particles = 10, 
                            lengths = [10, 20, 50, 100],
                            beta = 10.0,
                            num_seq_per_run = -1,
                            output_file=None, save_format='json',
                            save_fastas = True, struct_fasta_save_dir = None):
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
    seq_to_reward = {}
    
    sequence_list = []

    # Get all FASTA files in directory
    # if guided, look for guided files
    if guided:
        fasta_files = list(Path(directory_path).glob(f"guided*num_particles_{num_particles}*.fasta"))
    else:
        fasta_files = list(Path(directory_path).glob(f"*unguided*num_particles_{num_particles}*.fasta"))
    print(f"Found {len(fasta_files)} FASTA files")
    
    for fasta_file in fasta_files:
        print(f"\nProcessing: {fasta_file.name}")
        
        # Extract base filename (without .fasta extension)
        base_key = extract_base_filename(fasta_file)
        
        file_info = parse_filename_info(fasta_file.name)

        print(f"  Parsed info: {file_info}")
        print(" Want to match lengths: {}, beta: {}, num_particles: {}, guided: {}".format(lengths, beta, num_particles, 'guided' if guided else 'unguided'))

        if (file_info['guidance'] != ('guided' if guided else 'unguided')) or (file_info['num_particles'] != num_particles) or (file_info['length'] not in lengths) or (file_info['beta'] != beta):
            print(f"  Skipping file due to mismatch in guidance or num_particles")
            continue

        # Read sequences from FASTA file
        sequences = read_fasta_file(fasta_file)
        print(f"  Found {len(sequences)} sequences")

        if num_seq_per_run > 0:
            sequences_to_analyze = sequences[:num_seq_per_run]
        else:
            sequences_to_analyze = sequences

        sequence_list.extend(sequences_to_analyze)

        # Process each sequence
        for seq_idx, sequence in enumerate(sequences_to_analyze):
 
            if save_fastas:
                # save sequence in structured fasta dir
                if not os.path.exists(struct_fasta_save_dir):
                    os.makedirs(struct_fasta_save_dir)
            
                # struct
                struct_path = struct_fasta_save_dir +"/{}".format('guided' if guided else 'unguided') + \
                                           f"/particle_{num_particles}" + \
                                           f"/length_{file_info['length']}" 
                
                print('\nSaving to struct path: ', struct_path)

                if not os.path.exists(struct_path):
                    os.makedirs(struct_path)



                struct_fasta_filename = struct_path + f"/seed_{file_info['seed']}.fasta"
                
                with open(struct_fasta_filename, 'a') as fasta_fp:
                    fasta_fp.write(f">SEQUENCE_{seq_idx}\n")
                    fasta_fp.write(f"{sequence}\n")
                
                print(f"  Saved structured FASTA: {struct_fasta_filename}")

            if len(sequences_to_analyze) == 1:
                # Single sequence - use base key
                key = base_key
            else:
                # Multiple sequences - add sequence index
                key = f"{base_key}_seq_{seq_idx}"

            # tokenize sequence by reward function
            tokenized_seq = reward_func.tokenizer(sequence, return_tensors='pt')['input_ids'].to(device)


            # Get reward for sequence
            reward = reward_func(tokenized_seq).item()
            reward_dict[key] = reward
            seq_to_reward[sequence] = reward 
            
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
    
    return reward_dict, sequence_list, seq_to_reward 


def main(guided, num_particles, lengths, beta, num_seq_per_run, fksteer, reward_type):
    save_fastas = True

    overall_struct_dir = "./dplm_out_all_structured_all_particles_new/"
    if not os.path.exists(overall_struct_dir):
        os.makedirs(overall_struct_dir)

    struct_fasta_save_dir = overall_struct_dir + "/dplm_out_structured_{}_{}_num_per_run_{}_new".format(reward_type, "fksteer" if fksteer else "dfkc", num_seq_per_run) #./structured_reward_{}/".format()
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    scratch_dir = os.getenv('SCRATCH')
    hf_cache_dir = os.path.join(scratch_dir, 'huggingface_cache')

    denoiser = DPLMDenoiser(device=device)
    tokenizer = denoiser.dplm.tokenizer  # Initialize your tokenizer here

    reward_beta = 1.0  # Adjust this for reward scaling


    if reward_type == "esm2_llhd" or reward_type == "esm2_llhd_early":
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
    if fksteer:

        directory_path = "./dplm_out_FK_thermo/reward_guided_thermo_all_beta_10_fk_base"

        if reward_type == "esm2":
            directory_path = "./dplm_out_FK/reward_guided_esm2_uncond_true_all_beta_200_fk_base"
        elif reward_type == "esm2_llhd":
            directory_path = "./dplm_out_FK_proper_new/reward_guided_esm2_uncond_proper_true_all_beta_10_fk_base"
        elif reward_type == "esm2_llhd_early":
            directory_path = "./dplm_out_FK_proper_early_0p95_new/reward_guided_esm2_uncond_proper_early_true_all_beta_10_fk_base"
        
    else:
        directory_path = "./dplm_out_thermo/reward_guided_thermo_all_beta_10" #reward_guided_esm2_reference_15_mask_partial_cont_False_eos_bos_clamp"
        
        if reward_type == "esm2":
            directory_path = "./dplm_out/reward_guided_esm2_uncond_true_all_beta_200_new_settings"
        elif reward_type == "esm2_llhd":
            directory_path = "./dplm_out_proper/reward_guided_esm2_uncond_true_all_beta_10"
        elif reward_type == "esm2_llhd_early":
            directory_path = "./dplm_out_proper_early_stop_0p95_new/reward_guided_esm2_uncond_true_all_early_beta_10_mask_fill_True"

    # put true or false for guided
    guided_str = "guided" if guided else "unguided"

    json_dir = "./dplm_out_{}_{}_json/".format(reward_type, "fksteer" if fksteer else "dfkc")

    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    output_file = f"{json_dir}/protein_{reward_type}_cond_{guided_str}_num_particles_{num_particles}_beta_{beta}_lens_{'_'.join(map(str, lengths))}.json"  # or "protein_rewards.pkl" for pickle


    # Process all FASTA files and get rewards
    reward_dict, sequence_list, seq_to_reward = process_fasta_directory(
        directory_path=directory_path,
        reward_func = reward_fn,
        output_file=output_file,
        guided = guided,
        num_particles = num_particles,
        beta = beta,
        lengths = lengths,
        num_seq_per_run = num_seq_per_run,
        save_format='json',  # or 'pickle'
        save_fastas=save_fastas,
        struct_fasta_save_dir=struct_fasta_save_dir
    )
    
    # Print summary statistics
    if reward_dict:
        rewards = list(reward_dict.values())
        mean_reward = sum(rewards)/len(rewards)
        std_reward = torch.std(torch.tensor(rewards)).item()
        stderr_reward = std_reward/np.sqrt(len(rewards))

        print(f"\n=== Summary Statistics ===")
        print(f"Total sequences processed: {len(rewards)}")
        print(f"Mean reward: {mean_reward:.6f}")
        print(f"Std reward: {std_reward:.6f}")
        print(f"Min reward: {min(rewards):.6f}")
        print(f"Max reward: {max(rewards):.6f}")
        
        # Show first few entries as example
        print(f"\n=== First 5 entries ===")
        for i, (key, reward) in enumerate(list(reward_dict.items())[:5]):
            print(f"{key}: {reward:.6f}")

    seq_div, std_div = sequence_similarity.diversity_batch(sequences=sequence_list)

    print(f"\nSequence Diversity (avg pairwise identity): {seq_div:.6f} ± {std_div:.6f}")

    # save all sequences to a single fasta file
    fasta_superdir = "./concat_fastas_all_len_new_proper_reward/"
    if not os.path.exists(fasta_superdir):
        os.makedirs(fasta_superdir)
    fasta_savedir = f"{fasta_superdir}dplm_out_{reward_type}_all_seq_{'fksteer' if fksteer else 'dfkc'}/"

    if not os.path.exists(fasta_savedir):
        os.makedirs(fasta_savedir)

    fasta_output_file = f"{fasta_savedir}/all_sequences_{guided_str}_num_particles_{num_particles}_beta_{beta}_lens_{'_'.join(map(str, lengths))}_per_run_{num_seq_per_run}.fasta"
    with open(fasta_output_file, 'w') as fasta_fp:
        for idx, sequence in enumerate(sequence_list):
            fasta_fp.write(f">SEQUENCE_{idx}\n")
            fasta_fp.write(f"{sequence}\n")
    print(f"All sequences saved to {fasta_output_file}")

    return mean_reward, std_reward, seq_div, std_div, len(sequence_list), seq_to_reward 

def all_runs_length_ablation():
    guided = True  #True #False 
    #num_particles = 5
    lengths = [10, 50, 100]
    beta = 10.0 #200.0 #10.0
    num_seq_per_run = -1  # set to -1 to process all sequences
    fksteer = True   #True #False #True

    reward_type = "esm2_llhd" #"thermo" #"thermo"#"esm2_llhd"#"esm2_llhd_early"#_early" #"esm2" #"thermo" #"esm2" #"thermo"

    result_dict = {}
    seq_dict = {}
    print_top5 = True #False 


    for num_particle in [1, 5, 10]:
        
        num_particle_res = {}
        top5_seq = {}
        
        for length in lengths:
        #if True:
                    mean_reward, std_reward, mean_div, std_div, num_points, seq_to_reward = main(guided = guided,
                        num_particles = num_particle, 
                        lengths = [length], 
                        beta = beta, 
                        num_seq_per_run = num_seq_per_run, 
                        fksteer = fksteer, 
                        reward_type = reward_type)

                    num_particle_res[f"Length: {length}"] = (mean_reward, std_reward, mean_div, std_div, num_points)

                    # get top 5 largest sequences (keys) in seq_to_reward dict
                    sorted_seqs = sorted(seq_to_reward.items(), key=lambda x: x[1], reverse=True)
                    top5_seq[f"Length: {length}"] = sorted_seqs[:5]

                    # save seq_to_reward to a csv file
                    import pandas as pd
                    df = pd.DataFrame(list(seq_to_reward.items()), columns=['Sequence', 'Reward'])

                    csv_dir = "./seq_to_reward_{}/{}_particle_{}_len_{}.csv".format(reward_type, "fksteer" if fksteer else "dfkc", num_particle, length)
                    if not os.path.exists("./seq_to_reward_{}/".format(reward_type)):
                        os.makedirs("./seq_to_reward_{}/".format(reward_type))
                    df.to_csv(csv_dir, index=False)
                    print(f"Saved seq_to_reward to {csv_dir}")

        result_dict[f"Num Particles: {num_particle}"] = num_particle_res
        seq_dict[f"Num Particles: {num_particle}"] = top5_seq 
        #result_dict[f"Top 5 Sequences Particles: {num_particle}"] = top5_seq


    
    # iterate over num particles
    for num_particle, res in result_dict.items():
        print(f"\n\nResults for {num_particle}:")
        
        avg_mean_reward = 0.0
        avg_std_reward = 0.0
        avg_mean_div = 0.0
        avg_std_div = 0.0
        total_points = 0

        for length_key, stats in res.items():
            mean_reward, std_reward, mean_div, std_div, num_points = stats
            print(f"\n  {length_key} -> Mean Reward: {mean_reward:.6f} ± {std_reward:.6f} (stderr: {std_reward/np.sqrt(num_points):.6f}), Diversity: {mean_div:.6f} ± {std_div:.6f} (stderr: {std_reward/np.sqrt(num_points)}, num_points: {num_points})")
            
            if print_top5:
                print(f"    Top 5 sequences:")
                for seq, reward in seq_dict[num_particle][length_key]:
                    print(f"      {seq}: {reward}")

            avg_mean_reward += mean_reward * num_points
            avg_std_reward += std_reward * num_points
            avg_mean_div += mean_div * num_points
            avg_std_div += std_div * num_points
            total_points += num_points
        print(f"Averages all lengths for {num_particle}: Mean Reward: {avg_mean_reward/total_points:.6f} ± {avg_std_reward/total_points:.6f} (stderr: {avg_std_reward/(np.sqrt(num_points)*total_points)}), Diversity: {avg_mean_div/total_points:.6f} ± {avg_std_div/total_points:.6f} (stderr: {avg_std_div/(np.sqrt(num_points)*total_points)})")

def all_runs():
    #guided = True 
    #num_particles = 5
    lengths = [10, 50, 100]
    beta = 200.0 #200.0 #10.0
    num_seq_per_run = -1  # set to -1 to process all sequences
    #fksteer = False

    reward_type = "esm2" #"thermo" #"esm2" #"thermo"

    result_dict = {}



    for num_particle in [1, 5, 10]:
        
        num_particle_res = {}
    
        for guided in [True, False]:
            for fksteer in [True, False]:
                for length in lengths:

                    mean_reward, std_reward, mean_div, std_div, num_points = main(guided = guided,
                        num_particles = num_particle, 
                        lengths = lengths, 
                        beta = beta, 
                        num_seq_per_run = num_seq_per_run, 
                        fksteer = fksteer, 
                        reward_type = reward_type)

def single_run():
    guided = True 
    num_particles = 1
    lengths = [10, 50, 100]
    beta = 10.0 #200.0 #10.0
    num_seq_per_run = -1  # set to -1 to process all sequences
    fksteer = False

    reward_type = "thermo" #"thermo" #"esm2" #"thermo"

    result_dict = {}

    main(guided = guided,
        num_particles = num_particles, 
        lengths = lengths, 
        beta = beta, 
        num_seq_per_run = num_seq_per_run, 
        fksteer = fksteer, 
        reward_type = reward_type)

if __name__ == "__main__":
    #single_run()
    all_runs_length_ablation()
                    
                    