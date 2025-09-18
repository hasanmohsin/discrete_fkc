import torch
import torch.nn as nn
import numpy as np
import os
import sys
from transformers import EsmForMaskedLM, AutoTokenizer
import warnings


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from smc_sampler import RewardSampler
from samplers import DiffusionSampler
from dplm_denoiser import DPLMDenoiser
from utils import set_all_seeds

# Suppress transformers warnings
warnings.filterwarnings(
    'ignore', message='Some weights of the model checkpoint.*were not used when initializing.*')



# Add the parent directory to Python path to access dplm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from dplm.generate_dplm import initialize_generation

class ESM2ProteinReward():
    def __init__(self, tokenizer, device, beta=1.0,
                 model_name="esm2_t33_650M_UR50D", hf_cache_dir=None):
        self.tokenizer = tokenizer  # DPLM tokenizer
        self.beta = beta
        self.device = device

        self.invalid_seq_score = -10.0  # Score for invalid sequences

        self.name = "ESM2Reward_Uncond_" + model_name

        # Load ESM2 model and tokenizer
        self.esm_model = EsmForMaskedLM.from_pretrained(
            f"facebook/{model_name}", cache_dir=hf_cache_dir).to(device)
        self.esm_tokenizer = AutoTokenizer.from_pretrained(
            f"facebook/{model_name}", cache_dir=hf_cache_dir)
        self.esm_model.eval()

        print(f"Loaded ESM2 model: {model_name}")

    def _score_sequence_esm2(self, sequence):
        """
        Score a protein sequence based on ESM2's likelihood predictions.
        Uses the sequence itself as input to get position-specific probabilities.
        """
        # Clean sequence - remove special tokens and invalid characters
        # Keep only valid amino acids
        valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
        cleaned_sequence = ''.join([aa for aa in sequence if aa in valid_aas])
        
        if not cleaned_sequence:
            print(f"Warning: No valid amino acids found in sequence '{sequence}'")
            return 0.0
        
        # Tokenize the cleaned sequence for ESM2
        inputs = self.esm_tokenizer(
            cleaned_sequence, return_tensors="pt").to(self.device)
        
        scores = []
        
        with torch.no_grad():
            outputs = self.esm_model(**inputs)
            # Shape: [seq_length + 2, vocab_size] (includes BOS/EOS)
            logits = outputs.logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)
            
            # Score each position in the cleaned sequence
            for pos in range(len(cleaned_sequence)):
                seq_aa = cleaned_sequence[pos]
                
                # Position in tokenized sequence (add 1 for BOS token)
                token_pos = pos + 1
                
                # Ensure we don't go out of bounds
                if token_pos >= logits.shape[0]:
                    print(f"Warning: token_pos {token_pos} out of bounds for sequence length {logits.shape[0]}")
                    continue
                
                # Get token ID for sequence amino acid
                seq_token_id = self.esm_tokenizer.convert_tokens_to_ids(seq_aa)
                
                # Skip invalid amino acids (shouldn't happen after cleaning, but just in case)
                if seq_token_id is None:
                    print(f"Warning: Invalid amino acid '{seq_aa}' at position {pos}")
                    continue
                
                # Get log probability for this amino acid at this position
                score = log_probs[token_pos, seq_token_id].item()
                scores.append(score)
        
        # Return average score
        if not scores:
            return 0.0
        
        return sum(scores) / len(scores)

    def score_sequence(self, sequence):
        """
        Public method to score a protein sequence.
        Returns the average log-likelihood of the sequence under ESM2.
        """
        return self._score_sequence_esm2(sequence)

    def score_sequences(self, sequences):
        """
        Score multiple protein sequences.
        Returns a list of scores in the same order as input sequences.
        """
        return [self.score_sequence(seq) for seq in sequences]

    def __call__(self, input_seq):
        """
        Make the class callable for use with RewardSampler.
        """
        batch_size = input_seq.shape[0]
        log_rewards = []

        # Decode sequences from DPLM tokens to amino acid sequences
        decoded_sequences = self.tokenizer.batch_decode(
            input_seq, skip_special_tokens=True
        )

        # Clean sequences (remove spaces if any)
        cleaned_sequences = ["".join(seq.split(" "))
                             for seq in decoded_sequences]

        # Score each sequence
        for seq in cleaned_sequences:
            try:
                score = self._score_sequence_esm2(seq)
                # Apply temperature scaling
                reward = self.beta * score
                log_rewards.append(reward)
            except Exception as e:
                print(f"Error scoring sequence '{seq}': {e}")
                log_rewards.append(self.beta * self.invalid_seq_score)  # Fallback score

        # Convert to tensor and ensure same device as input
        log_rewards = torch.tensor(
            log_rewards, dtype=torch.float32, device=self.device)

        return log_rewards
        
class ESM2ProteinRewardReference():
    def __init__(self, reference_sequence, tokenizer, device, beta=1.0,
                 model_name="esm2_t33_650M_UR50D", hf_cache_dir=None):
        self.reference_sequence = reference_sequence
        self.tokenizer = tokenizer  # DPLM tokenizer
        self.beta = beta
        self.device = device

        self.name = "ESM2Reward_Reference_" + model_name

        # Load ESM2 model and tokenizer
        self.esm_model = EsmForMaskedLM.from_pretrained(
            f"facebook/{model_name}", cache_dir=hf_cache_dir).to(device)
        self.esm_tokenizer = AutoTokenizer.from_pretrained(
            f"facebook/{model_name}", cache_dir=hf_cache_dir)
        self.esm_model.eval()

        print(f"Loaded ESM2 model: {model_name}")
        print(f"Reference sequence length: {len(reference_sequence)}")

    def _score_sequence_esm2(self, sequence, reference_sequence=None):
        if reference_sequence is None:
            reference_sequence = self.reference_sequence

        # Handle length mismatch by truncating or padding
        if len(sequence) != len(reference_sequence):
            min_len = min(len(sequence), len(reference_sequence))
            sequence = sequence[:min_len]
            reference_sequence = reference_sequence[:min_len]

        # Tokenize reference sequence for ESM2
        inputs = self.esm_tokenizer(
            reference_sequence, return_tensors="pt").to(self.device)

        scores = []

        with torch.no_grad():
            outputs = self.esm_model(**inputs)
            logits = outputs.logits[0]
            log_probs = torch.log_softmax(logits, dim=-1)

            # Score each position
            for pos in range(len(sequence)):
                seq_aa = sequence[pos]

                # Position in tokenized sequence (add 1 for BOS token)
                token_pos = pos + 1

                # Get token ID for sequence amino acid
                seq_token_id = self.esm_tokenizer.convert_tokens_to_ids(seq_aa)

                # Skip invalid amino acids
                if seq_token_id is None:
                    continue

                # Get log probability for this amino acid at this position
                score = log_probs[token_pos, seq_token_id].item()
                scores.append(score)

        # Return average score
        if not scores:
            return 0.0

        return sum(scores) / len(scores)

    def __call__(self, input_seq):

        batch_size = input_seq.shape[0]
        log_rewards = []

        # Decode sequences from DPLM tokens to amino acid sequences
        decoded_sequences = self.tokenizer.batch_decode(
            input_seq, skip_special_tokens=True
        )

        # Clean sequences (remove spaces if any)
        cleaned_sequences = ["".join(seq.split(" "))
                             for seq in decoded_sequences]

        # Score each sequence
        for seq in cleaned_sequences:
            try:
                score = self._score_sequence_esm2(seq)
                # Apply temperature scaling
                reward = self.beta * score
                log_rewards.append(reward)
            except Exception as e:
                print(f"Error scoring sequence: {e}")
                log_rewards.append(0.0)  # Fallback score

        # Convert to tensor and ensure same device as input
        log_rewards = torch.tensor(
            log_rewards, dtype=torch.float32, device=self.device)

        return log_rewards


class SubstringReward():
    def __init__(self, target_string, tokenizer, device, beta=1.0):
        self.target_string = target_string
        self.tokenizer = tokenizer

        # find token ids for target string
        self.target_ids = tokenizer.encode(
            target_string, return_tensors="pt")[0]

        # eliminate first and last token, since its bos and eos tokens
        self.target_ids = self.target_ids[1:-1]

        if len(self.target_ids) == 1:
            self.target_ids = self.target_ids[0]

        self.beta = beta
        self.device = device

    # outputs log reward of target
    # same device as input seq
    def __call__(self, input_seq):

        # batch of input_seq [B, L]

        # count number of times the target id seq appears in each sample
        logr_val = (input_seq == self.target_ids).sum(dim=-1)

        # tempering:
        logr_val = self.beta * logr_val
        logr_val = logr_val.to(self.device)

        return logr_val


class BetaSheetReward():
    def __init__(self, device, beta=1.0):
        self.beta = beta
        self.device = device

    def __call__(self, input_seq):
        pass


def main():
    device = 'cuda'
    denoiser = DPLMDenoiser(device=device)
    seed = 3315

    set_all_seeds(seed)

    scratch_dir = os.getenv('SCRATCH')
    hf_cache_dir = os.path.join(scratch_dir, 'huggingface_cache')

    tokenizer = denoiser.dplm.tokenizer  # Initialize your tokenizer here

    seq_length = 50
    num_seqs = 5

    # Initialize without reference sequence
    reward_fn = ESM2ProteinReward(
        tokenizer=tokenizer,
        beta = 20.0,  # Adjust this for reward scaling
        hf_cache_dir=hf_cache_dir,
        device="cuda"
    )

    # Score a single sequence
    sequence = "MKLLVLSLVLVAPMAAQAAEITLVPSVKLQIGDRDNRGYYW"
    score = reward_fn.score_sequence(sequence)
    print(f"Score for sequence '{sequence}': {score}")

    # Define reference sequence (wildtype)
    #reference_sequence = "MSIQ"
    #reward_fn = ESM2ProteinRewardReference(
    #    reference_sequence=reference_sequence,
    #    tokenizer=tokenizer,
    #    device=device,
    #    beta=1.0  # Adjust this for reward scaling
    #)

    #target_string = "N"

    #reward_fn = SubstringReward(target_string, tokenizer, device=device)
    #print("Target IDs: ", reward_fn.target_ids)

    input_seq = initialize_generation(
        length=seq_length,
        num_seqs=num_seqs,
        tokenizer=denoiser.dplm.tokenizer,
        device=device
    )

    sampler = DiffusionSampler(denoiser=denoiser, steps=seq_length, temperature=1.0)
    r_sampler = RewardSampler(denoiser=denoiser,
                              log_reward_func=reward_fn,
                              resample=True,
                              adaptive_resampling=False,
                              steps=seq_length,
                              temperature=1.0)
    
    set_all_seeds(seed)
    x, x0, x_traj = sampler.sample(
        input_seq, batch_size = 5, return_traj=True, remasking='low_conf_noisy', log_wandb=False)

    num_particles = 1
    batch_num = 1

    input_seq_2 = initialize_generation(
        length=seq_length,
        num_seqs=batch_num * num_particles,
        tokenizer=denoiser.dplm.tokenizer,
        device=device
    )

    input_seq_particles = input_seq_2.reshape(batch_num, num_particles, -1)
    
    set_all_seeds(seed)
    x_r, x0_r, x_traj_r, ess_traj, log_weights_traj = r_sampler.sample(input_seq_particles, return_traj=True, remasking='low_conf_noisy',
                                                                       batch_size=batch_num, num_particles=num_particles, log_wandb=False)

    print("Unguided x: ", x)
    print("Guided x: ", x_r.view(-1, x_r.shape[-1]))

    # Compute the reward
    reward = reward_fn(x)
    print("Unguided Reward: ", reward.mean())
    print("Guided Reward: ", reward_fn(x_r).mean())

    # save guided and unguided as files

    tokenizer = denoiser.dplm.tokenizer

    output_tokens = x

    output_results = [
        "".join(seq.split(" "))
        for seq in tokenizer.batch_decode(
            output_tokens, skip_special_tokens=True
        )
    ]
    print("Unguided Results: ", output_results)

    output_results_guided = [
        "".join(seq.split(" "))
        for seq in tokenizer.batch_decode(
            x_r, skip_special_tokens=True
        )
    ]
    print("Guided Results: ", output_results_guided)

    saveto = "./dplm_out/reward_guided_esm2_uncond"

    os.makedirs(saveto, exist_ok=True)
    saveto_name = os.path.join(
        saveto, f"unguided_iter_{seq_length}_L_{seq_length}_beta_{reward_fn.beta}_seed_{seed}.fasta"
    )
    fp_save = open(saveto_name, "w")
    for idx, seq in enumerate(output_results):
        fp_save.write(f">SEQUENCE_{seq_length}_L={seq_length}\n")
        fp_save.write(f"{seq}\n")
    fp_save.close()

    # Save guided sequences
    saveto_name = os.path.join(
        saveto, f"guided_iter_{seq_length}_L_{seq_length}_beta_{reward_fn.beta}_seed_{seed}.fasta"
    )
    fp_save = open(saveto_name, "w")
    for idx, seq in enumerate(output_results_guided):
        fp_save.write(f">SEQUENCE_{seq_length}_L={seq_length}\n")
        fp_save.write(f"{seq}\n")
    fp_save.close()


if __name__ == "__main__":
    main()
