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

class ESM2ProperLikelihoodProteinReward():
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

    # use the transform described in Algorithm 2 of https://www.biorxiv.org/content/10.1101/2024.10.03.616542v1 
    # to calculate pseudo-likelihood when model is passed unmasked sequence
    def proper_llhd_transform(self, logits):
        alpha = 0.1 
        beta = 0.1 
        probs = torch.softmax(logits, dim=-1)

        transformed_probs = ((alpha + beta) / alpha) * probs - beta / alpha
        transformed_probs = torch.clamp(transformed_probs, min=1e-10)
        log_transformed_probs = torch.log(transformed_probs)

        return log_transformed_probs

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
            
            # Apply proper likelihood transformation
            log_probs = self.proper_llhd_transform(log_probs)

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
    

            
class ESM2ProperLikelihoodProteinRewardReference():
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

    # use the transform described in Algorithm 2 of https://www.biorxiv.org/content/10.1101/2024.10.03.616542v1 
    # to calculate pseudo-likelihood when model is passed unmasked sequence
    def proper_llhd_transform(self, logits):
        alpha = 0.1 
        beta = 0.1 
        probs = torch.softmax(logits, dim=-1)

        transformed_probs = ((alpha + beta) / alpha) * probs - beta / alpha
        transformed_probs = torch.clamp(transformed_probs, min=1e-10)
        log_transformed_probs = torch.log(transformed_probs)

        return log_transformed_probs

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

            # Apply proper likelihood transformation
            log_probs = self.proper_llhd_transform(log_probs)

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
                log_rewards.append(- self.beta * 10.0)  # Fallback score

        # Convert to tensor and ensure same device as input
        log_rewards = torch.tensor(
            log_rewards, dtype=torch.float32, device=self.device)

        return log_rewards