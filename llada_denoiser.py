import os
import sys
import math 
import torch 
from transformers import AutoTokenizer, AutoModel

# this tokenizer gives warning about parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add the parent directory to Python path to access dplm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from denoisers import Denoiser 

class LLaDADenoiser(Denoiser):

    def __init__(self, device = 'cuda', save_to_hf_cache = True):
        self.device = device

        if save_to_hf_cache:
            scratch_dir = os.getenv('SCRATCH')
            self.hf_cache_dir = os.path.join(scratch_dir, 'huggingface_cache')
            
            if not os.path.exists(self.hf_cache_dir):
                os.makedirs(self.hf_cache_dir)
        else:
            self.hf_cache_dir = None

        self.name = "LLaDA-8B-Instruct"
        self.model = AutoModel.from_pretrained('GSAI-ML/'+self.name, cache_dir = self.hf_cache_dir, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
        self.tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/'+self.name, cache_dir = self.hf_cache_dir, trust_remote_code=True)
        self.model.to(self.device).eval()
        
        self.mask_token = 126336
        self.length = None # works with flexible lengths, depending on input_seq in sampling


        self.space_token_id = self.tokenizer.encode(" ", add_special_tokens=False)[0]
        self.tokenizer.pad_token_id = self.space_token_id 
        self.tokenizer.padding_side = "left"  # for right padding

    def _set_length(self, length):
        self.length = length
        return

    # llada expects this prompt format for Instruct model 
    def apply_prompt_template(self, prompt_list):

        if "Instruct" not in self.name:
            return prompt_list
        
        # process for instruct model 
        new_prompts = []
        
        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        for p in prompt_list:
            m = [{"role": "user", "content": p}, ]
            new_prompt = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            new_prompts.append(new_prompt)

        return new_prompts

    def encode_prompt_list(self, prompt_list):
        input_ids = self.tokenizer(prompt_list, padding=True)['input_ids']

        input_ids = torch.tensor(input_ids).to(self.device)

        return input_ids 

    # adds length number of 
    def add_mask_tokens(self, input_ids):
        x = torch.full((input_ids.shape[0], input_ids.shape[1] + self.length), self.mask_token, dtype=torch.long).to(self.device)
        x[:, :input_ids.shape[1]] = input_ids.clone()
        return x

    def get_input_ids_template(self, prompt_list):
        prompt_list = self.apply_prompt_template(prompt_list)
        input_ids = self.encode_prompt_list(prompt_list)

        return input_ids

    # apply template, encode and add mask tokens of length self.length
    def prepare_init_seq(self, prompt_list, gen_length):
        self._set_length(gen_length)

        prompt_list = self.apply_prompt_template(prompt_list)
        input_ids = self.encode_prompt_list(prompt_list)
        init_seq = self.add_mask_tokens(input_ids)

        return init_seq

    # outputs logits, input seq is assumed to be tokenized 
    def __call__(self, input_seq):
        # Implement the denoising process using the DPLM model
        net_out = self.model(
            input_seq
        )

        logits = net_out.logits  # shape (batch_size, seq_len, vocab_size

        # logits over clean (non-mask) data (incl special tokens)
        return logits 
    


class LLaDAMOEDenoiser(LLaDADenoiser):

    def __init__(self, device ='cuda', save_to_hf_cache = True):
        super().__init__(device, save_to_hf_cache)
        self.name = 'LLaDA-MoE-7B-A1B-Instruct'
        self.model = AutoModel.from_pretrained('inclusionAI/'+ self.name, 
                                               cache_dir = self.hf_cache_dir, 
                                               trust_remote_code=True, 
                                               torch_dtype=torch.bfloat16)
        
        self.tokenizer = AutoTokenizer.from_pretrained('inclusionAI/'+self.name, 
                                                       cache_dir = self.hf_cache_dir, 
                                                       trust_remote_code=True)
        self.model.to(self.device).eval()

        self.mask_token = 156895

    def apply_prompt_template(self, prompt_list):

        if "Instruct" not in self.name:
            return prompt_list
        
        # process for instruct model 
        new_prompts = []
        
        # Add special tokens for the Instruct model. The Base model does not require the following two lines.
        for p in prompt_list:
            m = [{"role": "system", "content": "You are a helpful AI assistant."}, 
                 {"role": "user", "content": p}]
            new_prompt = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
            new_prompts.append(new_prompt)

        return new_prompts