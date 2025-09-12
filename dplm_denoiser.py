import os
import sys
import math 

# Add the parent directory to Python path to access dplm
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from byprot.models.dplm import DiffusionProteinLanguageModel as DPLM
from dplm.generate_dplm import initialize_generation

from denoisers import Denoiser 

class DPLMDenoiser(Denoiser):

    def __init__(self, device = 'cuda'):
        self.device = device
        self.dplm = DPLM.from_pretrained("airkingbd/dplm_650m").to(device)
        self.mask_token = self.dplm.mask_id
        self.length = None # works with flexible lengths, depending on input_seq in sampling
        self.name = "DPLM1Denoiser"

    # outputs logits, input seq is assumed to be tokenized 
    def __call__(self, input_seq):
        # Implement the denoising process using the DPLM model
        net_out = self.dplm.net(
            input_ids=input_seq,
        )

        logits = net_out["logits"]

        # logits have a value for the mask token as well (not just clean data)
        logits[..., self.dplm.mask_id] = -math.inf
        logits[..., self.dplm.x_id] = -math.inf
        logits[..., self.dplm.pad_id] = -math.inf
        logits[..., self.dplm.bos_id] = -math.inf
        logits[..., self.dplm.eos_id] = -math.inf

        # cut out logit value for mask id (which is last one according to dplm vanilla)
        if self.dplm.mask_id == logits.shape[-1] - 1:
            logits = logits[..., :self.dplm.mask_id]

        # logits over clean (non-mask) data (incl special tokens)
        return logits 