from transformers import AutoModelForCausalLM, AutoTokenizer
import os 
from evaluate import load

import torch 

class QwenPPL():
    def __init__(self, model_name=None, task = "text"):

        #scratch_dir = os.getenv('SCRATCH')
        #hf_cache_dir = os.path.join(scratch_dir, 'huggingface_cache')

        self.task = task

        if model_name is not None:
            self.model_name = model_name
        # if model name not specified use based on task (Qwen2.5 variants, 3B size)
        else:
            if self.task == "text":
                self.model_name = "Qwen/Qwen2.5-3B-Instruct"
            elif self.task == "code":
                self.model_name = "Qwen/Qwen2.5-Coder-3B-Instruct"

        self.perplexity = load("perplexity", module_type="metric")

    # text is a list of decoded strings
    # returns a list of perplexity values for each string
    def __call__(self, text):
        result = self.perplexity.compute(model_id=self.model_name,
                                        add_start_token=False,
                                        predictions=text) 

        return result['perplexities']
    
# computes likelihood under the target log probs, of the given samples
class TargetLikelihood():
    def __init__(self, log_prob_target):
        self.log_prob_target = log_prob_target

        # if its a tensor, normalize it to be a log prob distribution
        if isinstance(self.log_prob_target, torch.Tensor):
            self.log_prob_target = self.log_prob_target - torch.logsumexp(self.log_prob_target.flatten(), dim=-1)


    def __call__(self, samples):
        log_probs = self.log_prob_target(samples)
        return -log_probs.mean()