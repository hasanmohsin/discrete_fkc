import os 
import sys 
import torch 
import argparse 

from datasets import load_dataset
import evaluate 

os.environ["HF_ALLOW_CODE_EVAL"] = "1"


from code_utils import * 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samplers import DiffusionSampler
from prod_prompt_samplers import ProductPromptSampler, GeoAvgPromptSampler
from smc_sampler import AnnealSampler
from llada_denoiser import LLaDADenoiser, LLaDAMOEDenoiser
from utils import *
from eval import * 


def human_eval_load():
    scratch_dir = os.getenv('SCRATCH')
    hf_cache_dir = os.path.join(scratch_dir, 'huggingface_cache')

    data_dir = os.path.join(scratch_dir, 'code_data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataset = load_dataset("openai/openai_humaneval", cache_dir=hf_cache_dir)

    return dataset


def mbpp_load():
    scratch_dir = os.getenv('SCRATCH')
    hf_cache_dir = os.path.join(scratch_dir, 'huggingface_cache')

    data_dir = os.path.join(scratch_dir, 'code_data')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataset = load_dataset("mbpp", cache_dir=hf_cache_dir)

    return dataset


def parse_unit_tests(unit_test_str, entry_point):
    # unit_test_list is a list of strings, each string is a unit test
    # solution_code is a string, the code to be tested
    # replace candidate with entry point in unit_test_str
    unit_test_str = unit_test_str.replace("candidate", entry_point)
    # Split the body of `def check(...)` into individual assert statements (as strings).
    lines = unit_test_str.splitlines()
    asserts = []
    in_check = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        # Detect start of the check function
        if not in_check and stripped.startswith("def ") and "check" in stripped and stripped.endswith(":"):
            in_check = True
            continue
        if in_check:
            # Stop if we reach another top-level definition or non-indented block
            if line and not line.startswith((" ", "\t")) and not stripped.startswith("assert") and stripped != "":
                break
            if stripped.startswith("assert"):
                asserts.append(stripped)

    return asserts

def evaluate_code_unit_tests(unit_tests, solution_code):
    code_eval = evaluate.load("code_eval")

    print("unit tests: ", unit_tests)
    preds = []

    for ut in unit_tests:
        preds.append([solution_code])

    pass_at_k, results = code_eval.compute(references=unit_tests, predictions=preds, k=[1])

    print(pass_at_k)
    print(results)

    return pass_at_k

# extract the function definition from the code output (name specified by entry_point)
def parse_code_out(out_txt, entry_point, occurrence=2):
    lines = out_txt.splitlines()
    func_lines = []
    in_func = False
    indent_level = None
    match_count = 0

    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith(f"def {entry_point}("):
            match_count += 1
            if match_count == occurrence:
                in_func = True
                indent_level = len(line) - len(stripped)
                func_lines.append(line)
                continue
            # otherwise keep scanning for the desired occurrence
            continue
        if in_func:
            current_indent = len(line) - len(stripped)
            if stripped == "" or current_indent > indent_level:
                func_lines.append(line)
            else:
                break

    # fallback: if desired occurrence not found but at least one match exists, return the first occurrence
    if not func_lines and match_count > 0:
        in_func = False
        indent_level = None
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith(f"def {entry_point}("):
                in_func = True
                indent_level = len(line) - len(stripped)
                func_lines.append(line)
                continue
            if in_func:
                current_indent = len(line) - len(stripped)
                if stripped == "" or current_indent > indent_level:
                    func_lines.append(line)
                else:
                    break

    return "\n".join(func_lines)

def process_sample(sample, dataset):
    # for human eval, set 'test_list'
    if dataset == "human_eval":
        sample['test_list'] = parse_unit_tests(sample['test'], sample['entry_point'])
        sample['base_prompt'] = "# Write a function with the following specification: \n" + sample['prompt']
    # set entry_point, and 'prompt'
    elif dataset == "mbpp":
        sample['prompt'] = sample['text']

        # get entry point from sample code
        code_lines = sanitize.sanitize(sample['code'])

        # get def .... ( ... ): line
        for line in code_lines.splitlines():
            line = line.strip()
            if line.startswith("def "):
                entry_point = line.split("(")[0].replace("def ", "").strip()
                break

        sample['entry_point'] = entry_point

        sample['base_prompt'] = sample['prompt'] + " The function should have name: {}".format(entry_point)

    return sample

def code_gen_joint_only(moe, dataset, num_datapoints, seed, joint_temp, remask_strat="low_confidence", savedir="./results/human_eval_out_joint_random/", cutoff_resample=None):

    set_all_seeds(seed)

    if moe:
        llada_denoiser = LLaDAMOEDenoiser(device='cuda')
    else:
        llada_denoiser = LLaDADenoiser(device='cuda')
    steps = 128
    gen_length = steps

    sampler = DiffusionSampler(llada_denoiser, steps=steps, 
                               temperature=joint_temp)
   

    if dataset == "human_eval":
        total_data = human_eval_load()
    elif dataset == "mbpp":
        total_data = mbpp_load()

    for i in range(num_datapoints):
        sample = total_data['test'][i]
        sample = process_sample(sample, dataset)

        prompt_pre = "# Write a function with the following specification: \n"

        base_prompt = prompt_pre + sample['prompt'] #+ "\n#The function should pass the following test(s):"
        
        cond_list = sample['test_list']
        num_conditions = len(cond_list)

        joint_prompt = base_prompt + "\n".join(cond_list)

        prompt_list = [base_prompt + "\n" + cond for cond in cond_list]

        len(prompt_list)

        # inputs 
        prod_input_ids = llada_denoiser.get_input_ids_template(prompt_list)
        prod_inputs = llada_denoiser.apply_prompt_template(prompt_list)
        print("Prod inputs: ", prod_inputs)
        print("Length of prod inputs: ", len(prod_inputs[0]))

        joint_template = llada_denoiser.apply_prompt_template([joint_prompt])
        jp_input_ids = llada_denoiser.prepare_init_seq([joint_prompt], gen_length=gen_length)

        print("joint_template: ", joint_template)
        print("Length of joint template: ", (jp_input_ids != llada_denoiser.mask_token).sum(dim=-1))
        
        print("Decoded input: ", llada_denoiser.tokenizer.batch_decode(prod_input_ids, skip_special_tokens=False))
        print("Input: ", prod_input_ids)

        # Joint Prompt
        out = sampler.sample(init_seq=jp_input_ids, batch_size=1, 
                            remasking= remask_strat, 
                            log_wandb=False)
        #wandb.finish() 

        out_decoded = llada_denoiser.tokenizer.batch_decode(out, skip_special_tokens=True)

        out_after_prompt = out_decoded[0].replace(joint_template[0], "")

        print("Out after prompt removal: ", out_after_prompt)

        code_out = sanitize.sanitize(text=out_after_prompt, entrypoint = sample['entry_point'])

        print("Extracted Code Output: ", code_out)

        evals = evaluate_code_unit_tests(cond_list, code_out)

        print("Success: ", evals)

        print("Joint prompt output: ", out_decoded)

        save_path = savedir+"joint_only_{}_len_{}_point_{}/".format(remask_strat, num_conditions, i) #savedir
        os.makedirs(save_path, exist_ok=True)

        # save joint output 
        with open(os.path.join(save_path, f"jp_seed{seed}_point_{i}.txt"), 'w') as f:
            f.write("Prompt: \n")
            f.write(joint_prompt + "\n\n")
            f.write("Output: \n")
            f.write(out_decoded[0] + "\n\n")
            f.write("Unit tests success: \n")
            #f.write(f"PPL: {jp_ppl}\n")

    return 



def code_gen(moe, dataset, num_datapoints, seed, beta, num_particles, remask_strat="low_confidence", savedir="./results/human_eval_out_prod_random/", cutoff_resample=None, compute_ppl = False):
    set_all_seeds(seed)

    if moe:
        llada_denoiser = LLaDAMOEDenoiser(device='cuda')
    else:
        llada_denoiser = LLaDADenoiser(device='cuda')

    steps = 128
    gen_length = steps

    naive_temp = 1/beta


    base_eval_list = []
    anneal_eval_list = []

    sampler = DiffusionSampler(llada_denoiser, steps=steps, 
                               temperature=naive_temp)
   
    anneal_sampler = AnnealSampler(denoiser = llada_denoiser, 
                                   beta = beta,
                                   resample = False,
                                   adaptive_resampling=False, 
                                   steps=steps, 
                                   temperature=1.0)

    print("\ndataset: ", dataset)

    if dataset == "human_eval":
        total_data = human_eval_load()
    elif dataset == "mbpp":
        total_data = mbpp_load()

    # -1 means all datapoints are evaluated on

    if num_datapoints == -1 or num_datapoints > len(total_data['test']):
        num_datapoints = len(total_data['test'])

    print(f"Evaluating on {num_datapoints} out of {len(total_data['test'])} datapoints.")
    

    for p in range(num_datapoints):
        sample = total_data['test'][p]

        sample = process_sample(sample, dataset)

        prompt_pre = "Write a function with the following specification: \n"

        base_prompt = sample['base_prompt'] #prompt_pre + sample['prompt'] #+ "\nThe function should pass the following test(s):"
        
        cond_list = sample['test_list']
        num_conditions = len(cond_list)

        prompt = base_prompt #+ "\n".join(cond_list)
         

        template = llada_denoiser.apply_prompt_template([prompt])
        input_ids = llada_denoiser.prepare_init_seq([prompt], gen_length=gen_length)
     

        # Joint Prompt
        out = sampler.sample(init_seq=input_ids, batch_size=1, 
                            remasking= remask_strat, 
                            log_wandb=False)
        #wandb.finish() 

        out_decoded = llada_denoiser.tokenizer.batch_decode(out, skip_special_tokens=True)

        code_out = sanitize.sanitize(text = out_decoded[0], entrypoint = sample['entry_point'])
       
        print("\n Naive Anneal Extracted Code Output: ", code_out)

        evals = evaluate_code_unit_tests(cond_list, code_out)

        # Product Prompt
        if cutoff_resample < -1:
            cutoff_resample = None
        
        anneal_input_ids = input_ids.reshape((input_ids.shape[0],  1, input_ids.shape[1])).repeat((1, num_particles, 1))

        print("anneal_input_ids shape: ", anneal_input_ids.shape)
        #print("anneal_input_ids: ", anneal_input_ids)
        #print("sampler length: ", anneal_sampler.length)

        # Annealed prompt 
        anneal_sampler.length = input_ids.shape[-1]
        anneal_out = anneal_sampler.sample(init_seq=anneal_input_ids, 
                                            batch_size = 1, 
                                            num_particles = num_particles, 
                                            cfg_scale = 0., 
                                            remasking=remask_strat, 
                                            return_traj = False, 
                                            log_wandb = False) #110

        anneal_out_decoded = llada_denoiser.tokenizer.batch_decode(anneal_out, skip_special_tokens=True)
        #print("Joint Prompt Output: ", out)
   
        no_sp_template = llada_denoiser.encode_prompt_list(template)
        no_sp_template = llada_denoiser.tokenizer.batch_decode(no_sp_template, skip_special_tokens=True)

        # add template before prod out
        for i in range(len(anneal_out_decoded)):
            anneal_out_decoded[i] = no_sp_template[0] + anneal_out_decoded[i]

        # sample one from random particles to evaluate
        anneal_code_out = sanitize.sanitize(text = anneal_out_decoded[0], entrypoint = sample['entry_point'])
        # parse_code_out(prod_out_decoded[0], sample['entry_point'])

        print("Extracted Anneal Code Output: ", anneal_code_out)

        anneal_evals = evaluate_code_unit_tests(cond_list, anneal_code_out)

        print("\n\nNaive anneal success: ", evals)
        print("Anneal Success: ", anneal_evals)

        base_eval_list.append(evals['pass@1'])
        anneal_eval_list.append(anneal_evals['pass@1'])

        save_path = savedir+"code_out_{}_beta_{}_point_{}/".format(remask_strat, anneal_sampler.beta, p) #savedir
        os.makedirs(save_path, exist_ok=True)

        # save joint output 
        with open(os.path.join(save_path, f"base_temp_seed{seed}_point_{p}_pass_{evals['pass@1']}.txt"), 'w') as f:
            f.write("Prompt: \n")
            f.write(prompt + "\n\n")
            f.write("Output: \n")
            f.write(out_decoded[0] + "\n\n")
            

        # save product output
        for i in range(len(anneal_out_decoded)):
            with open(os.path.join(save_path, f"anneal_seed{seed}_particle_num_{i}_of_{num_particles}_beta_{anneal_sampler.beta}_pass_{anneal_evals['pass@1']}.txt"), 'w') as f:
                f.write("Prompt: \n")
                f.write(prompt + "\n\n")
                f.write("\n\n")
                f.write("Output: \n")
                f.write(f"Particle {i}:\n")
                f.write(anneal_out_decoded[i] + "\n\n")
              
    return base_eval_list, anneal_eval_list

def main(args):
    num_pts = args.num_points
    seed = args.seed
    num_particles = args.num_particles
    remask_strat = args.remask_strat
    savedir = args.savedir
    dataset = args.dataset
    moe = args.llada_moe

    if args.only_joint:
        base_acc = code_gen_joint_only(moe, dataset, num_pts, seed, args.joint_temp, remask_strat, savedir, cutoff_resample = args.cutoff_resample)
    else:
        base_acc, anneal_acc = code_gen(moe, dataset, num_pts, seed, args.beta, num_particles, remask_strat, savedir, cutoff_resample = args.cutoff_resample)

    print("\n\nAverage Naive Temp Accuracy over {} points: ".format(num_pts), np.array(base_acc).mean())
    print("Average SMC Anneal Accuracy over {} points: ".format(num_pts), np.array(anneal_acc).mean())

    return 

def parse_args():
    parser = argparse.ArgumentParser(description="Human Eval Code Experiment")
    parser.add_argument("--dataset", type=str, default="human_eval", choices=["human_eval", "mbpp"], help="Dataset to use: human_eval or mbpp")
    parser.add_argument("--num_points", type=int, default=1, help="Number of datapoints to evaluate on")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_particles", type=int, default=5, help="Number of particles for guided sampling")
    parser.add_argument("--remask_strat", type=str, default="random", help="Remasking strategy: random or low_confidence")
    parser.add_argument("--savedir", type=str, default="./results/human_eval_code_gen_prod_random/", help="Directory to save results")
    parser.add_argument("--cutoff_resample", type=int, default=-1, help="Timestep to stop resampling at")
    parser.add_argument("--only_joint", action='store_true', help="Only generate joint prompt")
    parser.add_argument("--joint_temp", type=float, default=1.0, help="Temperature for joint prompt sampling")
    parser.add_argument('--beta', type=float, default=1.0, help='Beta value for annealing sampler')
    parser.add_argument('--llada_moe', action='store_true', help='Use Mixture of Experts LLaDA model')
    args = parser.parse_args()

    args.savedir = args.savedir.replace("human_eval", args.dataset)
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

    exit()