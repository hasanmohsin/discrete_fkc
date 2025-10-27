import os 
import sys 
import torch 
import argparse 

from lang_utils import * 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samplers import DiffusionSampler
from prod_smc_samplers import ProductPromptSampler
from prod_new_samplers import NewProductPromptSampler, GeoAvgPromptSampler
from llada_denoiser import LLaDADenoiser
from utils import *
from eval import * 
from cond_bank_story import cond_bank


def story_gen_joint_only(num_cond, seed, joint_temp, remask_strat="low_confidence", savedir="./results/story_gen_prod_random/", cutoff_resample=None, compute_ppl = False, ppl_dir = "./results/story_gen_prod_ppl_pickles/"):

    set_all_seeds(seed)
    llada_denoiser = LLaDADenoiser(device='cuda')

    # randomly sample num_cond conditions from cond_bank
    # with replacement
    cond_idx = random.sample(range(len(cond_bank)), num_cond)
    num_conditions = len(cond_idx)
    base_prompt = "Write a story."
    conds = [cond_bank[i] for i in cond_idx]

    
    if num_cond <= 6:
        steps = 128
    elif num_cond <= 30:
        steps = 256
    elif num_cond <= 50:
        steps = 512
    gen_length = steps

    joint_prompt = base_prompt + "".join(conds)

    prompt_list = [base_prompt + cond for cond in conds]
    

    sampler = DiffusionSampler(llada_denoiser, steps=steps, 
                               temperature=joint_temp)
   

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

    print("Joint prompt output: ", out_decoded)

    jp_ppl = -1

    save_path = "./results/story_gen_randomized_prod_joint_only_{}_len_{}/".format(remask_strat, num_conditions) #savedir
    os.makedirs(save_path, exist_ok=True)

    # save joint output 
    with open(os.path.join(save_path, f"jp_seed{seed}_ppl_{jp_ppl:.2f}.txt"), 'w') as f:
        f.write("Prompt: \n")
        f.write(joint_prompt + "\n\n")
        f.write("Output: \n")
        f.write(out_decoded[0] + "\n\n")
        #f.write(f"PPL: {jp_ppl}\n")

    return jp_ppl



def story_gen(num_cond, seed, num_particles, remask_strat="low_confidence", savedir="./results/story_gen_prod_random/", cutoff_resample=None, compute_ppl = False, ppl_dir = "./results/story_gen_prod_ppl_pickles/"):

    set_all_seeds(seed)
    num_particles = num_particles

    llada_denoiser = LLaDADenoiser(device='cuda')

    # randomly sample num_cond conditions from cond_bank
    # with replacement
    cond_idx = random.sample(range(len(cond_bank)), num_cond)
    num_conditions = len(cond_idx)
    base_prompt = "Write a story."
    conds = [cond_bank[i] for i in cond_idx]

    
    if num_cond <= 6:
        steps = 128
    elif num_cond <= 30:
        steps = 256
    elif num_cond <= 50:
        steps = 512
    gen_length = steps

    joint_prompt = base_prompt + "".join(conds)

    prompt_list = [base_prompt + cond for cond in conds]
    

    sampler = DiffusionSampler(llada_denoiser, steps=steps, temperature=1.0)
    
    prod_sampler = NewProductPromptSampler(llada_denoiser, resample=True, 
                                        adaptive_resampling=False,
                                        steps=steps, temperature=1.0)

    #prod_sampler = GeoAvgPromptSampler(llada_denoiser, resample=True, 
    #                                    adaptive_resampling=False,
    #                                    steps=128, temperature=1.0)


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
   

    out_decoded = llada_denoiser.tokenizer.batch_decode(out, skip_special_tokens=True)


    if cutoff_resample < -1:
        cutoff_resample = None
    
    # Product prompt 
    prod_out = prod_sampler.sample(prompt_list= prod_input_ids, 
                                   #prompt_weights = torch.tensor([0.0, 2.0, 0.0, 0.0], device='cuda'),
                                         gen_length = gen_length,
                                         batch_size = 1, 
                                         num_particles = num_particles, 
                                         cfg_scale = 0., 
                                         remasking=remask_strat, 
                                         return_traj = False, 
                                         log_wandb = False,
                                         cut_off_resample = cutoff_resample) #110
        

    prod_out_decoded = llada_denoiser.tokenizer.batch_decode(prod_out, skip_special_tokens=True)
    #print("Joint Prompt Output: ", out)
   
    no_sp_joint_template = llada_denoiser.encode_prompt_list(joint_template)
    no_sp_joint_template = llada_denoiser.tokenizer.batch_decode(no_sp_joint_template, skip_special_tokens=True)

    # add template before prod out
    for i in range(len(prod_out_decoded)):
        prod_out_decoded[i] = no_sp_joint_template[0] + prod_out_decoded[i]

    print("Joint prompt output: ", out_decoded)
    print("Product prompt output: ", prod_out_decoded)

    if compute_ppl:
        #Perplexities under Qwen
        ppl = QwenPPL(task='text')
        jp_ppl = ppl(out_decoded)[0]
        prod_ppl = ppl(prod_out_decoded)

        print("Joint Prompt PPL: ", jp_ppl)
        print("Product Prompt PPL: ", prod_ppl)
    else:
        jp_ppl = -1
        prod_ppl = [-1.0 for _ in range(len(prod_out_decoded))]


    save_path = "./results/story_gen_randomized_prod_{}_len_{}/".format(remask_strat, num_conditions) #savedir
    os.makedirs(save_path, exist_ok=True)

    # save joint output 
    with open(os.path.join(save_path, f"jp_seed{seed}_ppl_{jp_ppl:.2f}.txt"), 'w') as f:
        f.write("Prompt: \n")
        f.write(joint_prompt + "\n\n")
        f.write("Output: \n")
        f.write(out_decoded[0] + "\n\n")
        #f.write(f"PPL: {jp_ppl}\n")

    # save product output
    for i in range(len(prod_out_decoded)):
        with open(os.path.join(save_path, f"prod_seed{seed}_particle_num_{i}_of_{num_particles}_ppl_{prod_ppl[i]:.2f}.txt"), 'w') as f:
            for j in range(len(prompt_list)):
                f.write(f"Prompt {j}: {prompt_list[j]}\n")
            f.write("\n\n")
            f.write("Output: \n")
            f.write(f"Particle {i}:\n")
            f.write(prod_out_decoded[i] + "\n\n")
            #f.write(f"PPL: {prod_ppl[i]}\n")

    return jp_ppl, prod_ppl[0]


def main_old():
    seeds = [21] #, 32] #[12, 13, 14, 15, 16]
    joint_ppl_list = []
    prod_ppl_list = []
    
    for seed in seeds:
        jp_ppl, prod_ppl = story_gen(seed)

        print(f"\n\nSeed: {seed}, Joint Prompt PPL: {jp_ppl}, Product Prompt PPL: {prod_ppl}")

        joint_ppl_list.append(jp_ppl)
        prod_ppl_list.append(prod_ppl)

    joint_ppl_list = np.array(joint_ppl_list)
    prod_ppl_list = np.array(prod_ppl_list)

    print(f"\n\nAverage Joint Prompt PPL: {np.mean(joint_ppl_list)}, Average Product Prompt PPL: {np.mean(prod_ppl_list)}")
    print(f"Std Joint Prompt PPL: {np.std(joint_ppl_list)}, Std Product Prompt PPL: {np.std(prod_ppl_list)}")
    return

def main(args):
    num_cond = args.num_cond
    seed = args.seed
    num_particles = args.num_particles
    remask_strat = args.remask_strat
    savedir = args.savedir

    if args.only_joint:
        jp_ppl = story_gen_joint_only(num_cond, seed, args.joint_temp, remask_strat, savedir, cutoff_resample = args.cutoff_resample)
    else:
        jp_ppl, prod_ppl = story_gen(num_cond, seed, num_particles, remask_strat, savedir, cutoff_resample = args.cutoff_resample)

    return 

def parse_args():
    parser = argparse.ArgumentParser(description="ESM2 Reference-based Reward Experiment")
    parser.add_argument("--num_cond", type=int, default=4, help="Number of conditions to use")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_particles", type=int, default=5, help="Number of particles for guided sampling")
    parser.add_argument("--remask_strat", type=str, default="random", help="Remasking strategy: random or low_confidence")
    parser.add_argument("--savedir", type=str, default="./results/story_gen_prod_random/", help="Directory to save results")
    parser.add_argument("--cutoff_resample", type=int, default=-1, help="Timestep to stop resampling at")
    parser.add_argument("--only_joint", action='store_true', help="Only generate joint prompt")
    parser.add_argument("--joint_temp", type=float, default=1.0, help="Temperature for joint prompt sampling")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

    exit() 