import os 
import sys 
import torch 
import argparse 

from lang_utils import * 

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from samplers import DiffusionSampler
from prod_smc_samplers import ProductPromptSampler_Seqs
from prod_prompt_samplers import ProductPromptSampler, GeoAvgPromptSampler
from llada_denoiser import LLaDADenoiser
from utils import *
from eval import * 
import lin_reg_utils 

w_vec_master_list = [
        [3., 4.]
]

invalid_err = 1000000.0
device = 'cuda'
scratch_dir = os.getenv('SCRATCH')

hf_cache_dir = os.path.join(scratch_dir, 'huggingface_cache')

if not os.path.exists(hf_cache_dir):
    os.makedirs(hf_cache_dir)


llada_denoiser = LLaDADenoiser(device='cuda')


def gen_prompt_lin_reg(w_vec_tr, num_splits_prod, n_samples, n_features, noise_std, tokenizer):
    X, y = lin_reg_utils.make_lin_reg_dataset(w_vec_tr, n_samples, n_features, noise_std)
        
    ques_setup = "Assume a model of the form y = a * x + b, where a and b are the parameters of the model. The observations are given as (x,y) points, where y has Gaussian noise with standard deviation {} added. ".format(noise_std)
    ans_format = " Output the final answer as: \"The best estimate for parameters of the model are: a = _, and b = _\" where _ is replaced with the values of a and then b."

    # split dataset into num_splits_prod splits
    n_samples_per_split = n_samples // num_splits_prod

    subprompts = []

    for j in range(num_splits_prod):
        X_tr = X[j*n_samples_per_split:(j+1)*n_samples_per_split, :]
        y_tr = y[j*n_samples_per_split:(j+1)*n_samples_per_split]
        
        data_str_j = lin_reg_utils.conv_dataset_to_str(X_tr, y_tr)
        
        subprompt_j = ques_setup + "Predict the parameters of linear regression for (x,y) points: " + data_str_j + ans_format
        subprompts.append(subprompt_j)

    data_joint_str = lin_reg_utils.conv_dataset_to_str(X, y)
    joint_prompt = ques_setup + "Predict the parameters of linear regression for (x,y) points: " + data_joint_str + ans_format

    least_squares = lin_reg_utils.get_least_squares_soln(X, y)
    print("Least squares: ", least_squares)
    
    ls_a = least_squares[0]
    ls_b = least_squares[1]

    return subprompts, joint_prompt, ls_a, ls_b
    

def lin_reg(w_id, seed, num_particles, 
            num_splits_prod, 
            n_samples, 
            n_features=2, 
            noise_std = 1.0,
            remask_strat="low_confidence", savedir="./results/story_gen_prod_random/"):
    

    set_all_seeds(seed)
    num_particles = num_particles


    w_vec_tr = w_vec_master_list[w_id]

    prompt_list, joint_prompt, ls_a, ls_b = gen_prompt_lin_reg(w_vec_tr, 
                                                   num_splits_prod=num_splits_prod, 
                                                   n_samples=n_samples, 
                                                   n_features=n_features, 
                                                   noise_std=noise_std, 
                                                   tokenizer=llada_denoiser.tokenizer)
    
    sampler = DiffusionSampler(llada_denoiser, steps=128, temperature=1.0)
    
    prod_sampler = ProductPromptSampler(llada_denoiser, resample=True, 
                                        adaptive_resampling=False,
                                        steps=128, temperature=1.0)

    # inputs 
    prod_input_ids = llada_denoiser.get_input_ids_template(prompt_list)
    prod_inputs = llada_denoiser.apply_prompt_template(prompt_list)
    print("Prod inputs: ", prod_inputs)
    print("Length of prod inputs: ", len(prod_inputs[0]))

    joint_template = llada_denoiser.apply_prompt_template([joint_prompt])
    jp_input_ids = llada_denoiser.prepare_init_seq([joint_prompt], gen_length=128)

    print("joint_template: ", joint_template)
    print("Length of joint template: ", (jp_input_ids != llada_denoiser.mask_token).sum(dim=-1))
   
    
    print("Decoded input: ", llada_denoiser.tokenizer.batch_decode(prod_input_ids, skip_special_tokens=False))
    print("Input: ", prod_input_ids)

    # Joint Prompt
    out = sampler.sample(init_seq=jp_input_ids, batch_size=1, 
                         remasking= remask_strat, 
                         log_wandb=False)
    #wandb.finish() 

    # remove prompt from output
    out = out[:, jp_input_ids.shape[1]:]
    out_decoded = llada_denoiser.tokenizer.batch_decode(out, skip_special_tokens=True)
    
    # Product prompt 
    prod_out = prod_sampler.sample(prompt_list= prod_input_ids, 
                                   #prompt_weights = torch.tensor([0.0, 2.0, 0.0, 0.0], device='cuda'),
                                         gen_length = 128,
                                         batch_size = 1, 
                                         num_particles = num_particles, 
                                         cfg_scale = 0., 
                                         remasking=remask_strat, 
                                         return_traj = False, 
                                         log_wandb = False,
                                         cut_off_resample = args.cutoff_resample) #110
        

    prod_out_decoded = llada_denoiser.tokenizer.batch_decode(prod_out, skip_special_tokens=True)
    #print("Joint Prompt Output: ", out)
   
    no_sp_joint_template = llada_denoiser.encode_prompt_list(joint_template)
    no_sp_joint_template = llada_denoiser.tokenizer.batch_decode(no_sp_joint_template, skip_special_tokens=True)

    # add template before prod out
    #for i in range(len(prod_out_decoded)):
    #    prod_out_decoded[i] = no_sp_joint_template[0] + prod_out_decoded[i]

    print("Joint prompt output: ", out_decoded)
    print("Product prompt output: ", prod_out_decoded)

    # parse and compute mse
    prod_a, prod_b = parse_a_b(prod_out_decoded[0])
    jp_a, jp_b = parse_a_b(out_decoded[0])

    if jp_a != invalid_err and jp_b != invalid_err:
        #calculate error
        err_joint_a_ = abs(jp_a - ls_a)
        err_joint_b_ = abs(jp_b - ls_b)

    else:
        err_joint_a_ = invalid_err ##-1
        err_joint_b_ = invalid_err ##-1

        incorrect_format_joint = 1
    
    print("\nIncorrect format joint: ", incorrect_format_joint)

    err_prod_a_ = abs(prod_a - ls_a)
    err_prod_b_ = abs(prod_b - ls_b)

    print("Product MSE a: ", err_prod_a_)
    print("Product MSE b: ", err_prod_b_)
    print("Joint MSE a: ", err_joint_a_)
    print("Joint MSE b: ", err_joint_b_)

    mse_prod = (err_prod_a_**2 + err_prod_b_**2)
    mse_joint = (err_joint_a_**2 + err_joint_b_**2)

    print("Product MSE total: ", np.sqrt(mse_prod))
    print("Joint MSE total: ", np.sqrt(mse_joint))

    save_path = "./results/lin_reg_results_redo_{}_w_id_{}/".format(remask_strat, w_id) #savedir
    os.makedirs(save_path, exist_ok=True)

    # save joint output 
    with open(os.path.join(save_path, f"jp_seed{seed}.txt"), 'w') as f:
        f.write("Prompt: \n")
        f.write(joint_prompt + "\n\n")
        f.write("Output: \n")
        f.write(out_decoded[0] + "\n\n")
        #f.write(f"PPL: {jp_ppl}\n")

    # save product output
    for i in range(len(prod_out_decoded)):
        with open(os.path.join(save_path, f"prod_seed{seed}_particle_num_{i}_of_{num_particles}.txt"), 'w') as f:
            for j in range(len(prompt_list)):
                f.write(f"Prompt {j}: {prompt_list[j]}\n")
            f.write("\n\n")
            f.write("Output: \n")
            f.write(f"Particle {i}:\n")
            f.write(prod_out_decoded[i] + "\n\n")
            #f.write(f"PPL: {prod_ppl[i]}\n")

    return err_prod_a_, err_prod_b_, err_joint_a_, err_joint_b_, incorrect_format_joint


def main(args):
    w_id = args.w_id
    seed = args.seed
    num_particles = args.num_particles
    remask_strat = args.remask_strat
    savedir = args.savedir

    num_splits_prod = args.num_splits_prod 
    n_samples = args.n_samples
    n_features = args.n_features
    noise_std = args.noise_std

    error_prod_a, error_prod_b, error_joint_a, error_joint_b, incorrect_format_joint = lin_reg(w_id, 
                                                                       seed, 
                                                                       num_particles,
                                                                       num_splits_prod,
                                                                       n_samples,
                                                                       n_features,
                                                                       noise_std,
                                                                       remask_strat,
                                                                       savedir)

    return 

"""
def main_over_n_samples(args):
    seeds = [12, 13, 14, 15, 16]
    num_particles_list = [1, 4, 8]
    n_samples_list = [10, 20, 50, 100]

    all_results = {}

    for n_samples in n_samples_list:
        all_results[n_samples] = {}
        for num_particles in num_particles_list:
            all_results[n_samples][num_particles] = {
                'prod_a_err': [],
                'prod_b_err': [],
                'joint_a_err': [],
                'joint_b_err': [],
                'joint_incorrect_format': []
            }
            for seed in seeds:
                err_prod_a, err_prod_b, err_joint_a, err_joint_b, incorrect_format_joint = lin_reg(w_id=0, 
                                                                                                   seed=seed, 
                                                                                                   num_particles=num_particles,
                                                                                                   num_splits_prod=5,
                                                                                                   n_samples=n_samples,
                                                                                                   n_features=2,
                                                                                                   noise_std=1.0,
                                                                                                   remask_strat="low_confidence",
                                                                                                   savedir="./results/story_gen_prod_random/")
                
                l2_err_joint_seed = np.sqrt(np.array(err_joint_a_all)**2 + np.array(err_joint_b_all)**2)
                l2_err_prod_seed = np.sqrt(np.array(err_prod_a_all)**2 + np.array(err_prod_b_all)**2)

                all_results[n_samples][num_particles]['prod_a_err'].append(err_prod_a)
                all_results[n_samples][num_particles]['prod_b_err'].append(err_prod_b)
                all_results[n_samples][num_particles]['joint_a_err'].append(err_joint_a)
                all_results[n_samples][num_particles]['joint_b_err'].append(err_joint_b)
                all_results[n_samples][num_particles]['joint_incorrect_format'].append(incorrect_format_joint)

            print(f"\n\nn_samples: {n_samples}, num_particles: {num_particles}")
            print("Product a err: ", all_results[n_samples][num_particles]['prod_a_err'])
            print("Product b err: ", all_results[n_samples][num_particles]['prod_b_err'])
            print("Joint a err: ", all_results[n_samples][num_particles]['joint_a_err'])
            print("Joint b err: ", all_results[n_samples][num_particles]['joint_b_err'])
            print("Joint incorrect format: ", all_results[n_samples][num_particles]['joint_incorrect_format'])

            print("Avg Product a err: ", np.mean(all_results[n_samples][num_particles]['prod_a_err']))
            print("Avg Product b err: ", np.mean(all_results[n_samples][num_particles]['prod_b_err']))
"""

def parse_args():
    parser = argparse.ArgumentParser(description="ESM2 Reference-based Reward Experiment")
    parser.add_argument("--w_id", type=int, default=0, help="ID of the gt parameters to use")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--num_particles", type=int, default=5, help="Number of particles for guided sampling")
    parser.add_argument("--remask_strat", type=str, default="random", help="Remasking strategy: random or low_confidence")
    parser.add_argument("--savedir", type=str, default="./results/story_gen_prod_random/", help="Directory to save results")
    
    parser.add_argument("--num_splits_prod", type=int, default=5, help="Number of splits for product prompt")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of samples in dataset")
    parser.add_argument("--n_features", type=int, default=2, help="Number of features in dataset")
    parser.add_argument("--noise_std", type=float, default=0.1, help="Standard deviation of noise in dataset")

    parser.add_argument("--cutoff_resample", type=int, default=128, help="Timestep to stop resampling at")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)

    exit() 