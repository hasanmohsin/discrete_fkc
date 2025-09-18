import os
import random 
import numpy as np
import torch 
import matplotlib.pyplot as plt

import wandb 

def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_wandb_run(config, project="discrete_fkc"):
    # check number of particles, if not in config set to 1
    if 'num_particles' not in config:
        config['num_particles'] = 1

    if 'denoiser_name_2' in config:
        run_name = f"{config['denoiser_name']}_prod_{config['denoiser_name_2']}"
    else:
        run_name = f"{config['denoiser_name']}"
    
    if 'reward' in config:
        run_name = run_name + f"{config['denoiser_name']}_reward_{config['reward']}"

    run_name = run_name + f"_sampler_{config['sampler']}"
    
    if 'anneal_beta' in config:
        run_name = run_name + f"_beta_{config['anneal_beta']}"

    # form run name from the config 
    run_name = run_name + f"_steps_{config['steps']}_temp{config['temperature']}_bs_{config['batch_size']}_num_particles_{config['num_particles']}"
    

    run_name = run_name + f"_{config['start_time']}"

    wandb.init(project=project, config=config, name=run_name, dir="~/scratch/discrete_fkc_wandb")
    return 

def decode(x, tokenizer):
    if tokenizer is not None:
        #txt_x = [
        #        "".join(seq.split(" "))
        #        for seq in tokenizer.batch_decode(
        #            x, skip_special_tokens=False
        #        )
        #    ]
        txt_x = tokenizer.batch_decode(x, skip_special_tokens=False)
    else:
        txt_x = None

    return txt_x

def get_logit_heatmap(logits):
    # clip logit values to prevent -inf values
    logits = torch.clamp(logits, min=-1000)

    logit_heatmap = []
    for i in range(logits.shape[0]):
        fig, ax = plt.subplots()
        ax.imshow(logits[i].detach().cpu().numpy(), aspect='auto', cmap='viridis')
        # set x and y labels
        ax.set_ylabel("Token")
        ax.set_xlabel("Vocab Index")

        logit_heatmap.append(wandb.Image(fig))
        plt.close(fig)

    return logit_heatmap

def plot_2d(x, title, step, log_prob_target = None):
    fig, ax = plt.subplots()
        
    # plot the target log prob
    if log_prob_target is not None:
        im = ax.imshow(log_prob_target.exp().detach().cpu().numpy())
        fig.colorbar(im, ax=ax)

        fig_target, ax_target = plt.subplots()
        im_target = ax_target.imshow(log_prob_target.exp().detach().cpu().numpy())
        fig_target.colorbar(im_target, ax=ax_target)
        wandb_tgt_im = wandb.Image(fig_target)
    else:
        wandb_tgt_im = None

    # Create scatter plots
    ax.scatter(x[:, 0].cpu(), x[:, 1].cpu())
            
    # Add titles and labels
    ax.set_title(f'{title} step {step}')
    ax.set_xlabel('Token 0')
    ax.set_ylabel('Token 1')
       
    wandb_im_x = wandb.Image(fig)

    return wandb_im_x, wandb_tgt_im

def wandb_get_additional_metrics(metric_dict):
    metrics = metric_dict.keys()

    metric_arrays = []

    for m in metrics:
        if m == "reward":
            # the dict should contain a list of reward values to log in this case 
            reward_metric = metric_dict[m]
            metric_arrays.append(reward_metric.cpu().numpy())

        
    return metric_arrays

def wandb_log_xt_smc(step, logits_prop, 
                     x_pre_unmask, x_pre_resample, x_r, x0, 
                     tokenizer, 
                     log_weights_r, ess_batch, 
                     additional_metrics = None, # dictionary of additional metrics to log
                     log_prob_target = None, mask_token = None, show_logits = None):
    batch_size, num_particles, length = x_r.shape  

    # flatten the particles for logging
    # but track which particle group each belongs to
    # each example in batch gets a number (0 to B-1) - all particles corresponding to that example get the same number
    particle_group = torch.arange(batch_size).unsqueeze(-1).expand(-1, num_particles)  # shape (batch_size, num_particles)
    particle_group = particle_group.reshape(batch_size * num_particles)
    particle_group = particle_group.cpu().numpy()

    x_pre_unmask = x_pre_unmask.view(batch_size * num_particles, length)
    x_pre_resample = x_pre_resample.view(batch_size * num_particles, length)
    x_r = x_r.view(batch_size * num_particles, length)
    #x0 = x0.view(batch_size * num_particles, length)

    log_weights_r = log_weights_r.view(batch_size * num_particles)

    ess_batch = torch.tensor(ess_batch).reshape(batch_size, -1).expand(-1, num_particles).reshape(batch_size * num_particles)  # shape (batch_size * num_particles)
    ess_batch = ess_batch.cpu().numpy()  # shape (batch_size, num_resampling_steps)

    txt_x_pre_um = decode(x_pre_unmask, tokenizer)
    txt_x_pre_res = decode(x_pre_resample, tokenizer)
    txt_x = decode(x_r, tokenizer)

    #print("x_r: ", x_r.shape)
    #print("x0: ", x0.shape)
    txt_x0 = decode(x0, tokenizer)
    

    x_pre_um_np = x_pre_unmask.cpu().numpy()
    x_pre_resample_np = x_pre_resample.cpu().numpy()
    x_r_np = x_r.cpu().numpy()
    x0_np = x0.cpu().numpy()


    if mask_token is not None:
        avg_num_masked = (x_pre_unmask == mask_token).sum().item() / x_pre_unmask.shape[0]
        
    else:
        avg_num_masked = None
    
    
    table_data = [[particle_group[i], 
                   np.array2string(x_pre_um_np[i]), 
                   np.array2string(x_pre_resample_np[i]), 
                   np.array2string(x_r_np[i]), 
                   np.array2string(x0_np[i]), 
                   log_weights_r[i].item(),
                   ess_batch[i].item(), 
                   txt_x_pre_um[i] if txt_x_pre_um is not None else None, 
                   txt_x_pre_res[i] if txt_x_pre_res is not None else None,
                   txt_x[i] if txt_x is not None else None,
                   txt_x0[i] if txt_x0 is not None else None] for i in range(x_pre_um_np.shape[0])]
    columns=['Particle Group', 'xt', 'xt pre resample', 'xt next (resampled)', 'x0_t', 
                                 'log weight', 'ESS', 'text xt', 'text x pre resample', 'text x next (resampled)', 'text x0']
                                 
    table = wandb.Table(columns=['Particle Group', 'xt', 'xt pre resample', 'xt next (resampled)', 'x0_t', 
                                 'log weight', 'ESS', 'text xt', 'text x pre resample', 'text x next (resampled)', 'text x0'], data=table_data)
    
    """
    table_data = [[particle_group[i], 
                   txt_x_pre_um[i] if txt_x_pre_um is not None else None, 
                   txt_x_pre_res[i] if txt_x_pre_res is not None else None,
                   txt_x[i] if txt_x is not None else None,
                   txt_x0[i] if txt_x0 is not None else None] for i in range(x_pre_um_np.shape[0])]
    table = wandb.Table(columns=['Particle Group', 'text xt', 'text x pre resample', 'text x next (resampled)', 'text x0'], data=table_data)
    """                             
    if show_logits:
        logit_heatmap = get_logit_heatmap(logits_prop)
        table.add_column("Logits", logit_heatmap)

    if additional_metrics:
        metric_arrays = wandb_get_additional_metrics(additional_metrics)
        for i, m in enumerate(additional_metrics.keys()):
            table.add_column(m, metric_arrays[i])

    log_info = {"step": step,
                "num masked": avg_num_masked,
                "avg_ess": np.mean(ess_batch),
                "samples": table}
    
    if x_pre_unmask.shape[-1] == 2:
        # plot scatter plot of x and x0 and log to wandb
        wandb_xt_pre_um_im, wandb_tgt_im = plot_2d(x_pre_unmask, 'xt', step, log_prob_target)
        wandb_x_pre_rs_im, _ = plot_2d(x_pre_resample, 'xt pre resample', step, log_prob_target)
        wandb_xr_im, _ = plot_2d(x_r, 'xt next (resampled)', step, log_prob_target)
        wandb_x0_im, _ = plot_2d(x0, 'x0', step, log_prob_target)

        log_info["Plot xt"] = wandb_xt_pre_um_im
        log_info["Plot xt pre resample"] = wandb_x_pre_rs_im
        log_info["Plot xt next (resampled)"] = wandb_xr_im
        log_info["Plot x0_t"] = wandb_x0_im
        
        if wandb_tgt_im is not None:
            log_info["Target Log Prob"] = wandb_tgt_im

    #wandb.log(log_info, step=step)
    return log_info 

# show_logits = True displays logits heatmap 
def wandb_log_xt(step, logits, x, x0, tokenizer, log_prob_target =None, mask_token = None, additional_metrics=False, show_logits = False):
    txt_x = decode(x, tokenizer)
    txt_x0 = decode(x0, tokenizer)

    x_np = x.cpu().numpy()
    x0_np = x0.cpu().numpy()


    if mask_token is not None:
        avg_num_masked = (x == mask_token).sum().item() / x.shape[0]
        
    else:
        avg_num_masked = None

    table_data = [[np.array2string(x_np[i]), np.array2string(x0_np[i]), txt_x[i] if txt_x is not None else None, txt_x0[i] if txt_x0 is not None else None] for i in range(x.shape[0])]
    table = wandb.Table(columns=['xt', 'x0_t', 'text x', 'text x0'], data=table_data)

    # log the tensor x and the decoded text to wandb
    
    if show_logits:
        logit_heatmap = get_logit_heatmap(logits)
        table.add_column("Logits", logit_heatmap)

    if additional_metrics:
        metric_arrays = wandb_get_additional_metrics(additional_metrics)
        for i, m in enumerate(additional_metrics.keys()):
            table.add_column(m, metric_arrays[i])

    log_info = {"step": step,
                "num masked": avg_num_masked,
                "samples": table}
    
    if x.shape[-1] == 2:
        # plot scatter plot of x and x0 and log to wandb
        wandb_xt_im, wandb_tgt_im = plot_2d(x, 'xt', step, log_prob_target)
        wandb_x0_im, _ = plot_2d(x0, 'x0', step, log_prob_target)
        
        log_info["Plot xt"] = wandb_xt_im
        log_info["Plot x0_t"] = wandb_x0_im
        
        if wandb_tgt_im is not None:
            log_info["Target Log Prob"] = wandb_tgt_im

    #wandb.log(log_info, step=step)
    return log_info 


def save_samples_to_file(x, tokenizer, filename):
    """
    Save the generated texts to a file.
    """

    if len(x.shape) == 2:   
        generated_texts = tokenizer.batch_decode(x, skip_special_tokens=True)

        particle_group = torch.arange(x.shape[0]) # all belong to separate groups
        particle_group = particle_group.cpu().numpy()
    # smc sample case, 
    elif len(x.shape) == 3:
        batch_size, num_particles, length = x.shape  
        x = x.view(batch_size * num_particles, length)
        generated_texts = tokenizer.batch_decode(x, skip_special_tokens=True)

        particle_group = torch.arange(batch_size).unsqueeze(-1).expand(-1, num_particles)  # shape (batch_size, num_particles)
        particle_group = particle_group.reshape(batch_size * num_particles)
        particle_group = particle_group.cpu().numpy()

    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    
    with open(filename, 'w') as f:
        print("number of samples: ", len(generated_texts))
        for j in range(len(generated_texts)):
            text = generated_texts[j]
            f.write("\n\nText index {} Sample Group {}:".format(j, particle_group[j]) + '\n')
            f.write("%s\n\n" % text)
    print("Generated texts saved to: ", filename)
    return

