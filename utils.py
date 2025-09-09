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

    # form run name from the config 
    run_name = f"sampler_{config['sampler']}_steps_{config['steps']}_temp{config['temperature']}_bs_{config['batch_size']}_num_particles_{config['num_particles']}_{config['start_time']}"
    wandb.init(project=project, config=config, name=run_name, dir="~/scratch/discrete_fkc_wandb")
    return 

# show_logits = True displays logits heatmap 
def wandb_log_xt(step, logits, x, x0, tokenizer, log_prob_target =None, mask_token = None, show_logits = False):
    if tokenizer is not None:
        txt_x = [
            "".join(seq.split(" "))
            for seq in tokenizer.batch_decode(
                x, skip_special_tokens=False
            )
        ]

        txt_x0 = [
            "".join(seq.split(" "))
            for seq in tokenizer.batch_decode(
                x0, skip_special_tokens=False
            )
        ]
    else:
        txt_x = None
        txt_x0 = None
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
        # clip logit values to prevent -inf values
        logits = torch.clamp(logits, min=-1000)

        logit_heatmap  =[]
        for i in range(logits.shape[0]):
            fig, ax = plt.subplots()
            ax.imshow(logits[i].detach().cpu().numpy(), aspect='auto', cmap='viridis')
            # set x and y labels
            ax.set_ylabel("Token")
            ax.set_xlabel("Vocab Index")

            logit_heatmap.append(wandb.Image(fig))
            plt.close(fig)
        table.add_column("Logits", logit_heatmap)


    log_info = {"step": step,
                "num masked": avg_num_masked,
                "samples": table}
    
    if x.shape[-1] == 2:
        # plot scatter plot of x and x0 and log to wandb
        fig_xt, ax_xt = plt.subplots()
        fig_x0, ax_x0 = plt.subplots()
        
        # plot the target log prob
        if log_prob_target is not None:
            im_xt = ax_xt.imshow(log_prob_target.exp().detach().cpu().numpy())
            fig_xt.colorbar(im_xt, ax=ax_xt)

            im_x0 = ax_x0.imshow(log_prob_target.exp().detach().cpu().numpy())
            fig_x0.colorbar(im_x0, ax=ax_x0)

            fig_target, ax_target = plt.subplots()
            im_target = ax_target.imshow(log_prob_target.exp().detach().cpu().numpy())
            fig_target.colorbar(im_target, ax=ax_target)
            log_info["Target log prob"] = wandb.Image(fig_target)


        # Create scatter plots
        ax_xt.scatter(x[:, 0].cpu(), x[:, 1].cpu())
        ax_x0.scatter(x0[:, 0].cpu(), x0[:, 1].cpu())
        
        # Add titles and labels
        ax_xt.set_title(f'xt step {step}')
        ax_x0.set_title(f'x0 at step {step}')
        ax_xt.set_xlabel('Token 0')
        ax_xt.set_ylabel('Token 1')
        ax_x0.set_xlabel('Token 0')
        ax_x0.set_ylabel('Token 1')

        log_info["Plot xt"] = wandb.Image(fig_xt)
        log_info["Plot x0_t"] = wandb.Image(fig_x0)

    wandb.log(log_info, step=step)