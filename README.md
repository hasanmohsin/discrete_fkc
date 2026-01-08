# Discrete Feynman-Kac Correctors

Implementation for Discrete Feynman-Kac Correctors.

<img width="2000" height="1771" alt="image" src="https://github.com/user-attachments/assets/98f99b67-d16d-48c8-8082-b182718f21c6" />

Example commands for running various experiments are given below:

### For Language experiments:

- coding problems with annealing:
  - Human eval dataset
  ```
  python code_anneal_eval_exp.py --dataset "human_eval" --num_points -1 --seed 0 --num_particles 4 --savedir "./results/human_eval/" --beta 10.0 
  ```
  - MBPP dataset:
  ```
  python code_anneal_eval_exp.py --dataset "mbpp_san" --num_points -1 --seed 0 --num_particles 4 --savedir "./results/mbpp_san/" --beta 20.0 
  ``` 

- amortized linear regression with products:

```
python lin_reg_sampler_exp.py --seed ${seed} --num_particles 5 --num_splits_prod 5 --n_samples 20 --savedir "./results/lin_reg_prod"
```

- multiconstraint story generation with products:

```
python randomized_story_exp.py --seed 0 --num_cond 10 --num_particles 8 --remask_strat "random" --savedir "./results/story_gen_prod"
```


### For Protein experiments:

Running these experiments will require setting up the environment in [the DPLM repo](https://github.com/bytedance/dplm).

- ESM2 reward:
```
python unconditional_esm_reward_exp.py --seed 0 --seq_length 50 --num_particles 5 --batch_num 1 --num_seqs 5 --beta 10.0 --save "./results/dplm_out_esm2_reward"
```

- Thermostability reward:

```
python thermostability_reward_exp.py --seed 0 --seq_length 50 --num_particles 5 --batch_num 1 --num_seqs 5 --beta 10.0 --save "./results/dplm_out_thermo_reward"
``` 


