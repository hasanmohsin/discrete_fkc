## Discrete Feynman-Kac Correctors

Code for Discrete Feynman-Kac Correctors.

<img width="2000" height="1771" alt="image" src="https://github.com/user-attachments/assets/98f99b67-d16d-48c8-8082-b182718f21c6" />


For Protein experiments with ESM2 reward, an example prompt is:

```
python unconditional_esm_reward.py --seed ${seed} --seq_length 10 --num_particles ${num_particles} --batch_num ${batch_size} --num_seqs 5 --beta 40.0 --save "./dplm_out/"
```

For running story generation experiments, an example prompt is:

```
python randomized_story_exp.py --seed ${seed} --num_cond 10 --num_particles 8 --remask_strat "random"
```
