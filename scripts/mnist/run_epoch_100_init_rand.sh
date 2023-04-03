#!/bin/bash

# Scipt to run output mixture, stochastic, top-1, top-2 and single model

sbatch scripts/mnist/schedule_mnist_original_model.sh -g gate_layers_top_k -k 1 -m mnist_rand_init_top_1 -mt moe_top_k_model -r 10 -M 10 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_original_model.sh -g gate_layers_top_k -k 2 -m mnist_rand_init_top_2 -mt moe_top_k_model -r 10 -M 10 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_original_model.sh -m mnist_rand_init  -mt moe_expectation_model -r 10 -M 10 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_original_model.sh -m mnist_stochastic_rand_init  -mt moe_stochastic_model -r 10 -M 10 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_with_attention.sh  -k 1 -m mnist_with_attn_rand_init_top_1 -mt moe_top_k_model -r 10 -M 10 -E 100  

sbatch aaai_2022/scripts/mnist/schedule_mnist_with_attention.sh  -k 2 -m mnist_with_attn_rand_init_top_2 -mt moe_top_k_model -r 10 -M 10 -E 100

sbatch aaai_2022/scripts/mnist/schedule_mnist_with_attention.sh -m mnist_with_attn_rand_init -mt moe_expectation_model -r 10 -M 10 -E 100 

sbatch aaai_2022/scripts/mnist/schedule_mnist_with_attention.sh -m mnist_with_attn_stochastic_rand_init -mt moe_stochastic_model -r 10 -M 10 -E 100 

sbatch aaai_2022/scripts/mnist/schedule_mnist_single_model.sh -r 10 -E 100