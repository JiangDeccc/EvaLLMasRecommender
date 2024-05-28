#!/bin/sh -x

# user profile exp.
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history10_candidate20_shuffle_profile_gpt3.5_history_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history10_candidate20_shuffle_profile_gpt3.5_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history10_candidate20_shuffle_profile_gpt4_history_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history10_candidate20_shuffle_profile_gpt4_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history10_candidate20_shuffle_profile_glm4_history_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history10_candidate20_shuffle_profile_glm4_sample
python llm_metrics.py --results_path ../../llm_results/sports/results_gpt3.5_seed0_test_history10_candidate20_shuffle_profile_gpt3.5_history_sample --dataset sports
python llm_metrics.py --results_path ../../llm_results/sports/results_gpt3.5_seed0_test_history10_candidate20_shuffle_profile_gpt3.5_sample --dataset sports
python llm_metrics.py --results_path ../../llm_results/sports/results_gpt3.5_seed0_test_history10_candidate20_shuffle_profile_gpt4_history_sample --dataset sports
python llm_metrics.py --results_path ../../llm_results/sports/results_gpt3.5_seed0_test_history10_candidate20_shuffle_profile_gpt4_sample --dataset sports
python llm_metrics.py --results_path ../../llm_results/sports/results_gpt3.5_seed0_test_history10_candidate20_shuffle_profile_glm4_history_sample --dataset sports
python llm_metrics.py --results_path ../../llm_results/sports/results_gpt3.5_seed0_test_history10_candidate20_shuffle_profile_glm4_sample --dataset sports

python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history1_candidate20_shuffle_profile_gpt3.5_history_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history1_candidate20_shuffle_profile_gpt3.5_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history2_candidate20_shuffle_profile_gpt3.5_history_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history2_candidate20_shuffle_profile_gpt3.5_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history3_candidate20_shuffle_profile_gpt3.5_history_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history3_candidate20_shuffle_profile_gpt3.5_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history4_candidate20_shuffle_profile_gpt3.5_history_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history4_candidate20_shuffle_profile_gpt3.5_sample

# rerank exp.
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history10_multi_candidate20_shuffle_sample_1k --sample_num 1000
python llm_metrics.py --results_path ../../llm_results/sports/results_gpt3.5_seed0_test_history10_multi_candidate20_shuffle_sample_1k --sample_num 1000 --dataset sports

python llm_metrics.py --sota True --model_name BPRMF --history_max 10 --sample_num 1000
python llm_metrics.py --sota True --model_name LightGCN --history_max 10 --sample_num 1000
python llm_metrics.py --sota True --model_name SASRec --history_max 10 --sample_num 1000
python llm_metrics.py --sota True --model_name GRU4Rec --history_max 10 --sample_num 1000

# llmrank
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history10_candidate20_shuffle_llmrank_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history10_candidate20_shuffle_llmrank_rf_sample

# history sensibility
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history1_candidate20_shuffle_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history2_candidate20_shuffle_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history3_candidate20_shuffle_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history4_candidate20_shuffle_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history5_candidate20_shuffle_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history15_candidate20_shuffle_sample
python llm_metrics.py --results_path ../../llm_results/beauty/results_gpt3.5_seed0_test_history20_candidate20_shuffle_sample