#!/bin/sh -x

python llm_metrics.py --sota True --model_name BM25 --history_max 10 --sample_num 1000 --dataset sports
python llm_metrics.py --sota True --model_name BPRMF --history_max 10 --sample_num 1000 --dataset sports
python llm_metrics.py --sota True --model_name SASRec --history_max 10 --sample_num 1000 --dataset sports
python llm_metrics.py --sota True --model_name LightGCN --history_max 10 --sample_num 1000 --dataset sports
python llm_metrics.py --sota True --model_name GRU4Rec --history_max 10 --sample_num 1000 --dataset sports