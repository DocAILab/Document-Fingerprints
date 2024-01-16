#!/bin/bash

# 设置变量，用于存储命令行参数
dataset_name="relish"
dataset_dir="./datasets/RELISH"
results_dir="./results"


# 调用 Python 脚本并传递参数
python main.py --dataset_name "$dataset_name" --dataset_dir "$dataset_dir" --results_dir "$results_dir"