@echo off
cd /d C:\Users\Administrator\Desktop\??\DVSR
F:\Anaconda\envs\dvsr_gpu\python.exe train_VGSR_CUB.py --config experiments/04_hyperparameter_tuning/TUNE-021_topo01_cond_text001/config.yaml > experiments/04_hyperparameter_tuning/TUNE-021_topo01_cond_text001/logs/run_bg.txt 2>&1
