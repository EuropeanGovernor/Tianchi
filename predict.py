import subprocess

subprocess.run(['llamafactory-cli train ../LLaMA-Factory/examples/predict.yaml'])
subprocess.run(['cp -r ../LLaMA-Factor/saves/qwen2_vl-7b/lora/sft-infer ../MIRE/data/demo_pred.jsonl'])
subprocess.run(['python ../MIRE/mire_baseline/convert2submit.py'])
subprocess.run(['python ../MIRE/mire_baseline/cal_acc.py'])