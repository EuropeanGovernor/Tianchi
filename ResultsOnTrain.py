# 这个脚本用来测试模型在训练集上的效果

import json
import pandas as pd

true,false=[],[]
# 这个路径填在训练集上推理得到的sumbit.csv的路径
zeroshot=pd.read_csv('../LLaMA-Factory/saves/pred/baseline_lora_nohid_augment_epoch10_train/submit.csv')

# 这个填训练集路径
with open('./train/train.json','r') as f:
    train=json.load(f)

img,text=0,0
for predict_value, train_item in zip(zeroshot['predict'], train):
    output_value = train_item['output']  

    if predict_value == output_value:
        if train_item['instruction'].startswith('Picture'):img+=1
        else:text+=1
        true.append(train_item)
    else:
        train_item['predict'] = predict_value
        false.append(train_item)

print(f'训练集预测正确的数量：{len(true)}')
print(f"预测对的图片数量：{img},预测对的文本数量：{text}")