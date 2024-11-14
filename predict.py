import json
import pandas as pd
import subprocess

exp='baseline_lora_nohid_epoch100'
ckpt=2800

for num in [10,5,1]:
    subprocess.run(['bash','../LLaMA-Factory/predict.sh',\
                    str(num), \
                    f'../LLaMA-Factory/saves/pred/{exp}_{num}pred',\
                    f'../LLaMA-Factory/saves/{exp}/checkpoint-{ckpt}/'],check=True)

    subprocess.run(['python','../MIRE/mire_baseline/convert2submit.py'\
                    '--test_file', '../LLaMA-Factory/data/mire_test/test_nohid_10pred.json'\
                    '--prediction_file',f'../LLaMA-Factory/saves/pred/{exp}_{num}pred/generated_predictions.jsonl',\
                    '--save_path',f'../LLaMA-Factory/saves/pred/{exp}_{num}pred/submit.csv'],check=True)


    with open(f'../LLaMA-Factory/data/mire_test/test_nohid_10pred.json','r') as f:
        test_data = json.load(f)

    submit_data = pd.read_csv(f'../LLaMA-Factory/saves/pred/{exp}_{num}pred/submit.csv')
    
    if num==1:
        with open("../WWW2025/img_cls.json") as img:
            img_cls=json.load(img)
        with open("../WWW2025/chat_cls.json") as chat:
            chat_cls=json.load(chat)
    
    for i in test_data:
        temp = i['instruction'].split("\n")
        if i['instruction'].startswith('Picture'):
            if num!=1 : temp[-1] = f'\n你是一个电商领域识图专家,可以理解消费者上传的软件截图或实物拍摄图。现在,请你对消费者上传的图片进行分类。你只需要回答图片可能的{num}个分类结果,不需要其他多余的话。以下是可以参考的分类标签,分类标签:'+submit_data[['id']==i['id']]['predict']
            else: 
                for label in submit_data[['id']==i['id']]['predict']: label=label+":"+img_cls[label]
                temp[-1] = f'\n你是一个电商领域识图专家,可以理解消费者上传的软件截图或实物拍摄图。现在,请你对消费者上传的图片进行分类。你只需要回答图片最有可能的{num}个分类结果,不需要其他多余的话。以下是可以参考的分类标签以及对应的解释:'+submit_data[['id']==i['id']]['predict']
        else:
            if num!=1 : temp[-1] = f'\n请直接只输出有可能的{num}个分类标签结果，不需要其他多余的话。以下是可以参考的分类标签为：'+submit_data[['id']==i['id']]['predict']
            else : 
                for label in submit_data[['id']==i['id']]['predict']: label=label+":"+chat_cls[label]
                temp[-1] = f'\n请直接只输出最有可能的{num}个分类标签结果，不需要其他多余的话。以下是可以参考的分类标签和对应的解释：'+submit_data[['id']==i['id']]['predict']
        temp='\n'.join(temp[:-1])+temp[-1]

    with open('../LLaMA-Factory/data/mire_test/test_nohid_10pred.json','w') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)
    
    subprocess.run(['cp','-r',"../LLaMA-Factory/data/mire_test/test_nohid_10pred.json",f"../LLaMA-Factory/saves/pred/{exp}_{num}pred/test_nohid_{num}pred.json"])