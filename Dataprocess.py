import re
import json
import torch
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class DataPreprocess():
    """
    train_PATH:原始train.json文件路径
    test_PATH:原始test.json文件路径
    output_train_PATH:清洗后导出train.json文件路径
    output_test_PATH:清洗后导出test.json文件路径
    """
    def __init__(self,train_PATH,test_PATH,output_train_PATH,output_test_PATH):
        with open(train_PATH) as f:
            self.train_data=json.load(f)
        
        with open(test_PATH) as f:
            self.test_data=json.load(f)

        self.output_train_PATH=output_train_PATH
        self.output_test_PATH=output_test_PATH
    def remove_link_blank(self,data,tag):
        for i in tqdm(range(len(data))):
            counter=0
            temp=data[i]['instruction'].split('\n')

            if temp[0].startswith("Picture"):
                pass
            else:
                dialog=temp[2:-3]
                for j in range(len(dialog)):
                
                    dialog[j-counter]=re.sub(r'<dxm:highlight>|</dxm:highlight>','',dialog[j-counter])
                    dialog[j-counter]=re.sub(r'【|】|-|—|~|：|\|"','',dialog[j-counter])
                    dialog[j-counter]=re.sub(r'https?://\S+|<http>','',dialog[j-counter])
                    dialog[j-counter]=re.sub(u'[\U00010000-\U0010ffff]|[\u2100-\u21FF]|[\u2600-\u2B55]|[\uD800-\uDFFF]','',dialog[j-counter])

                    
                    if dialog[j-counter]=='客服: ' or dialog[j-counter]=='用户: ':
                        dialog.pop(j-counter)
                        counter+=1
            data[i]['image']=[f'data/mire_{tag}/images/'+_ for _ in data[i]['image']]
            data[i]['instruction']='\n'.join(temp[:2]+dialog+temp[-3:])
        return data
                
                
    def get_dialog(self):
        train_dialog=self.remove_link_blank(self.train_data,'train')
        test_dialog=self.remove_link_blank(self.test_data,'test')
        with open(self.output_train_PATH,'w') as f:
            json.dump(train_dialog,f,ensure_ascii=False,indent=4)
        with open(self.output_test_PATH,'w') as f:
            json.dump(test_dialog,f,ensure_ascii=False,indent=4)

DataPreprocess(train_PATH=None,test_PATH=None,output_train_PATH=None,output_test_PATH=None)