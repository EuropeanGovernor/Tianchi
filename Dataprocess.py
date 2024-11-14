import re
import json
class DataPreprocess():
    """
    PATH1:原始train.json文件路径
    PATH2:清洗后导出train.json文件路径
    args:如果是对训练集预处理,arg为'train';如果是对测试集预处理,arg为'test'
    """
    def __init__(self,PATH1,PATH2,arg,):
        with open(PATH1) as f:
            self.data=json.load(f)

        self.PATH2=PATH2
        self.arg=arg

    def remove_link_blank(self,data):
        res=[]
        for i in range(len(data)):
            counter=0
            data[i]['image']=[f'data/mire_{self.arg}/images/'+_ for _ in data[i]['image']]
            temp=data[i]['instruction'].split('\n')

            if temp[0].startswith("Picture"): pass
            else:
                dialog=temp[2:-3]
                for j in range(len(dialog)):
                    dialog[j-counter]=re.sub(r'<dxm:highlight>|</dxm:highlight>','',dialog[j-counter])
                    dialog[j-counter]=re.sub(r'【|】|-|—|~|：|<http>|\||"','',dialog[j-counter])
                    dialog[j-counter]=re.sub(r'https?://\S+','',dialog[j-counter])
                    dialog[j-counter]=re.sub(u'[\U00010000-\U0010ffff]|[\u2100-\u21FF]|[\u2600-\u2B55]|[\uD800-\uDFFF]','',dialog[j-counter])

                    if dialog[j-counter]=='客服: ' or dialog[j-counter]=='用户: ':
                        dialog.pop(j-counter)
                        counter+=1

            res.append(data[i])
        print(f'训练集大小为：{len(res)}')
        return res
    def get_dialog(self):
        dialog=self.remove_link_blank(self.data)
        with open(self.PATH2,'w') as f:
            json.dump(dialog,f,ensure_ascii=False,indent=4)

DataPreprocess('./train/train.json','../LLaMA-Factory/data/mire_train/train.json','train').get_dialog()