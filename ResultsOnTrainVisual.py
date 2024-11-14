import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# 第一个文件为预测train.json的结果
with open('./baseline_lora_nohid_epoch100_true.json') as f:
    zeroshot = json.load(f)

with open('./train/train.json') as f:
    train = json.load(f)

# 初始化两个文件的img和chat计数器
zeroshot_img = defaultdict(int)
zeroshot_chat = defaultdict(int)
train_img = defaultdict(int)
train_chat = defaultdict(int)

# 遍历train.json文件
for item in train:
    if item['instruction'].startswith('Picture'):
        train_img[item['output']] += 1
    else:
        train_chat[item['output']] += 1

# 遍历zeroshot.json文件
for item in zeroshot:
    if item['instruction'].startswith('Picture'):
        zeroshot_img[item['output']] += 1
    else:
        zeroshot_chat[item['output']] += 1

# 提取x轴和y轴的数据
x_train_img = list(train_img.keys())
y_train_img = list(train_img.values())
x_zeroshot_img = list(zeroshot_img.keys())
y_zeroshot_img = list(zeroshot_img.values())

x_train_chat = list(train_chat.keys())
y_train_chat = list(train_chat.values())
x_zeroshot_chat = list(zeroshot_chat.keys())
y_zeroshot_chat = list(zeroshot_chat.values())
plt.rcParams['font.family'] = 'Noto Sans CJK JP'
# 创建图表
plt.figure(figsize=(12, 8))

# 绘制img的条形图
plt.subplot(2, 1, 1)  # 2行1列的第一个
plt.bar(x_train_img, y_train_img, color='skyblue', label='Train Img')
plt.bar(x_zeroshot_img, y_zeroshot_img, color='lightgreen', label='Zeroshot Img')

# 设置标题和坐标轴标签
plt.title('Img 数据分布')
plt.xlabel('意图')
plt.ylabel('数量')
plt.xticks(rotation=45, fontsize=8)
plt.legend()

# 绘制chat的条形图
plt.subplot(2, 1, 2)  # 2行1列的第二个
plt.bar(x_train_chat, y_train_chat, color='skyblue', label='Train Chat')
plt.bar(x_zeroshot_chat, y_zeroshot_chat, color='lightgreen', label='Zeroshot Chat')

# 设置标题和坐标轴标签
plt.title('Chat 数据分布')
plt.xlabel('意图')
plt.ylabel('数量')
plt.xticks(rotation=45, fontsize=8)
plt.legend()

# 显示图表
plt.tight_layout()
plt.show()