# __author:Administrator
# 抖音文本情感分类, 使用Bert+softmax
# date: 2022/9/15

import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split
import torch
from utils import evaluate, myDataset
from model import EmotionClassifier
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup

import warnings
warnings.filterwarnings('ignore')

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# 参数设置
parms = {
    'learn_rate': 5e-5, # 学习率
    'batch_size': 32,  # 批处理大小·
    'max_length': 32, # 文本长度
    'epochs': 10, # 训练轮次
    'n_classes': 3, # 类别数量
    'logger_steps': 50, # 验证步数
    'drop_rate': 0.3,  # 丢失率
    'pretrain_file': "E:/Code/Pretrain/zh/bert-base-chinese",  # 预训练模型,自己提前下载到本地
    'data_file': './data', # 数据读取和保存文件
    'raw_data': '/DY.csv', # 原始数据文件
    'save_file': '/save' # 保存文件
}

if not os.path.exists(parms['data_file'] + parms['save_file']):
    os.makedirs(parms['data_file'] + parms['save_file'])

start_time = time.time()
# 数据loading                         train: val: test: = 8: 1: 1
texts = pd.read_csv(parms["data_file"] + parms['raw_data'])
texts, labels = texts['评论内容'].tolist(), (texts['label'] + 1).tolist()
x_train, x_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=1234, stratify=labels)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=1234, stratify=y_test)

# tokenizer
tokenizer = BertTokenizer.from_pretrained(parms['pretrain_file'])
train_dataset = myDataset(parms['max_length'], x_train, y_train, tokenizer)
val_dataset = myDataset(parms['max_length'], x_val, y_val, tokenizer)
test_dataset = myDataset(parms['max_length'], x_test, y_test, tokenizer)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionClassifier(parms)
model.to(device)
# 差分学习率
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
bert_param_optimizer = list(model.bert.named_parameters())

optimizer_grouped_parameters = [
    {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay': 0.1},
    {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
]

optimizer = AdamW(optimizer_grouped_parameters, lr=parms['learn_rate'], eps=1e-8)

# 训练
train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=parms['batch_size'])
t_total = len(train_dataloader) * parms['epochs']
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1*t_total, num_training_steps=t_total)
global_step = 0
tr_loss = 0.0
best_acc = 0.0
epoch = 0
while epoch < int(parms['epochs']):
    losses = []
    epoch_iterator = tqdm(train_dataloader, desc='Iteration')
    for step, batch in enumerate(epoch_iterator):
        model.train()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        label = batch['label'].to(device)
        outputs = model(input_ids, attention_mask, token_type_ids, label)
        loss = outputs[0]
        losses.append(loss.item())
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        if global_step % parms['logger_steps'] == 0:

            print("global step %d, epoch: %d,  loss: %.5f, lr: %.10f"
                  % (global_step, epoch, np.mean(losses),  float(scheduler.get_last_lr()[0])))

            result = evaluate(model, val_dataset, device, parms)
            # saving best model
            if result['acc'] > best_acc:
                best_acc = result['acc']
                torch.save(model.state_dict(), parms['data_file'] + parms['save_file'] + '/model.pt')

    epoch += 1

print("************train_best_acc: %.5f**********" % best_acc)

# 测试
print("************测试结果**********")
model.load_state_dict(torch.load(parms['data_file'] + parms['save_file']+ '/model.pt'))
evaluate(model, test_dataset, device, parms)
print("***********消耗时间：%.5f***********" % (time.time() -start_time ))
print("***********结束***********")