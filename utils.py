# __author:Administrator
# 一些配置文件
# date: 2022/9/15

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
from torch.utils.data import Dataset


class myDataset(Dataset):
    def __init__(self, text_len, texts, labels, tokenizer):
        self.text_len = text_len
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        encoding = self.tokenizer.encode_plus(
                                            text,
                                            add_special_tokens=True,
                                            max_length=self.text_len,
                                            padding='max_length',
                                            truncation=True,
                                            return_token_type_ids=True,
                                            return_attention_mask=True,
                                            return_tensors='pt',
                                        )

        samples = {
            'text':text,
            'input_ids':encoding['input_ids'].flatten(),
            'attention_mask':encoding['attention_mask'].flatten(),
            'token_type_ids':encoding['token_type_ids'].flatten(),
            'label':torch.tensor(label, dtype=torch.long)
        }

        return samples


def compute_metrics(true_labels, pred_labels):
    assert len(pred_labels) == len(true_labels)
    results = {}
    acc = {'acc': accuracy_score(true_labels, pred_labels)}
    p = {'precision': precision_score(true_labels, pred_labels, average='macro')}
    r = {'recall': recall_score(true_labels, pred_labels, average='macro')}
    f1 = {'f1': f1_score(true_labels, pred_labels,average='macro')}
    rp = {'report': classification_report(true_labels, pred_labels, target_names=['负面', '中性', '正面'])}
    results.update(acc)
    results.update(p)
    results.update(r)
    results.update(f1)
    results.update(rp)

    return results

def evaluate(model, dataset, device, parms):
    '''验证'''
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=parms['batch_size'])

    eval_loss = 0.0
    eval_step = 0
    label_pred = None
    label_true = None

    model.eval()
    for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        label = batch['label'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask, token_type_ids, label)
            tmp_loss, label_logits = outputs
            eval_loss += tmp_loss.mean().item()

        eval_step += 1
        #
        if label_pred is None:
            label_pred = label_logits.detach().cpu().numpy()
            label_true = label.detach().cpu().numpy()
        else:
            label_pred = np.append(label_pred, label_logits.detach().cpu().numpy(), axis=0)
            label_true = np.append(label_true, label.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / eval_step
    results = {
        'val_loss': eval_loss
    }
    label_pred = np.argmax(label_pred, axis=1)
    total_result = compute_metrics(label_true, label_pred)  # 验证
    results.update(total_result)

    for key in sorted(results.keys()):
        print(key, str(results[key]))

    return results