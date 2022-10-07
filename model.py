# __author:Administrator
# date: 2022/9/15

import torch
import torch.nn as nn
from transformers import BertModel


class EmotionClassifier(nn.Module):
    def __init__(self, parms):
        super(EmotionClassifier, self).__init__()
        self.parms = parms
        self.drop = nn.Dropout(self.parms['drop_rate'])
        self.bert = BertModel.from_pretrained(self.parms['pretrain_file']) # 加载预训练模型
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.parms['n_classes'])

    def forward(self, input_ids, attention_mask, token_type_ids, label_ids=None):

        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict = False
        )

        logits = self.classifier(self.drop(pooled_output))
        total_loss = 0
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(logits.view(-1, self.parms['n_classes']), label_ids.view(-1))
        total_loss += loss

        outputs = (logits,)
        outputs = (total_loss,) + outputs

        return outputs  # loss, logits