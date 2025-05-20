# -*- coding: utf-8 -*-
# @Time:2024/6/13 15:45
# @File:test.py
# @software:PyCharm
import torch
import json
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

data1 = torch.load('frozen_tag_embedding/ram_plus_tag_embedding_class_4585_des_51.pth',map_location='cpu')
print(data1.shape)
data2 = torch.load('frozen_tag_embedding/embedding.pth',map_location='cpu')
print(data2.shape)
if torch.equal(data1, data2):
    print('相同')
else:
    print('不同')
    print(data1[0,:])
    print(data2[0,:])