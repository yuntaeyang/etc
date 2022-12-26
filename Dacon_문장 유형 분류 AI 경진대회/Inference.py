import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import f1_score

import transformers
from transformers import AutoTokenizer, AdamW, AutoModel
from transformers import get_linear_schedule_with_warmup

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import random
import torch.backends.cudnn as cudnn

from sklearn.model_selection import StratifiedKFold

from adamp import AdamP
from Dataset import *
from FocalLoss import *
from Model import *

def parse_args():
    parser = argparse.ArgumentParser(description='Process some arguments')
    parser.add_argument('--batch_size', default=16, type=int, help='batch for training.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    args.device = torch.device('cuda')
    return args

path = "/content/drive/MyDrive/문장 유형 분류 AI 경진대회/open/"
test=pd.read_csv(f"{path}test.csv")
submission = pd.read_csv(f"{path}sample_submission.csv")

model_nm = 'klue/roberta-large'
tokenizer = AutoTokenizer.from_pretrained(model_nm)

# 예측 
def inference(model, dataset_test, tokenizer, batch_size, device):
    test_dataset = SentenceTypeDataset(dataset_test, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    type_probs, polarity_probs, tense_probs, clarity_probs = [], [], [], []
    with torch.no_grad():
        model.eval()
        for data_input, _, _, _, _ in tqdm(test_loader):
            attention_mask = data_input['attention_mask'].to(device)
            input_ids = data_input['input_ids'].squeeze(1).to(device)


            type_output, polarity_output, tense_output, clarity_output = model(input_ids, attention_mask)
            type_probs.append(type_output)
            polarity_probs.append(polarity_output)
            tense_probs.append(tense_output)
            clarity_probs.append(clarity_output)
    
    return torch.cat(type_probs).cpu().detach().numpy(), \
            torch.cat(polarity_probs).cpu().detach().numpy(), \
            torch.cat(tense_probs).cpu().detach().numpy(), \
            torch.cat(clarity_probs).cpu().detach().numpy()

def inference_main(args):
  res_1 = np.zeros((len(test),4))
  res_2 = np.zeros((len(test),3))
  res_3 = np.zeros((len(test),3))
  res_4 = np.zeros((len(test),2)) 
  for i in range(5): 
    print(f'fold{i} 모델 추론중...')
    # load my model
    model = torch.load(str(path)+'model'+str(i)+'.pt')

    test_pred_type, test_pred_polarity, test_pred_tense, test_pred_certainty = inference(model, test, tokenizer, args.batch_size, args.device)

    res_1 += test_pred_type / 5
    res_2 += test_pred_polarity / 5 
    res_3 += test_pred_tense / 5 
    res_4 += test_pred_certainty / 5  

  
  test_type = ['대화형' if i==0 else '사실형' if i==1 else '예측형' if i==2 else '추론형' for i in [np.argmax(p) for p in res_1]]
  test_polarity = ['긍정' if i==0 else '미정' if i==1 else '부정' for i in [np.argmax(p) for p in res_2]]
  test_tense = ['과거' if i==0 else '미래' if i==1 else '현재' for i in [np.argmax(p) for p in res_3]]
  test_certainty = ['불확실' if i==0 else '확실' for i in [np.argmax(p) for p in res_4]]

  label_sum = []
  for i in range(len(test_type)):
    label_sum.append(f'{test_type[i]}-{test_polarity[i]}-{test_tense[i]}-{test_certainty[i]}')

  submission['label'] = label_sum
  submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    args = parse_args()
    inference_main(args)