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
    parser.add_argument('--seed', default=2021, type=int, help='seed for initializing training.')
    parser.add_argument('--epochs', default=10, type=int, help='epoch for training.')
    parser.add_argument('--batch_size', default=16, type=int, help='batch for training.')
    parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
    args = parser.parse_args()
    args.device = torch.device('cuda')
    return args

path = "/content/drive/MyDrive/문장 유형 분류 AI 경진대회/open/"
train=pd.read_csv(f"{path}train.csv")
train_1 = train[(train['label'] != '추론형-미정-현재-불확실') & (train['label'] != '예측형-부정-미래-불확실') & (train['label'] != '추론형-미정-현재-확실') & (train['label'] != '추론형-미정-미래-확실') & (train['label'] != '사실형-미정-과거-확실') & (train['label'] != '예측형-미정-현재-불확실') & (train['label'] != '대화형-미정-미래-확실') & (train['label'] != '예측형-미정-미래-확실') & (train['label'] != '추론형-미정-과거-불확실') & (train['label'] != '대화형-부정-과거-불확실') & (train['label'] != '대화형-부정-미래-확실') & (train['label'] != '대화형-미정-과거-확실') & (train['label'] != '예측형-미정-현재-확실') & (train['label'] != '예측형-부정-현재-불확실') & (train['label'] != '예측형-부정-과거-확실') & (train['label'] != '예측형-미정-과거-확실')]
train_2 = train[(train['label'] == '추론형-미정-현재-불확실') | (train['label'] == '예측형-부정-미래-불확실') | (train['label'] == '추론형-미정-현재-확실') | (train['label'] == '추론형-미정-미래-확실') | (train['label'] == '사실형-미정-과거-확실') | (train['label'] == '예측형-미정-현재-불확실') | (train['label'] == '대화형-미정-미래-확실') | (train['label'] == '예측형-미정-미래-확실') | (train['label'] == '추론형-미정-과거-불확실') | (train['label'] == '대화형-부정-과거-불확실') | (train['label'] == '대화형-부정-미래-확실') | (train['label'] == '대화형-미정-과거-확실') | (train['label'] == '예측형-미정-현재-확실') | (train['label'] == '예측형-부정-현재-불확실') | (train['label'] == '예측형-부정-과거-확실') | (train['label'] == '예측형-미정-과거-확실')]

model_nm = 'klue/roberta-large'
base_model = AutoModel.from_pretrained(model_nm)
tokenizer = AutoTokenizer.from_pretrained(model_nm)

def training(train_dataset,val_dataset, base_model, tokenizer, fold, epochs, batch, device):
  train_tmp = train_dataset[['문장', '유형', '극성', '시제', '확실성']]
  train_tmp = pd.get_dummies(train_tmp, columns=['유형', '극성', '시제', '확실성'])
  val_tmp = val_dataset[['문장', '유형', '극성', '시제', '확실성']]
  val_tmp = pd.get_dummies(val_tmp, columns=['유형', '극성', '시제', '확실성'])

  train_type = train_tmp.iloc[:,1:5].values.tolist()
  train_polarity = train_tmp.iloc[:,5:8].values.tolist()
  train_tense = train_tmp.iloc[:,8:11].values.tolist()
  train_certainty = train_tmp.iloc[:,11:13].values.tolist()
  train_labels = {
      'type': train_type,
      'polarity': train_polarity,
      'tense': train_tense,
      'certainty': train_certainty
      }

  val_type = val_tmp.iloc[:,1:5].values.tolist()
  val_polarity = val_tmp.iloc[:,5:8].values.tolist()
  val_tense = val_tmp.iloc[:,8:11].values.tolist()
  val_certainty = val_tmp.iloc[:,11:13].values.tolist()
  val_labels = {
      'type': val_type,
      'polarity': val_polarity,
      'tense': val_tense,
      'certainty': val_certainty
      }

  model = SentenceClassifier(base_model).to(device)
  
  dataset_train = SentenceTypeDataset(train_tmp, tokenizer, train_labels)
  dataset_val = SentenceTypeDataset(val_tmp, tokenizer, val_labels)

  train_loader = DataLoader(dataset_train, batch_size=batch, shuffle=True)
  valid_loader = DataLoader(dataset_val, batch_size=batch, shuffle=False)

  optimizer = AdamP(model.parameters(), lr=1e-5, betas=(0.9, 0.999), weight_decay=1e-2)

  total_steps = len(train_loader) * epochs

  criterion = {
        'type' : FocalLoss().to(device),
        'polarity' : FocalLoss().to(device),
        'tense' : FocalLoss().to(device),
        'certainty' : FocalLoss().to(device)
    }

  scheduler = get_linear_schedule_with_warmup(optimizer, 
                                              num_warmup_steps = 0,
                                              num_training_steps = total_steps)

  for epoch in range(epochs):
    total_loss_train = 0
        
    model.train() 
        
    for train_input, type_label, polarity_label, tense_label, certainty_label in tqdm(train_loader):
      attention_mask = train_input['attention_mask'].to(device)
      input_ids = train_input['input_ids'].squeeze(1).to(device)
      type_label = type_label.to(device)
      polarity_label = polarity_label.to(device)
      tense_label = tense_label.to(device)
      certainty_label = certainty_label.to(device)

      optimizer.zero_grad()
            
      type_output, polarity_output, tense_output, certainty_output = model(input_ids, attention_mask) # from the forward function
            
      loss = 0.25*criterion['type'](type_output, type_label.float()) + \
             0.25*criterion['polarity'](polarity_output, polarity_label.float()) + \
             0.25*criterion['tense'](tense_output, tense_label.float()) + \
             0.25*criterion['certainty'](certainty_output, certainty_label.float())
      total_loss_train += loss.item()

      loss.backward()
      optimizer.step()
      scheduler.step()

    with torch.no_grad(): # since we should not change gradient for validation 
      total_loss_val = 0
            
      model.eval() # deactivate training
            
    
      for val_input, vtype_label, vpolarity_label, vtense_label, vcertainty_label in tqdm(valid_loader):
        attention_mask = val_input['attention_mask'].to(device)
        input_ids = val_input['input_ids'].squeeze(1).to(device)

        vtype_label = vtype_label.to(device)
        vpolarity_label = vpolarity_label.to(device)
        vtense_label = vtense_label.to(device)
        vcertainty_label = vcertainty_label.to(device)
                
        vtype_output, vpolarity_output, vtense_output, vcertainty_output = model(input_ids, attention_mask) # from the forward function

        loss = 0.25*criterion['type'](vtype_output, vtype_label.float()) + \
               0.25*criterion['polarity'](vpolarity_output, vpolarity_label.float()) + \
               0.25*criterion['tense'](vtense_output, vtense_label.float()) + \
               0.25*criterion['certainty'](vcertainty_output, vcertainty_label.float())

        total_loss_val += loss.item()

    print(f'Epochs: {epoch + 1} '
    f'| Train Loss: {total_loss_train / len(train_loader): .3f} '
    f'| Val Loss: {total_loss_val / len(valid_loader): .3f} ')
    torch.save(model, str(path)+'model'+str(fold)+'.pt')


def main(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # kfold
    kfold=[]

    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=2021)
    for train_idx, val_idx in splitter.split(train_1.iloc[:, :-1],train_1.iloc[:, -1]):
        kfold.append((train_1.iloc[train_idx,:],train_1.iloc[val_idx,:]))

    for fold,(train_datasets, valid_datasets) in enumerate(kfold):
        print(f'fold{fold} 학습중...')
        training(train_dataset=pd.concat([train_datasets, train_2]).reset_index(drop=True),val_dataset=valid_datasets.reset_index(drop=True), base_model= base_model, tokenizer=tokenizer, fold=fold,epochs=args.epochs, batch=args.batch_size, device=args.device)

if __name__ == "__main__":
    args = parse_args()
    main(args)
        