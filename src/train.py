import pickle as pickle
import os
import pandas as pd
import torch
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from transformers import Trainer, TrainingArguments
from importlib import import_module
from load_data import *
import wandb
import numpy as np
import random

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train():
  seed_everything()
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', default='bert-base-multilingual-cased')
  parser.add_argument('--version', default='v3', type=str)
  parser.add_argument('--valid_ratio', type=float, default=0.1)
  parser.add_argument('--epochs', type=int, default=5)
  parser.add_argument('--lr', type=float, default=5e-5)
  parser.add_argument('--weight_decay', type=float, default=0.001)
  parser.add_argument('--max_len', type=int, default=100)

  args = parser.parse_args()

  os.environ['WANDB_PROJECT'] = f'[Pstage-NLP]'
  wandb.login()

  # load model and tokenizer
  MODEL_NAME = args.model_name
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


  # load dataset
  RE_valid_dataset = None
  evaluation_startegy = 'no'
  total_dataset = load_aug_data("../data/train/aug_train.tsv")
  if args.valid_ratio > 0.0:
    train_dataset, valid_dataset = train_test_split(total_dataset,
                                                    test_size=args.valid_ratio, random_state=42,
                                                    shuffle=True, stratify=total_dataset.label)
    valid_label = valid_dataset['label'].values
    tokenized_valid = tokenized_dataset(valid_dataset, tokenizer, max_length=args.max_len)
    RE_valid_dataset = RE_Dataset(tokenized_valid, valid_label)
    evaluation_startegy = 'steps'
  else:
    train_dataset = total_dataset


  train_label = train_dataset['label'].values
  tokenized_train = tokenized_dataset(train_dataset, tokenizer, max_length=args.max_len)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_label)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


  # setting model hyperparameter
  model_config = AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 42
  model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=model_config)
  model.parameters
  model.to(device)


  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=f'../results/{args.version}',           # output directory
    save_total_limit=3,              # number of total save model.
    save_steps=100,                   # model saving step.
    num_train_epochs=args.epochs,     # total number of training epochs
    learning_rate=args.lr,            # learning_rate
    per_device_train_batch_size=16,   # batch size per device during training
    per_device_eval_batch_size=16,    # batch size for evaluation
    warmup_steps=500,                 # number of warmup steps for learning rate scheduler
    weight_decay=args.weight_decay,               # strength of weight decay
    report_to= ['wandb'],
    logging_steps=100,                # log saving step.
    evaluation_strategy=evaluation_startegy,         # evaluation strategy to adopt during training
                                      # `no`: No evaluation during training.
                                      # `steps`: Evaluate every `eval_steps`.
                                      # `epoch`: Evaluate every end of epoch.
    eval_steps = 500,               # evaluation step.
    seed = 42,
  )

  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_valid_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()

if __name__ == '__main__':
  train()