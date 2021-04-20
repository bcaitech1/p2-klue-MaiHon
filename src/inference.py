from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
from importlib import import_module


def pred_answrs(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
          )
    logits = outputs[0]
    logits = logits.detach().cpu().numpy()
    result = np.argmax(logits, axis=-1)
    output_pred.append(result)

  return list(np.array(output_pred).reshape(-1))


def load_test_dataset(dataset_dir, tokenizer):
  test_dataset = load_data(dataset_dir, '../data/label_type.pkl')
  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer)
  return tokenized_test, test_label


def main(args, version):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  TOK_NAME = args.model_type
  tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

  # load my model
  model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
  model.parameters
  model.to(device)

  # load test datset
  test_dataset_dir = "../data/test/test.tsv"
  test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer)
  test_dataset = RE_Dataset(test_dataset, test_label)

  # predict answer
  pred_answer = pred_answrs(model, test_dataset, device)
  
  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
  output = pd.DataFrame(pred_answer, columns=['pred'])
  output.to_csv(f'../prediction/[{version}]pred_answer.csv', index=False)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_type', default="bert-base-multilingual-cased", type=str)
  parser.add_argument('--model_dir', type=str, default="../results/v2/checkpoint-5500")
  args = parser.parse_args()
  print(args)

  version = args.model_dir.split("/")[2]
  main(args, version)