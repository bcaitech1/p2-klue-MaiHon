from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from load_data import *
import pandas as pd
import torch
import random
import numpy as np
import argparse
from my_model import BertForSequenceClassification, XLMRobertaForSequenceClassification


def pred_answrs(model, test_dl, device):
    output_pred = []

    for i, data in enumerate(test_dl):
        with torch.no_grad():
            data = tuple(t.to(device) for t in data)
            data = {'input_ids': data[0],
			          'attention_mask': data[1],
			          'token_type_ids': data[2],
			          'e1_mask': data[4],
			          'e2_mask': data[5],
			          }
            logits = 0
            for m in model:
                m.eval()
                logits += m(**data)[0].detach().cpu().numpy()
        result = np.argmax(logits, axis=-1)
        output_pred.extend(result)
    return np.array(output_pred).reshape(-1, 1)


def load_test_dataset(dataset_dir, tokenizer, xlm=False, max_len=None):
	test_dataset = load_data2(dataset_dir, '../data/label_type.pkl')
	test_label = test_dataset['label'].values
	# tokenizing dataset
	features = tokenized_dataset2(test_dataset, tokenizer, xlm=xlm, max_length=max_len)
	return features, test_label


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args, version):
    seed_everything()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model_info = [
        # '../results/Enriching-250_v2/checkpoint-best',
        # '../results/Enriching-250_v3/checkpoint-best',
        # '../results/Enriching-250_v4/checkpoint-best',
        # '../results/Enriching-250_v5/checkpoint-best',
        # '../results/Enriching-250_v6/checkpoint-best',
        '../results/Enriching_Kfold_XLM-250_v1/0_checkpoint-best_acc',
        '../results/Enriching_Kfold_XLM-250_v1/1_checkpoint-best_acc',
        '../results/Enriching_Kfold_XLM-250_v1/2_checkpoint-best_acc',
        '../results/Enriching_Kfold_XLM-250_v1/3_checkpoint-best_acc',
        '../results/Enriching_Kfold_XLM-250_v1/4_checkpoint-best_acc',
    ]

    models = []
    # TOK_NAME = 'bert-base-multilingual-cased'
    TOK_NAME = 'xlm-roberta-base'
    for info in model_info:
        tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

        # load my model
        if TOK_NAME.startswith("bert"):
            model = BertForSequenceClassification.from_pretrained(info)
        elif TOK_NAME.startswith("xlm"):
            model = XLMRobertaForSequenceClassification.from_pretrained(info)
        model.parameters
        model.to(device)
        models.append(model)

    # load test datset
    test_dataset_dir = "../data/test/test.tsv"
    xlm = True if TOK_NAME.startswith("xlm") else False
    test_features, test_label = load_test_dataset(test_dataset_dir, tokenizer, xlm=xlm, max_len=args.max_len)
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_e1_mask = torch.tensor([f.e1_mask for f in test_features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor([f.e2_mask for f in test_features], dtype=torch.long)  # add e2 mask
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    test_ds = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_e1_mask, all_e2_mask)
    test_dl = torch.utils.data.DataLoader(
		test_ds,
		batch_size=32,
		shuffle=False,
		num_workers=3)

    # predict answer
    pred_answer = pred_answrs(models, test_dl, device)


    # make csv file with predicted answer
    # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.
    output = pd.DataFrame(pred_answer, columns=['pred'])
    output.to_csv(f'../prediction/[{args.postfix}]pred_answer.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

	# model dir
    parser.add_argument('--postfix', type=str, default='v1')
    parser.add_argument('--max_len', type=int, default=100)
    args = parser.parse_args()
    print(args)

    main(args, args.postfix)