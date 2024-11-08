from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from data.dataset import PeptideBERTDataset
from transformers import AutoTokenizer
import torch

class AmpData:
    def __init__(self, dfname, tokenizer_name='/data3/lyr/project_AMP_pre/PeptideBERT-master/prot_bert_bfd', max_len=300):
       
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_len = max_len
        
        self.seqs, self.labels = self.get_seqs_labels()
        # self.dfname = dfname # 传递一个名字给它，而不是一个dataframe
        
    def get_seqs_labels(self):        
        # isolate the amino acid sequences and their respective AMP labels
        seqs = list(df['aa_seq'])
        labels = list(df['AMP'].astype(int))
        
#         assert len(seqs) == len(labels)
        return seqs, labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        seq = " ".join("".join(self.seqs[idx].split()))
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_len)
        
        # sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        # sample['labels'] = torch.tensor(self.labels[idx])
        sample = {key: val for key, val in seq_ids.items()}
        sample['labels'] = self.labels[idx]
        return sample

def load_data(config):
    print(f'{"="*30}{"DATA":^20}{"="*30}')

    # with np.load(f'./data/{config["task"]}/train.npz') as train,\
    #      np.load(f'./data/{config["task"]}/val.npz') as val,\
    #      np.load(f'./data/{config["task"]}/test.npz') as test:
    #     train_inputs = train['inputs']
    #     train_labels = train['labels']
    #     val_inputs = val['inputs']
    #     val_labels = val['labels']
    #     test_inputs = test['inputs']
    #     test_labels = test['labels']
    train = pd.read_csv(f'./data/{config["task"]}/train.csv', index_col = 0)
    val = pd.read_csv(f'./data/{config["task"]}/val.csv', index_col = 0)
    test = pd.read_csv(f'./data/{config["task"]}/test.csv', index_col = 0)
    train = train.sample(frac=1, random_state = 0)
    val = val.sample(frac=1, random_state = 0)
    test = test.sample(frac=1, random_state = 0)
    
    train_dataset = AmpData(train)
    val_dataset = AmpData(val)
    test_dataset = AmpData(test)

    train_labels = []
    attention_mask = []
    train_inputs = []
    for index, content in enumerate(train_dataset):
        train_labels.append(content['labels'])
        attention_mask.append(content['attention_mask'])
        train_inputs.append(content['input_ids'])
    train_labels = np.array(train_labels)
    attention_mask = np.array(attention_mask)
    train_inputs = np.array(train_inputs)

    val_labels = []
    attention_mask_val = []
    val_inputs = []
    for index, content in enumerate(val_dataset):
        val_labels.append(content['labels'])
        attention_mask_val.append(content['attention_mask'])
        val_inputs.append(content['input_ids'])
    val_labels = np.array(val_labels)
    attention_mask_val = np.array(attention_mask_val)
    val_inputs = np.array(val_inputs)

    test_labels = []
    attention_mask_test = []
    test_inputs = []
    for index, content in enumerate(test_dataset):
        test_labels.append(content['labels'])
        attention_mask_test.append(content['attention_mask'])
        test_inputs.append(content['input_ids'])
    test_labels = np.array(test_labels)
    attention_mask_test = np.array(attention_mask_test)
    test_inputs = np.array(test_inputs)

    train_dataset = PeptideBERTDataset(input_ids=train_inputs, attention_masks=attention_mask, labels=train_labels)
    val_dataset = PeptideBERTDataset(input_ids=val_inputs, attention_masks=attention_mask_val, labels=val_labels)
    test_dataset = PeptideBERTDataset(input_ids=test_inputs, attention_masks=attention_mask_test, labels=test_labels)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    print('Batch size: ', config['batch_size'])

    print('Train dataset samples: ', len(train_dataset))
    print('Validation dataset samples: ', len(val_dataset))
    print('Test dataset samples: ', len(test_dataset))

    print('Train dataset batches: ', len(train_data_loader))
    print('Validataion dataset batches: ', len(val_data_loader))
    print('Test dataset batches: ', len(test_data_loader))

    print()

    return train_data_loader, val_data_loader, test_data_loader
