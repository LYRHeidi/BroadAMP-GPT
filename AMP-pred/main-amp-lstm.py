import torch
import wandb
from datetime import datetime
import yaml
import os
import shutil
# from data.dataloader import load_data
from model.networklstm import create_model, cri_opt_sch
from model.utils import train, validate, test

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

def train_model():
    print(f'{"="*30}{"TRAINING":^20}{"="*30}')

    best_acc = 0
    for epoch in range(config['epochs']):
        train_loss = train(model, train_data_loader, optimizer, criterion, scheduler, device)
        curr_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch+1}/{config["epochs"]} - Train Loss: {train_loss}\tLR: {curr_lr}')
        val_loss, val_acc = validate(model, val_data_loader, criterion, device)
        print(f'Epoch {epoch+1}/{config["epochs"]} - Validation Loss: {val_loss}\tValidation Accuracy: {val_acc}\n')
        scheduler.step(val_acc)
        if not config['debug']:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'lr': curr_lr
            })

        if val_acc >= best_acc and not config['debug']:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'acc': val_acc,
                'lr': curr_lr
            }, f'{save_dir}/model.pt')
            print('Model Saved\n')
    wandb.finish()


config = yaml.load(open('./config.yaml', 'r'), Loader=yaml.FullLoader)
config['device'] = device

#####################################################################################
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from data.dataset import PeptideBERTDataset
from transformers import AutoTokenizer

class AmpData:
    def __init__(self, df, tokenizer_name='Rostlab/prot_bert_bfd', max_len=300):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_len = max_len

        self.seqs, self.labels = self.get_seqs_labels()

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

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample

print(f'{"="*30}{"DATA":^20}{"="*30}')

traindf = pd.read_csv(f'./data/{config["task"]}/train.csv', index_col = 0)
valdf = pd.read_csv(f'./data/{config["task"]}/val.csv', index_col = 0)
testdf = pd.read_csv(f'./data/{config["task"]}/test.csv', index_col = 0)
traindf = traindf.sample(frac=1, random_state = 0)
valdf = valdf.sample(frac=1, random_state = 0)
testdf = testdf.sample(frac=1, random_state = 0)

df=traindf
train_dataset = AmpData(df)
# train_dataset = AmpData(traindf)
df=valdf
val_dataset = AmpData(df)
# val_dataset = AmpData(valdf)
df=testdf
test_dataset = AmpData(df)
# test_dataset = AmpData(testdf)
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
#####################################################################################
# train_data_loader, val_data_loader, test_data_loader = load_data(config)
config['sch']['steps'] = len(train_data_loader)

model = create_model(config)
print(model)
criterion, optimizer, scheduler = cri_opt_sch(config, model)

if not config['debug']:
    run_name = f'{config["task"]}-{datetime.now().strftime("%m%d_%H%M")}'
    wandb.init(project='PeptideBERT', name=run_name)

    save_dir = f'./checkpoints/{run_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    shutil.copy('./config.yaml', f'{save_dir}/config.yaml')
    # shutil.copy('./model/networklstm.py', f'{save_dir}/network.py')

train_model()
if not config['debug']:
    model.load_state_dict(torch.load(f'{save_dir}/model.pt')['model_state_dict'], strict=False)

test_acc = test(model, test_data_loader, device)
print(f'Test Accuracy: {test_acc}%')