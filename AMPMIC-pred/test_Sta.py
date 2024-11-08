import os
import sys
import torch
import pandas as pd

sys.path.insert(0, "../")
from collections import OrderedDict
from linformer_pytorch import LinformerLM
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np

config = OrderedDict(
    batch_size=32,
    lr=0.001,
    gamma=0.9,
    no_cuda=False,
    num_epochs=100,
    output_dir="./output",
    logs_dir="./output",
    seed=22,
    num_tokens=22, 
    seq_len=190,
    ch=64,
    dummy_d=64,
    weightfile = "./weights/Sta_weight/best_checkpoint.pth",
    filename = './data/test.txt'
)

def get_model(device):
    """
    Gets the device that the model is running on. Currently running standard linformer
    """
    model = LinformerLM(config["num_tokens"], input_size=config["seq_len"], channels=config["ch"], dim_k=20, dim_ff=200,
                        nhead=8, depth=6, activation="gelu", checkpoint_level="C0", causal=True, dropout=0.2,
                        dropout_ff=0.2)
    model.to(device)
    return model

def get_optimizer(model_parameters):
    """
    Gets an optimizer. Just as an example, I put in SGD

    """
    return torch.optim.SGD(model_parameters, lr=config["lr"])

import re
pattern = "(\[[^\]]+]|<|A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)
class TestDataset(Dataset):
    def __init__(self, data, content, block_size, aug_prob=0.5, prop=None):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data['sequence']), len(chars)
        print('data has %d peptide, %d unique characters.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data['sequence']
        self.prop = prop
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        smiles = self.data[idx]
        smiles = smiles.strip()

        smiles += str('<') * (self.max_len - len(regex.findall(smiles)))

        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]

        smiles = regex.findall(smiles)

        dix = [self.stoi[s] for s in smiles]

        x = torch.tensor(dix, dtype=torch.float32)
        return x

def read_peptide(filename):
    data = pd.read_csv(filename, encoding='ISO-8859-1')
    data = data.dropna(axis=0).reset_index(drop=True)
    peptide = data['sequence'].str.upper()

    lens = [len(regex.findall(i.strip()))
            for i in (list(peptide.values))]
    max_len = max(lens)
    print('Max len: ', max_len)

    peptide_pad = [i + str('<') * (max_len - len(regex.findall(i.strip())))
                   for i in peptide]

    whole_string = ' '.join(peptide_pad)
    whole_string = sorted(list(set(regex.findall(whole_string))))
    print(whole_string)
    print(len(whole_string))

    data = np.array(peptide).tolist()

    return data, whole_string, max_len

def get_loader(test=False):
    """
    Gets data and a loader. Just dummy data, but replace with what you want
    """
    data = pd.read_csv(config['filename'], encoding='ISO-8859-1')

    data = data.dropna(axis=0).reset_index(drop=True)
    chars = ['<', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X',
             'Y']

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    dataset = TestDataset(data, chars, 190)
    return DataLoader(dataset, batch_size=config["batch_size"], num_workers=2)

def main():
    """
    Train a model
    """
    global output_dir
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(config["seed"])

    device = torch.device("cuda" if not config["no_cuda"] and torch.cuda.is_available() else "cpu")

    model = get_model(device)

    test_loader = get_loader(test=True)

    dict = torch.load(config['weightfile'])
    model.load_state_dict(dict['state_dict'])

    model.eval()
    with torch.no_grad():

        predictions = []
        for batch_x in tqdm(test_loader):
            batch_x = batch_x.to(device)
            prediction = model(batch_x)
            predictions.extend(prediction.cpu().detach().numpy())

        predictions = np.array(predictions)
    
        predictions = np.exp(predictions[:, 0])

    pd.DataFrame(predictions).to_csv(config['output_dir'] + './Sta_test_result.csv', sep="\t", header=False, index=False)

if __name__ == "__main__":
    main()
