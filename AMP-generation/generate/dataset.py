import torch
from torch.utils.data import Dataset
from utils import SmilesEnumerator
import numpy as np
import re
import pandas as pd
import math
import random

pattern = "(\[[^\]]+]|<|A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)


def is_valid(peptide):
    pep_dict = {'<', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z'}

    whole_string = ' '.join(peptide)
    whole_string = sorted(list(set(regex.findall(whole_string))))

    flag = [False for data in whole_string if data not in pep_dict]

    peptide = peptide.replace('<', '')
    lens = len(regex.findall(peptide.strip()))
    # print(lens)

    if flag:
        return False
    elif lens >= 1 and lens < 300:
        return True
    else:
        # print(peptide)
        return False


def read_peptide(filename):
    # filename = '/root/workspace/fentn/molgpt/data/train_Escherichia_coli.csv'
    data = pd.read_csv(filename, encoding='ISO-8859-1')

    data = data.dropna(axis=0).reset_index(drop=True)

    peptide = data['sequence'].str.upper()

    # pattern = "(\[[^\]]+]|<|A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    # regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip()))
            for i in (list(peptide.values))]
    max_len = max(lens)
    print('Max len: ', max_len)

    peptide_pad = [i + str('<') * (max_len - len(regex.findall(i.strip())))
                   for i in peptide]

    whole_string = ' '.join(peptide_pad)
    whole_string = sorted(list(set(regex.findall(whole_string))))
    print(whole_string)

    data = np.array(peptide)
    random.shuffle(data)
    traindata = data[:int(0.8 * len(data))]
    testdata = data[int(0.8 * len(data)):]

    traindata = traindata.tolist()
    testdata = testdata.tolist()

    # data = traindata[0]

    # traindata = pd.DataFrame(traindata)
    # testdata = pd.DataFrame(testdata)

    return traindata, testdata, whole_string, max_len


class SmileDataset(Dataset):

    def __init__(self, args, data, content, block_size, aug_prob=0.5, prop=None):
        chars = sorted(list(set(content)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d peptide, %d unique characters.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = block_size
        self.vocab_size = vocab_size
        self.data = data
        self.prop = prop
        # self.sca = scaffold
        # self.scaf_max_len = scaffold_maxlen
        self.debug = args.debug
        self.tfm = SmilesEnumerator()
        self.aug_prob = aug_prob

    def __len__(self):
        if self.debug:
            return math.ceil(len(self.data) / (self.max_len + 1))
        else:
            return len(self.data)

    def __getitem__(self, idx):
        # smiles, prop= self.data[idx], self.prop[idx]   # self.prop.iloc[idx, :].values  --> if multiple properties
        smiles = self.data[idx]
        smiles = smiles.strip()
        # scaffold = scaffold.strip()

        p = np.random.uniform()
        if p < self.aug_prob:
            smiles = self.tfm.randomize_smiles(smiles)

        # pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        # regex = re.compile(pattern)
        smiles += str('<') * (self.max_len - len(regex.findall(smiles)))

        if len(regex.findall(smiles)) > self.max_len:
            smiles = smiles[:self.max_len]

        smiles = regex.findall(smiles)

        # scaffold += str('<')*(self.scaf_max_len - len(regex.findall(scaffold)))

        # if len(regex.findall(scaffold)) > self.scaf_max_len:
        #     scaffold = scaffold[:self.scaf_max_len]

        # scaffold=regex.findall(scaffold)

        dix = [self.stoi[s] for s in smiles]
        # sca_dix = [self.stoi[s] for s in scaffold]

        # sca_tensor = torch.tensor(sca_dix, dtype=torch.long)
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        # prop = torch.tensor([prop], dtype=torch.long)

        # prop = torch.tensor([prop], dtype = torch.float)
        return x, y
        # return x, y, prop