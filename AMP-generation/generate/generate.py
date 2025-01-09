from utils import  sample
import math
from tqdm import tqdm
import argparse
from model import GPT, GPTConfig
import pandas as pd
import torch
import numpy as np
import re

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_weight', type=str, default='AMP_Escherichia_nopretrain.pt', help="path of model weights", required=False)
        parser.add_argument('--lstm', action='store_true', default=False, help='use lstm for transforming scaffold')
        parser.add_argument('--csv_name', type=str, default='gen_AMP', help="name to save the generated mols in csv format", required=False)
        parser.add_argument('--batch_size', type=int, default = 8, help="batch size", required=False)
        parser.add_argument('--gen_size', type=int, default = 560000, help="number of times to generate from a batch", required=False)
        parser.add_argument('--vocab_size', type=int, default = 22, help="number of layers", required=False) 
        parser.add_argument('--block_size', type=int, default = 132, help="number of layers", required=False)   
        parser.add_argument('--props', nargs="+", default = [], help="properties to be used for condition", required=False)
        parser.add_argument('--n_layer', type=int, default = 8, help="number of layers", required=False)
        parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
        parser.add_argument('--n_embd', type=int, default = 256, help="embedding dimension", required=False)
        parser.add_argument('--lstm_layers', type=int, default = 2, help="number of layers in lstm", required=False)
        parser.add_argument('--sequence_size', type=int, default = 132, help="number of generate sequence length", required=False)
        args = parser.parse_args()

        frequence = {'A': 0.06152815486054793,
                     'C': 0.02912450195691266,
                     'D': 0.022777758189062445,
                     'E': 0.020415359119918198,
                     'F': 0.07986319241211523,
                     'G': 0.15094672261203765,
                     'H': 0.011494658157328726,
                     'I': 0.04717746200768661,
                     'K': 0.11075067874898628,
                     'L': 0.06498360424526639,
                     'M': 0.10422763654314023,
                     'N': 0.015866859419625543,
                     'P': 0.019004971615951483,
                     'Q': 0.019780684743133176,
                     'R': 0.08991220337787807,
                     'S': 0.03529494728676704,
                     'T': 0.022248862875074927,
                     'V': 0.043968830436162336,
                     'W': 0.033249885406015305,
                     'Y': 0.01738302598638976}

        pattern = "(\[[^\]]+]|<|A|B|C|D|E|F|G|H|I|J|K|L|M|N|O|P|Q|R|S|T|U|V|W|X|Y|Z|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        chars = ['<', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']  
        stoi = { ch:i for i,ch in enumerate(chars)}
        itos = { i:ch for ch,i in stoi.items() }
        print(itos)
        print(len(itos))

        num_props = len(args.props)
        mconf = GPTConfig(args.vocab_size, args.block_size, num_props = num_props,
                       n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
                       lstm = args.lstm, lstm_layers = args.lstm_layers)

        model = GPT(mconf)

        model.load_state_dict(torch.load(args.model_weight))
        model.to('cuda')
        print('Model loaded')

        gen_iter = math.ceil(args.gen_size / args.batch_size)
        contexts = np.random.choice(list(frequence.keys()), gen_iter, p=list(frequence.values()))

        all_dfs = []
        all_metrics = []
        
        count = 0

        molecules = []
        generate_peptides = []
        count += 1
        for i in tqdm(range(gen_iter)):
                context = contexts.tolist()[i]
                x = torch.tensor([stoi[s] for s in regex.findall(context)], dtype=torch.long)[None,...].repeat(args.batch_size, 1).to('cuda')  
                # y = sample(model, x, args.block_size, temperature=1, sample=True, top_k=10)
                y = sample(model, x, args.sequence_size-1, temperature=1, sample=True, top_k=10)
                for gen_mol in y:
                    completion = ''.join([itos[int(i)] for i in gen_mol])
                    completion = completion.replace('<', '')
                    generate_peptides.append(completion)

        # unique_smiles = list(set(generate_peptides))
        unique_smiles = generate_peptides

        results = pd.DataFrame(unique_smiles)
        all_dfs.append(results)

        results = pd.concat(all_dfs)

results.columns = ['sequence']
results.to_csv('./gen_amp/' + args.csv_name + '.csv', index = None)


