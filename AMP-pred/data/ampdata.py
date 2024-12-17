from transformers import AutoTokenizer

class AmpData():
    def __init__(self, df, tokenizer_name='/data3/lyr/project_AMP_pre/PeptideBERT-master/prot_bert_bfd', max_len=300):
       
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
        
        sample = {key: val for key, val in seq_ids.items()}
        sample['labels'] = self.labels[idx]
        return sample
