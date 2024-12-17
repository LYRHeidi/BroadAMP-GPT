import torch
from torch import nn
from transformers import BertModel, BertConfig, logging

logging.set_verbosity_error()

class PeptideBERT(torch.nn.Module):
    def __init__(self, bert_config):
        super(PeptideBERT, self).__init__()

        self.protbert = BertModel.from_pretrained(
            'Rostlab/prot_bert_bfd',
            ignore_mismatched_sizes=True
        )
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=64, kernel_size=16),
            nn.Tanh(),
            nn.MaxPool1d(kernel_size=5, stride=1)
        )

        self.lstm = nn.LSTM(input_size=64, hidden_size=100,
                            batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, inputs, attention_mask):
        with torch.no_grad():
            output = self.protbert(inputs, attention_mask=attention_mask)

        x = torch.Tensor(output.last_hidden_state)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = torch.Tensor(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.fc(x)
        x = x[:, -1, :]
        return x
    

def create_model(config):
    bert_config = BertConfig(
        vocab_size=config['vocab_size'],
        hidden_size=config['network']['hidden_size'],
        num_hidden_layers=config['network']['hidden_layers'],
        num_attention_heads=config['network']['attn_heads'],
        hidden_dropout_prob=config['network']['dropout']
    )
    model = PeptideBERT(bert_config).to(config['device'])

    return model


def cri_opt_sch(config, model):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['optim']['lr'])

    if config['sch']['name'] == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['optim']['lr'],
            epochs=config['epochs'],
            steps_per_epoch=config['sch']['steps']
        )
    elif config['sch']['name'] == 'lronplateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=config['sch']['factor'],
            patience=config['sch']['patience']
        )

    return criterion, optimizer, scheduler
