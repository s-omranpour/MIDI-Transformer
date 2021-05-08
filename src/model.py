import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np
import pickle
import itertools
from tqdm.notebook import tqdm
from fast_transformers.masking import TriangularCausalMask, LengthMask

from src.modules import CPEmbedding, CPHeadLayer, Encoder, utils
from deepnote.utils import clean_cp


class CPTransformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        self.emb = CPEmbedding(config['embedding'])
        self.encoder = Encoder(config['encoder'])
        self.head = CPHeadLayer(config['head'])
        self.head.register_embeddings(self.emb.emb_layers['ttype'])
        

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.config['lr'])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.config['max_epochs'], eta_min=0.)
        return [opt], [sch]

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x, x_length_mask=None, y=None):
        h = self.emb(x)
        h = self.encoder(h, x_length_mask=x_length_mask)
        return self.head(h, y)

    
    def calculate_loss(self, logits, labels):
        losses = {}

        mask = torch.zeros_like(labels)
        mask[:, :, 0] = 1.                              ## ttype
        for i in [1,2,3]:                               ## metrical
            mask[:, :, i] = (labels[:, :, 0] == 0)
        for i in [4,5,6,7]:                             ## note
            mask[:, :, i] = (labels[:, :, 0] == 1)

        for i,k in enumerate(logits.keys()):
            l = self.criterion(logits[k].transpose(1,2), labels[..., i]) * mask[..., i]
            s = torch.sum(mask[..., i])
            l = torch.mean(torch.sum(l) / s if s > 0. else torch.sum(l))
            losses[k] = 0.*l if torch.isnan(l) else l
        return losses


    def step(self, batch, mode='train'):
        logits = self.forward(x=batch['X'], 
                              x_length_mask=batch['X_len'], 
                              y=batch['labels'])

        losses = self.calculate_loss(logits, batch['labels'].long())
        total_loss = sum(losses.values()) / len(logits)
        for attr in losses:
            self.log(mode + '_' + attr, losses[attr].item())
        self.log(mode + '_loss', total_loss.item())
        return total_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch)

    def validation_step(self, batch, batch_idx):
        self.step(batch, mode='val')

    def generate(self, prompt=None, max_len=1024, temperatures=None):
        self.eval()
        with torch.no_grad():
            res = [[0]*8] if prompt is None else prompt
            for _ in tqdm(range(max_len)):
                inp = torch.tensor(res).long().to(self.device).unsqueeze(0)
                h = self.encoder(self.emb(inp))
                next_word = self.head.infer(h, temperatures)
                res += [next_word]
        return clean_cp(np.array(res))