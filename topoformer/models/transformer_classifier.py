import math
import torch
import torch.nn as nn
import scipy.spatial
import numpy as np
from .constraints import Constraints
from .device import DEVICE
from .custom_layers import LocallyConnected
from .attention import AttentionOutput
import random
import logging

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()

        self.attention = AttentionOutput(config)
        self.norm1 = nn.LayerNorm(config['hidden_size'])
        self.norm2 = nn.LayerNorm(config['hidden_size'])
        self.dropout1 = nn.Dropout(config['dropout'])
        self.dropout2 = nn.Dropout(config['dropout'])

    def forward(self, embeddings, mask):
        attention, _ = self.attention(embeddings, mask)

        # Add skip connection
        x = self.dropout1(self.norm1(attention + embeddings))
        forward = self.feed_forward(x)
        out = self.dropout2(self.norm2(forward + x))
        return out

class TransformerBlockDefault(TransformerBlock):
    def __init__(self, config):
        super().__init__(config)

        self.feed_forward = nn.Sequential(
            nn.Linear(config['hidden_size'], config['forward_expansion'] * config['hidden_size']),
            nn.ReLU(),
            nn.Linear(config['forward_expansion'] * config['hidden_size'], config['hidden_size']),
        )

class TransformerBlockV3(TransformerBlock):
    def __init__(self, config):
        super().__init__(config)

        dim = config['hidden_size']*config['forward_expansion']
        assert dim == math.isqrt(dim) ** 2, 'expanded FF dimension must be perfect square to support spatial constraints'
        if width_ff is None:
            # default to using the spatial reweighting width
            width_ff = config['width_sr']
        self.feed_forward = nn.Sequential(
            LocallyConnected(config['hidden_size'], dim, DEVICE=DEVICE, width=width_ff, wrap=config['wrap'], feedforward=True),
            nn.ReLU(),
            LocallyConnected(dim, config['hidden_size'], DEVICE=DEVICE, width=width_ff, wrap=config['wrap'], feedforward=True),
        )



class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.word_embedding = nn.Embedding(config['src_vocab_size'], config['hidden_size'])
        self.position_embedding = nn.Embedding(config['max_length'], config['hidden_size'])

        TransformerBlockClass = TransformerBlockV3 if config['model_version'] == 'v3' else TransformerBlockDefault
        self.layers = nn.ModuleList(
            [
                TransformerBlockClass(config)
                for _ in range(config['num_hidden_layers'])
            ]
        )

        self.dropout = nn.Dropout(config['dropout'])

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(DEVICE)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            out = layer(out, mask)
        return out


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        config,
        src_pad_idx
    ):
        super(TransformerClassifier, self).__init__()

        self.encoder = Encoder(
            config
        )
        self.classifier = nn.Linear(config['hidden_size'], 2)
        self.act = nn.Tanh()

        self.src_pad_idx = src_pad_idx
        self.pooling_mechanism = config['pooling_mechanism']

    def make_src_mask(self, src):
        src_mask = (src == self.src_pad_idx).unsqueeze(1)
        return src_mask

    def forward(self, src):
        src_mask = self.make_src_mask(src)
        enc_src = self.encoder(src, src_mask)
        # add pooling layer for downstream task
        if self.pooling_mechanism == "AVERAGE":
            pooled = enc_src.mean(dim=1)
        elif self.pooling_mechanism == "CLASSIFICATION":
            pooled = enc_src[:, 0, :]

        out = self.act(pooled)
        out = self.classifier(out)
        return out