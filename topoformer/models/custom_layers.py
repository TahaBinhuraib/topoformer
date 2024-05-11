import torch.nn as nn
import torch.nn.functional as F
import torch
from .device import DEVICE
from .constraints import Constraints
import logging


class LocalityMask(nn.Module):
    def __init__(self,
                 hidden_size=400,
                 width=0.3,
                 wrap=True,
                 local_querying=True,
                 learned_spatial_querying=True,
                 mask_type="circular",
                 normalize_mask=True,
                 ):
        super().__init__()
        
        self.local_querying = local_querying
        self.learned_spatial_querying = learned_spatial_querying
        self.normalize_queries = normalize_mask
        self.width = width
        
        if not local_querying:
            logging.info(f"Using spatial querying with width: {width}")
            self.constraint = Constraints(hidden_size, width, wrap)
            locality_mask = getattr(self.constraint, mask_type)()
            self.register_buffer('locality_mask', torch.Tensor(locality_mask).to(DEVICE))
            if self.normalize_queries:
                print('Normalizing queries...')
                self.locality_mask = self.locality_mask/torch.norm(self.locality_mask)

            if learned_spatial_querying:
                self.locality_weight = nn.Parameter(torch.randn(hidden_size))
            else:
                self.locality_weight = 1

    def forward(self, queries):
        # If width -1 then return only queries(same as multiplying with identity) If local querying, then no locality operation is performed on queries.
        if self.width == -1 or self.local_querying:
            return queries
        
        # Locality operations.
        locality_mask = self.locality_mask
        if self.learned_spatial_querying:
            locality_mask = locality_mask * self.locality_weight
    
        queries = torch.matmul(queries, locality_mask)
        return queries
    

class LocallyConnected(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        width=0.3,
        wrap=True,
        bias=True,
        feedforward=False,
        mask_type='circular'
    ):
        super().__init__(in_features, out_features, bias=bias)

        if width != -1:
            self.weight.data = torch.abs(self.weight.data) * 10
            logging.info(f"using mask_type: {mask_type} in locally connected...")

            self.locality_constraint = getattr(Constraints(in_features, width, wrap), mask_type)()
            self.locality_constraint = torch.Tensor(self.locality_constraint).to(DEVICE)
        else:
            self.locality_constraint = None

    def forward(self, input):
        if self.locality_constraint is not None:
            self.weight.data = torch.abs(
                self.weight.data * self.locality_constraint.transpose(0, 1)
            )
        return F.linear(input, self.weight, self.bias)
