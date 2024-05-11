from dataclasses import dataclass
from typing import List


@dataclass
class ViTLayers:
    n_layers: int

    def layers(self, prepend='') -> List[str]:
        layer_names = []
        for i in range(self.n_layers):
            layer_names.append(f"{prepend}encoder.blocks.{i}.attention.head.query")
            layer_names.append(f"{prepend}encoder.blocks.{i}.attention.head.value")
            layer_names.append(f"{prepend}encoder.blocks.{i}.attention.head.key")
            layer_names.append(f"{prepend}encoder.blocks.{i}.attention.output_projection")
        return layer_names


@dataclass
class GRULayers:
    layers = [
        "rnn.rnn_cell_list.0.h_upd",
        "rnn.rnn_cell_list.0.h_reset",
        "rnn.rnn_cell_list.0.h_new",
    ]

@dataclass
class TransformerLayers:
    weight_tying: bool = False
    layers = [
        "encoder.layers.0.attention.head.value",
        "encoder.layers.0.attention.head.key",
        'encoder.layers.0.attention.head.query',
        "encoder.layers.0.attention.output_projection",
    ]


@dataclass  
class RNNLayers:  
    n_layers: int  
      
    @property  
    def layers(self) -> List[str]:  
        layer_names = []  
        for i in range(self.n_layers):  
            layer_names.append(f"rnn.rnn_cell_list.{i}.h2h")  
            layer_names.append(f"rnn.rnn_cell_list.{i}.x2h")  
        return layer_names