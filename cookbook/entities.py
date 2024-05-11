from dataclasses import dataclass

@dataclass
class ViTModelConfig:
    patch_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    hidden_dropout_prob: float
    attention_probs_dropout_prob: float
    initializer_range: float
    image_size: int
    num_classes: int
    num_channels: int
    qkv_bias: bool
    sr: float
    sq: float
    learned_spatial_querying: bool
    local_querying: bool
    mask_type: str
    relu: bool

    def __getitem__(self, item):
        return self.__dict__[item]


@dataclass
class TextModelConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    forward_expansion: int
    qkv_bias: bool
    dropout: float
    sr: float
    sq: float
    max_length: int
    learned_spatial_querying: bool
    local_querying: bool
    hidden_dropout_prob: float
    pooling_mechanism: str
    attention_probs_dropout_prob: float
    mask_type: str
    relu: bool
    src_vocab_size: int
    model_version: str

    def __getitem__(self, item):
        return self.__dict__[item]