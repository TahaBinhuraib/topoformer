from omegaconf import OmegaConf
import hydra
import logging
import sys
import numpy as np
from utils import WandbLogger, Logger, get_global_output_folder, set_seed
from topoformer import TransformerClassifier
from hydra.utils import get_original_cwd
from torch import nn, optim
from layers import TransformerLayers
from entities import TextModelConfig
from visualizations import AttentionVisualizer
from trainer import Trainer
from prepare_data import IMDBDataset


@hydra.main(config_path="./conf", config_name="config_text", version_base="1.2")
def main(cfg) -> None:
    Logger.setup_logging(name="my_logger")
    wandb_config = {f"{outer_key}_{inner_key}": inner_value for outer_key, inner_dict in dict(cfg).items() for inner_key, inner_value in inner_dict.items()}
    print(wandb_config)

    wandb_logger = WandbLogger(cfg.params.wandb_name, wandb_config)

    set_seed(seed=cfg.params.seed) 
    train_dataloader, valid_dataloader, test_dataloader, vocab, pad_index = IMDBDataset(cfg.params.dataset_name).prepare_data()
    model = TransformerClassifier(TextModelConfig(**cfg.model, src_vocab_size=len(vocab)), pad_index)
    probe = TransformerLayers(cfg.model.num_hidden_layers).layers

    optimizer = optim.AdamW(model.parameters(), lr=cfg.params.lr)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, train_dataloader, test_dataloader, test_dataloader, cfg.params.epochs, wandb_logger=wandb_logger, task="text_classification")()
 
    trainer.train()

    for i in probe:
        visualizer = AttentionVisualizer(
            model=model,
            layer_to_probe=i,
            test_dataloader=test_dataloader,
            modality="text",
            map_side=int(np.sqrt(cfg.model.hidden_size))
        )
        visualizer.visualize_attention()


if __name__ == "__main__":
    main()