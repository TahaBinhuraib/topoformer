from omegaconf import OmegaConf
import hydra
import logging
import sys
from utils import WandbLogger, Logger, get_global_output_folder
from topoformer import ViTForClassification
from hydra.utils import get_original_cwd
from torch import nn, optim
from layers import ViTLayers
from entities import ViTModelConfig
from visualizations import AttentionVisualizer
from trainer import Trainer
from prepare_data import CIFAR10Dataset
import numpy as np


@hydra.main(config_path="./conf", config_name="config", version_base="1.2")
def main(cfg) -> None:
    Logger.setup_logging(name="my_logger")
    
    wandb_config = {f"{outer_key}_{inner_key}": inner_value for outer_key, inner_dict in dict(cfg).items() for inner_key, inner_value in inner_dict.items()}
    print(wandb_config)

    wandb_logger = WandbLogger(cfg.params.wandb_name, wandb_config)
    
    model = ViTForClassification(ViTModelConfig(**cfg.model))
    probe = ViTLayers(cfg.model.num_hidden_layers).layers()

    train_dataloader, test_dataloader, _ = CIFAR10Dataset(cfg.params.batch_size, root_data_dir="./data").prepare_data()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.params.lr)
    loss_fn = nn.CrossEntropyLoss()
    trainer = Trainer(model, optimizer, loss_fn, train_dataloader, test_dataloader, test_dataloader, cfg.params.epochs, wandb_logger=wandb_logger)()
 
    trainer.train()

    for i in probe:
        visualizer = AttentionVisualizer(
            model=model,
            layer_to_probe=i,
            test_dataloader=test_dataloader,
            map_side=int(np.sqrt(cfg.model.hidden_size))
        )
        visualizer.visualize_attention()


if __name__ == "__main__":
    main()