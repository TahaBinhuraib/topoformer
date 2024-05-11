import wandb
import os
import logging
import hydra
import torch
import json
import random
import string
import matplotlib.pyplot as plt
import numpy as np

class WandbLogger:
    def __init__(self, project_name, config=None, entity=None):
        """Initializes the Wandb logger."""
        self.project_name = project_name
        self.config = config
        self.entity = entity
        logging.info("this is a test logger")
        wandb.init(project=self.project_name, config=self.config, entity=self.entity)

    def log_metrics(self, metrics_dict, commit=True):
        """Simple dict logger."""
        wandb.log(metrics_dict, commit=commit)

    def log_image(self, image_name, plt_figure, commit=True):
        """Log an image to wandb."""
        wandb.log({image_name: wandb.Image(plt_figure)}, commit=commit)

    def finish(self):
        """Finishes Wandb run"""
        wandb.finish()


class Logger:
    @staticmethod
    def setup_logging(log_level=logging.DEBUG, name=__name__):
        """Setup Logger"""
        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        # remove existing handler so we don't log twice...
        while logger.handlers:
            logger.handlers.pop()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s >> %(message)s")

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)


def save_checkpoint(experiment_name, model, epoch, base_dir="experiments"):
    # For now epoch will be final!
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)
    cpfile = os.path.join(outdir, f"model_{epoch}.pt")
    torch.save(model.state_dict(), cpfile)


def save_experiment(
    experiment_name,
    config,
    model,
    train_losses,
    test_losses,
    accuracies,
    base_dir="experiments",
):
    outdir = os.path.join(base_dir, experiment_name)
    with open(jsonfile, "w") as f:
        data = {
            "train_losses": train_losses,
            "test_losses": test_losses,
            "accuracies": accuracies,
        }
        json.dump(data, f, sort_keys=True, indent=4)

    save_checkpoint(experiment_name, model, "final", base_dir=base_dir)


def get_global_output_folder():
    return hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

def get_random_name():
    """
    generate a random 10-digit string of uppercase and lowercase letters, and digits
    """
    return ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=10))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
