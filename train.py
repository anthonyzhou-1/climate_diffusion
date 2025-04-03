# Default imports
from datetime import datetime
import argparse
import os 
import torch 

# Custom imports
from common.utils import save_yaml, get_yaml, dict2namespace
from lightning.pytorch.callbacks import LearningRateMonitor
from dataset.datamodule import ClimateDataModule
from modules.train_module import TrainModule

# Lightning imports
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

def main(config):
    torch.set_float32_matmul_precision('high') # to use tensor cores if available
    seed = config.seed
    now = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    seed_everything(seed)

    name = f"ClimaDiT_{config.name}_{config.strategy}_{now}"
    wandb_logger = WandbLogger(project=config.project_name,
                               name=name)
    config.log_dir = config.log_dir + '/' + name 

    os.makedirs(config.log_dir, exist_ok=True) 
    save_yaml(config, config.log_dir + "/config.yml")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_t2m_72",
        filename= "model_{epoch:02d}-{val_t2m_72:.2f}",
        dirpath=config.log_dir,
        save_last=True,
        save_top_k=1
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    datamodule = ClimateDataModule(config=config)
    model = TrainModule(config=config,
                        normalizer=datamodule.normalizer)

    trainer = L.Trainer(devices = config.devices,
                        accelerator = config.accelerator,
                        strategy = config.strategy,
                        check_val_every_n_epoch = config.training.check_val_every_n_epoch,
                        max_epochs = config.training.max_epochs,
                        default_root_dir = config.log_dir,
                        callbacks=[checkpoint_callback, lr_monitor],
                        logger=wandb_logger,
                        gradient_clip_val=config.training.gradient_clip_val,
                        accumulate_grad_batches=config.training.gradient_accumulation_steps)
    
    if config.training.checkpoint is not None:
        trainer.fit(model=model,
                datamodule=datamodule,
                ckpt_path=config.training.checkpoint)
    else:
        trainer.fit(model=model, 
                datamodule=datamodule)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument("--config", default=None)
    parser.add_argument('--devices', nargs='+', help='<Required> Set flag', default=[])
    args = parser.parse_args()
    config = get_yaml(args.config)
    config = dict2namespace(config)
    if len(args.devices) > 0:
        config.devices = [int(device) for device in args.devices]
    main(config)