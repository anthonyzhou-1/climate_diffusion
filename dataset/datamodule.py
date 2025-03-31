import lightning as L
from torch.utils.data import DataLoader
from dataset.plasim import PLASIMData, SURFACE_FEATURES, MULTI_LEVEL_FEATURES, CONSTANTS_FEATURES, YEARLY_FEATURES

class ClimateDataModule(L.LightningDataModule):
    def __init__(self,
                 config) -> None:
        
        super().__init__()
        self.config = config

        self.train_dataset = PLASIMData(data_path=self.config.data.train_data_path,
                                    norm_stats_path=self.config.data.norm_stats_path,
                                    boundary_path=self.config.data.boundary_path,
                                    surface_vars=SURFACE_FEATURES,
                                    multi_level_vars=MULTI_LEVEL_FEATURES,
                                    constant_names=CONSTANTS_FEATURES,
                                    yearly_names=YEARLY_FEATURES,
                                    nsteps=self.config.data.training_nsteps,     
                                    )
        
        self.val_dataset = PLASIMData(data_path=self.config.data.val_data_path,
                                norm_stats_path=self.config.data.norm_stats_path,
                                boundary_path=self.config.data.boundary_path,
                                surface_vars=SURFACE_FEATURES,
                                multi_level_vars=MULTI_LEVEL_FEATURES,
                                constant_names=CONSTANTS_FEATURES,
                                yearly_names=YEARLY_FEATURES,
                                nsteps=self.config.data.val_nsteps,
                                load_into_memory=True)
        
        self.normalizer = self.train_dataset.normalizer


    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass
        
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        # Eager imports to avoid specific dependencies that are not needed in most cases

        if stage == "fit":
            pass 

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            pass

        if stage == "predict":
            pass

    def train_dataloader(self):
        self.pin_memory = False if self.num_workers == 0 else True
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=self.num_workers, 
                          pin_memory=self.pin_memory,
                          drop_last=True,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=False, 
                          num_workers=self.num_workers,
                          drop_last=False)

    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        return None