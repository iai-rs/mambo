import os
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
from torch.optim import Adam, SGD
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop

from models.ddpm import *
from models.unet import Unet
from datasets.vindr_unified import VINDR_Dataset
import config_unified as cfg

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import wandb

def cycle(dl):
    while True:
        for data in dl:
            yield data

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        results_folder: str,
        wandb_run
    ) -> None:
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.train_data = cycle(train_data)
        self.optimizer = optimizer
        self.save_every = cfg.save_every
        self.model = DDP(model, device_ids=[gpu_id])
        self.min_train_loss = torch.inf
        self.total_epochs = cfg.total_epochs
        self.image_size = cfg.image_size
        self.wandb_run = wandb_run

        self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=30, min_lr=1e-7, verbose=True)

        self.results_folder = results_folder

    def _run_batch(self, x, t, c=None):
        self.optimizer.zero_grad()

        loss = p_losses(self.model, x, t, loss_type='l2', cond=c)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, batch_idx, x, y):
        # for batch_idx, (x, y) in pbar:
        #     x = x.to(self.gpu_id)
        #     y = y.to(self.gpu_id)

        #     batch_size = x.shape[0]
        #     t = torch.randint(0, cfg.timesteps, (batch_size,), device=self.gpu_id).long()

        #     loss = self._run_batch(x, t)
        #     train_loss += loss
            # pbar.set_description(
            #     f'GPU: {self.gpu_id}, '
            #     f'epoch: {epoch+1}/{self.total_epochs}: '
            #     f'train loss: {train_loss/(batch_idx+1):.8f}, '
            #     f'lr: {self.optimizer.state_dict()["param_groups"][0]["lr"]:.8f}, '
            #     )
        # if train_loss < self.min_train_loss:
        #     self.min_train_loss = train_loss
        #     self._save_checkpoint(epoch)

        # self.lr_scheduler.step(train_loss)
        # return train_loss
        return None

    def _save_checkpoint(self, epoch):
        filename = f'model_{epoch}.pt'
        checkpoint_path = Path(self.results_folder, 'models') / filename
        if not os.path.exists(checkpoint_path):
            checkpoint = {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optim_state": self.optimizer.state_dict()
            }
            torch.save(checkpoint, checkpoint_path)
            model_artifact = wandb.Artifact(
                    name=f"{cfg.exp_name}",
                    type="model"
                )
            model_artifact.add_file(checkpoint_path)
            self.wandb_run.log_artifact(model_artifact)


    def train(self):
        train_loss = 0
        for i in range(self.total_epochs):
            self.model.train()
            x = next(self.train_data)
#             x = next(self.train_data)
            x = x.to(self.gpu_id)
#             y = y.to(self.gpu_id)
            batch_size = x.shape[0]
            t = torch.randint(0, cfg.total_timesteps, (batch_size,), device=self.gpu_id).long()
            loss = self._run_batch(x, t)
            train_loss += loss
            if (i+1) % cfg.save_every == 0:
                if self.gpu_id == 0:
                    if train_loss < self.min_train_loss:
                        self.min_train_loss = train_loss
                        self._save_checkpoint(i)
                    self.model.eval()
                    with torch.no_grad():
                        #ctx = torch.tensor([0, 0, 1, 1, 2, 2]).int().to(self.gpu_id)
                        noise = torch.randn_like(x[0][0])
                        x[0][0] = noise
                        samples = sample(self.model, image_size=(batch_size, cfg.channels, self.image_size, self.image_size), sampling_timesteps=cfg.total_timesteps)             
                        samples = samples[:, 0:1, :, :] # save only patch
                        save_image(samples, str(Path(self.results_folder, 'images') / f'sample-{i}.png'), nrow = batch_size)
                        images = wandb.Image(samples)
                        #self.wandb_run.log({"samples": images}, commit=False)
                        self.wandb_run.log({"loss": round(train_loss/(cfg.save_every*torch.cuda.device_count()*batch_size), 8), "samples": images})
                self.lr_scheduler.step(train_loss)
                train_loss = 0

            if self.gpu_id == 0:
                print(i)
                if (i+1) % self.save_every == 0:
                    self._save_checkpoint(i)
                # self.model.eval()
                # with torch.no_grad():
                #     #ctx = torch.tensor([0, 0, 1, 1, 2, 2]).int().to(self.gpu_id)
                #     samples = sample(self.model, image_size=(6, 1, self.image_size, self.image_size))
                #     save_image(samples, str(Path(self.results_folder, 'images') / f'sample-{epoch}.png'), nrow = 6)
                #     images = wandb.Image(samples)
                #     #self.wandb_run.log({"samples": images}, commit=False)
                #     self.wandb_run.log({"loss": round(train_loss/len(self.train_data), 8), "samples": images})

def load_train_objs(image_size=256, input_dim=128, dim_mults=(1, 2, 2, 4, 4), channels=1, optimizer_type='adam', lr=2e-4, artifact_dir=''):

#     csv_path = '/data/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/finding_annotations.csv'
#     images_path = '/data/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images/'    
#     precrop_transform = Compose([ToTensor(), Resize((256, 256), antialias=False)])
#     crop_transform = RandomCrop((256, 256))
#     postcrop_transform = Normalize(0.5, 1.0)
#     dataset = VINDR_Dataset(csv_path, images_path, is_test=False, is_healthy=False, precrop_transform=precrop_transform, crop_transform=crop_transform, postcrop_transform=postcrop_transform)

    if cfg.dataset == 'vindr':
        csv_path = r"/data/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/finding_annotations.csv"
        images_path = r"/data/vindr-mammo-a-large-scale-benchmark-dataset-for-computer-aided-detection-and-diagnosis-in-full-field-digital-mammography-1.0.0/images"
        transform = Compose([Normalize(0.007, 0.01), Resize((image_size, image_size), antialias=False)])
        dataset = VINDR_Dataset(csv_path, images_path, transform=transform)
#     elif cfg.dataset == 'rsna':
#         csv_path = 'src/data/rsna_train_with_shape.csv'
#         images_path = '/data/train_images'
#         transform = Compose([Normalize(0.007, 0.01), Resize((image_size, image_size), antialias=False)])
#         dataset = RSNA_Dataset(csv_path, images_path, transform=transform)

        
    model = Unet(dim=input_dim, dim_mults=dim_mults, channels=channels)
    
    
    model_checkpoint = torch.load(f'{artifact_dir}/model_5999.pt')
    model.load_state_dict((dict([(n.replace('module.', ''), p) for n, p in model_checkpoint['model_state'].items()])))

    if optimizer_type == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    else:
        optimizer = SGD(model.parameters(), lr=lr)

    return dataset, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int, num_workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
        sampler=DistributedSampler(dataset)
    )

def main(rank: int, world_size: int, results_folder, wandb_run, artifact_dir):
    ddp_setup(rank, world_size)
    dataset, model, optimizer = load_train_objs(cfg.image_size, cfg.input_dim, cfg.dim_mults, cfg.channels, cfg.optimizer_type, cfg.learning_rate, artifact_dir)
    train_data = prepare_dataloader(dataset, cfg.batch_size, num_workers=64)

#     from time import time
#     import multiprocessing as mp

#     for num_workers in range(2, 80, 2):
#         train_loader = prepare_dataloader(dataset, batch_size, num_workers)
#         start = time()
#         for epoch in range(1, 3):
#             for i, data in enumerate(train_loader, 0):
#                 pass
#         end = time()
#         print("Finish with:{} second, num_workers={}".format(end - start, num_workers))

    trainer = Trainer(model, train_data, optimizer, rank, results_folder, wandb_run)
    trainer.train()

    destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    results_folder = './results'
    dtime = datetime.now().strftime("%y_%m_%d_%H_%M")
    results_folder = Path(results_folder, cfg.exp_name, dtime)
    if not os.path.exists(results_folder):
        results_folder.mkdir(exist_ok = True, parents=True)
        os.mkdir(Path(results_folder, 'models'))
        os.mkdir(Path(results_folder, 'images'))

    wandb.login(key="2606655dd7f8aa1906cb33680b26872da3bbb10b")
    wandb_run = wandb.init(
        # set the wandb project where this run will be logged
        project="mambo",
                # track hyperparameters and run metadata
        config={
            "learning_rate": cfg.learning_rate,
            "epochs": cfg.total_epochs,
            "batch_size": cfg.batch_size,
            "input_dim": cfg.input_dim,
            "image_shape": cfg.image_size,
            "dim_mults": cfg.dim_mults
        }
    )
    artifact = wandb_run.use_artifact('ivi-cvrs/mambo/vindr_unified_4x:v107', type='model')
    artifact_dir = artifact.download()
    
    mp.spawn(main, args=(world_size, results_folder, wandb_run, artifact_dir), nprocs=world_size)
    wandb.finish()
