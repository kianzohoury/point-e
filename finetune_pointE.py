# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: November 10, 2023
#
# This code is licensed under the MIT License.
# ==============================================================================

import torch
import torch.optim as optim

from tqdm.auto import tqdm
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

import os
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import argparse

import glob
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sys
import pickle
import pandas as pd
import time
import numpy as np
from datetime import datetime
from pathlib import Path


# imports from machina-labs project
from ..dataset import shapenetcore, transform
module_path = Path(__file__).parent.resolve()
sys.path.append(str(Path(module_path, "example_material/")))

def setup_ddp(gpu, args):
    dist.init_process_group(                                   
        backend='nccl',      # backend='gloo',#                                    
        init_method='env://',     
        world_size=args.world_size,                              
        rank=gpu)

    torch.manual_seed(0)
    torch.cuda.set_device(gpu)


# EDIT: accept ShapeNetCore dataset
class pointE_train_dataset(Dataset):
    def __init__(self, pts_path, split="train", defect_type: str = "removal"):
        # self.captions = pd.read_csv(f'{module_path}/example_material/Cap3D_automated_Objaverse.csv', header=None)
        # self.valid_uid = list(pickle.load(open(f'{module_path}/example_material/training_set.pkl','rb')))
        # self.final_uid = self.valid_uid
        # self.n2idx = {}
        # for i in range(len(self.captions)):
        #     self.n2idx[self.captions[0][i]] = i
        self.pts_path = pts_path

        # this is the index of the 1024 points in the point cloud, randomly generated via torch.randperm(16384)[:1024]
        # it is fixed during training
        # you can also generate your own index via Farthest Point Sampling (FPS) algorithm
        # self.pointcloud_1024index = torch.load('./example_material/pointE_pointcloud_1024index.pt')
        self.dataset = shapenetcore.ShapeNetCore(
            root=f"{module_path.parent}/Shapenetcore_benchmark",
            split=split,
            max_points=1365 if defect_type == "removal" else 1024,
            input_transform=transform.RandomTransform(
                removal_amount=0.25,
                noise_amount=0.05,
                noise_type="gaussian",
                prob_both=0,
                task="completion" if defect_type == "removal" else "denoising"
            ),
            use_rotations=False
        )

    def __len__(self):
        # return len(self.final_uid)
        return len(self.dataset)

    def __getitem__(self, idx):
        _, class_label, defect_type, x, y = self.dataset[idx]
        caption = f"{class_label.lower()} {self.dataset.id_to_defect_type[defect_type]} defect"
        data_tensor = x.permute(1, 0)
        # create dummy color channels since the models expect (x, y, z, r, g, b) format
        color_channels = torch.zeros((3, data_tensor.shape[1]))
        data_tensor = torch.cat([data_tensor, color_channels], dim=0)
        return {'caption': caption, 'pts': data_tensor}

class pointE_val_dataset(pointE_train_dataset):
    def __init__(self, pts_path, split="val", **kwargs):
        super(pointE_val_dataset, self).__init__(pts_path, split=split, **kwargs)


def train(rank, args):
    if args.gpus > 1:
        setup_ddp(rank, args)

    niter = args.epoch
    batch_size = args.batch_size
    learning_rate = args.lr
    save_name = args.save_name

    torch.manual_seed(rank+int(learning_rate*1e6)+int(datetime.now().timestamp()))

    resume_flag = True if args.resume_name != 'none' else False
    if resume_flag:
        model_list = glob.glob('./model_ckpts/%s*.pth'%save_name)
        idx_rank = []
        for l in model_list:
            idx_rank.append(int(l.split('/')[-1].split('_')[-2][5:]) * 21000 + int(l.split('/')[-1].split('_')[-1].split('.')[0]))
        newest = np.argmax(np.array(idx_rank))
        args.resume_name = model_list[newest].split('/')[-1].split('.')[0]

    start_epoch = 0 if not resume_flag else int(args.resume_name.split('_')[-2][5:])
    start_iter = 0 if not resume_flag else int(args.resume_name.split('_')[-1].split('.')[0])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if resume_flag:
        print('reload from ./model_ckpts/%s.pth'%args.resume_name)
        checkpoint = torch.load('./model_ckpts/%s.pth'%args.resume_name, map_location=device)

    base_name = 'base40M-textvec'
    model = model_from_config(MODEL_CONFIGS[base_name], device)
    if not resume_flag:
        model.load_state_dict(load_checkpoint(base_name, device))
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    model.train()
    if args.gpus > 1:
        model = DistributedDataParallel(
                model, device_ids=[rank], find_unused_parameters=False
        )
    
    diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
    my_dataset_train = pointE_train_dataset(pts_path=args.dataset_path, defect_type=args.defect_type)
    data_loader = DataLoader(my_dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    my_dataset_val = pointE_val_dataset(pts_path=args.dataset_path, defect_type=args.defect_type)
    data_loader_val = DataLoader(my_dataset_val, batch_size=batch_size, num_workers=8, prefetch_factor=4, drop_last=True)
    optimizer= optim.AdamW(model.parameters(), lr=learning_rate)
    total_iter_per_epoch = int(len(my_dataset_train)/batch_size)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, niter*total_iter_per_epoch)
    if resume_flag:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
    # tensorboard writer
    log_dir = str(Path(module_path.parent, "runs"))
    tb_writer = SummaryWriter(log_dir=log_dir)
    print(f"Saving tensorboard logging files to: {log_dir}")

    best_val_loss = float("inf")
    for epoch in range(start_epoch, niter):
        s = time.time()
        for i, data in enumerate(data_loader):
            if i + start_iter == total_iter_per_epoch:
                start_iter = 0
                break
            s2 = time.time()
            prompt = data['caption']
            model_kwargs=dict(texts=prompt)
            t = torch.randint(0, DIFFUSION_CONFIGS[base_name]['timesteps'], size=(batch_size,), device=device) 
            x_start = data['pts'].cuda()

            optimizer.zero_grad()
            loss = diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs)
            final_loss = torch.mean(loss['loss'])

            skip_step = torch.isnan(final_loss.detach()) or not torch.isfinite(final_loss.detach())
            skip_step_tensor = torch.tensor(skip_step, dtype=torch.int).to(device)
            if args.gpus > 1:
                dist.all_reduce(skip_step_tensor, op=dist.ReduceOp.SUM)
            skip_step = skip_step_tensor.item() > 0
            if skip_step:
                del final_loss
                torch.cuda.empty_cache()
            else:
                final_loss.backward()
                optimizer.step()
                lr_scheduler.step()
                if args.gpus == 1 or (args.gpus >1 and dist.get_rank() == 0):
                    print('rank: ',rank,time.time()-s2,' epoch: ', epoch, i, final_loss.item())
                if i%100 == 0:
                    with torch.no_grad():
                        val_loss = []
                        for j, dataval in enumerate(data_loader_val):
                            prompt = data['caption']
                            model_kwargs=dict(texts=prompt)
                            t = torch.randint(0, DIFFUSION_CONFIGS[base_name]['timesteps'], size=(batch_size,), device=device) 
                            x_start = data['pts'].cuda()
                            loss = diffusion.training_losses(model, x_start, t, model_kwargs=model_kwargs)
                            final_loss = torch.mean(loss['loss'])
                            print('validation %d/%d: '%(j, len(data_loader_val)), final_loss.item())
                            val_loss.append(final_loss.item())
                        val_mean_loss = torch.mean(torch.Tensor(val_loss)).item()
                        print('rank: ',rank, i, val_mean_loss)
                        
                        # log to tensorboard
                        tb_writer.add_scalar("loss/val", val_mean_loss, global_step=epoch)
                        
                # EDIT: save best model only
                if val_mean_loss < best_val_loss:
                    best_val_loss = val_mean_loss
                    print("Saving best model...")
                    checkpoint_dir = Path(f"{module_path.parent}/checkpoints")
                    if not checkpoint_dir.exists():
                        checkpoint_dir.mkdir()
                        
                    # add fix for non distributed training
                    if hasattr(model, "module"):
                        model_state_dict = model.module.state_dict()
                    else:
                        model_state_dict = model.state_dict()
                    torch.save({'model_state_dict': model_state_dict, 
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': lr_scheduler.state_dict(),
                                }, f"{str(checkpoint_dir)}/{save_name}.pth")

# EDIT: change default training args
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    model_group = parser.add_argument_group('Model settings')
    model_group.add_argument('--port', type = str, default = '12356', help = 'port for parallel')
    model_group.add_argument('--gpus', type = int, default = 1, help = 'how many gpu use')
    model_group.add_argument('--resume_name', type = str, default = 'none', help = 'any name different from "none" will resume the training')
    model_group.add_argument('--save_name', type = str, default = 'defect_diffusion', help = 'name for the save file')
    model_group.add_argument('--lr', type = float, default = 1e-5, help = 'learning rate')
    model_group.add_argument('--batch_size', type = int, default = 64, help = 'batch size')
    model_group.add_argument('--epoch', type = int, default = 1, help = 'total epoch')
    # modified
    model_group.add_argument('--dataset_path', type = str, default = '../Shapenetcore_benchmark', help = 'the directory where the point clouds are stored')
    model_group.add_argument('--defect_type', type = str, default = "removal", help = 'defect type to model')

    args = parser.parse_args()
    if args.gpus == 1:
        train(args.gpus, args)
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = args.port
        args.world_size = args.gpus
        mp.spawn(train, nprocs=args.gpus, args=(args,))


