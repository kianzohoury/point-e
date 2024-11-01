# ==============================================================================
# Copyright (c) 2023 Tiange Luo, tiange.cs@gmail.com
# Last modified: September 20, 2023
#
# This code is licensed under the MIT License.
# ==============================================================================
import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint, fetch_file_cached
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

import os
import sys
from PIL import Image
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path

from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.pc_to_mesh import marching_cubes_mesh
from point_e.util.plotting import plot_point_cloud
from point_e.util.point_cloud import PointCloud
from point_e.util.mesh import TriMesh
import numpy as np
import pickle

import argparse
import random
import time
import trimesh

# imports from machina-labs project
from ..dataset import shapenetcore, transform
module_path = Path(__file__).parent.resolve()
sys.path.append(str(Path(module_path, "checkpoints/")))
Path(module_path.parent, "synthetic_data").mkdir(exist_ok=True)

# EDIT: change defaults
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', help="path to finetuned model")
parser.add_argument('--save_dir', default=Path(module_path.parent, "synthetic_data"), type=str, help="result files save to here")
parser.add_argument('--num_generate', default=1, type=int, help="number of point clouds to generate")
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('creating base model...')
base_name = 'base40M-textvec'
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('loading finetuned checkpoint: ', opt.checkpoint)
base_model.load_state_dict(torch.load(opt.checkpoint, map_location=device)['model_state_dict'])

### results (.ply) saved here
num_generate = opt.num_generate
# save_dir = os.path.join('./pointE_inference', opt.save_name)
# os.makedirs(save_dir, exist_ok=True)

# EDIT: skip upsampling since I can't download the checkpoint for some reason...
# print('downloading upsampler checkpoint...')
# upsampler_model.load_state_dict(load_checkpoint('upsample', device))

sampler = PointCloudSampler(
    device=device,
    models=[base_model],
    diffusions=[base_diffusion],
    num_points=[1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0],
    model_kwargs_key_filter=('texts', ), # Do not condition the upsampler at all
    use_karras=[True],
    karras_steps=[64],
    sigma_min=[1e-3],
    sigma_max=[120],
    s_churn=[3]
)

print('creating SDF model...')
name = 'sdf'
model = model_from_config(MODEL_CONFIGS[name], device)
model.eval()
print('loading SDF model...')
model.load_state_dict(load_checkpoint(name, device))

batch_size = 1

import pickle
import pandas as pd

# EDIT: test on ShapeNetCore dataset
dataset = shapenetcore.ShapeNetCore(
    root=f"{module_path.parent}/Shapenetcore_benchmark",
    split="train",
    max_points=1024,
    downsampling_mode="uniform",
    input_transform=transform.RandomTransform(
        removal_amount=0.2,
        noise_amount=0.02,
        noise_type="uniform",
        prob_both=0,
        task="completion" if "removal" in opt.checkpoint else "denoising"
    ),
    use_rotations=False
)

test_uids = list(range(len(dataset)))
### add the below random command to parallel test
random.shuffle(test_uids)
test_uids = test_uids[:num_generate]

print('start mesh generation, generated mesh saved as .ply')
for i in range(len(test_uids)):
    s = time.time()
    ### skip if output mesh exists
    if os.path.exists(os.path.join(opt.save_dir,'%s.ply'%(test_uids[i]))):
       continue
   # only need a test prompt since we're generating point clouds
    _, class_label, defect_type, x, y = dataset[i]
    prompt = f"{class_label.lower()} {dataset.id_to_defect_type[defect_type]} defect"
    samples = None
    for x in tqdm(sampler.sample_batch_progressive(batch_size=batch_size, model_kwargs=dict(texts=[prompt,]*batch_size))):
        samples = x

    pc = sampler.output_to_point_clouds(samples)[0]

    # EDIT: just save the point cloud, no mesh
    # mesh = marching_cubes_mesh(
    #     pc=pc,
    #     model=model,
    #     batch_size=4096,
    #     grid_size=128, 
    #     progress=True,
    # )

    with open(os.path.join(opt.save_dir, f'{prompt.lower().replace(" ", "_")}_{test_uids[i]}.ply'), 'wb') as f:
        pc.write_ply(f)
    # print('mesh generation progress: %d/%d'%(i,len(test_uids)), 'time cost:', time.time()-s)
