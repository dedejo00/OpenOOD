from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator

from openood.networks import get_network
from tqdm import tqdm
import numpy as np
import torch
import os
import os.path as osp

def save_arr_to_dir(arr, dir):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    # Save the array to the file
    with open(dir, 'wb') as f:
        np.save(f, arr)

def calc_nqm_like(class_channels):
    class_channels = torch.tensor(np.array(class_channels))
    print(class_channels.shape)
    print(class_channels.max())
    print(class_channels.min())
    mean = torch.mean(class_channels, dim=(0))
    std = torch.std(class_channels, dim=(0), correction=0)
    std = std.sum(dim=(1,2))
    mean = mean.sum(dim=(1,2))
    nqm = std / mean
    print(nqm.shape)
    print(nqm.max())
    print(nqm.min())
    #exit(0)
    return nqm

save_root = f'./results/nqm'

# load config files for cifar10 baseline
config_files = [
    './configs/datasets/cifar10/cifar10.yml',
    './configs/datasets/cifar10/cifar10_ood.yml',
    './configs/networks/nca.yml',
    './configs/pipelines/test/test_ood.yml',
    './configs/preprocessors/base_preprocessor.yml',
    './configs/postprocessors/msp.yml',
]
config = config.Config(*config_files)
config.num_workers = 1
config.save_output = False
config.parse_refs()
id_loader_dict = get_dataloader(config)
ood_loader_dict = get_ood_dataloader(config)
net = get_network(config.network).to(config.device)
evaluator = get_evaluator(config)

net.eval()
modes = ['test']
for mode in modes:
    dl = id_loader_dict[mode]
    dataiter = iter(dl)

    logits_list = []

    label_list = []

    for i in tqdm(range(1,
                        len(dataiter) + 1),
                  desc='Extracting reults...',
                  position=0,
                  leave=True):
        batch = next(dataiter)
        data = batch['data'].to(config.device)
        label = batch['label']
        class_channels_list = []
        for j in range(100):
            with torch.no_grad():
                logits_cls, class_channel = net.classify(data, steps=20, return_class_channel=True)
                class_channels_list.append(class_channel.detach().cpu())
        nqm = calc_nqm_like(class_channels_list)
        pred = torch.argmax(logits_cls, dim=-1).detach().cpu()
        print("ID #################")
        print(nqm.mean())
        break
    break

ood_splits = ['farood']
for ood_split in ood_splits:
    for dataset_name, ood_dl in ood_loader_dict[ood_split].items():
        dataiter = iter(ood_dl)
        for i in tqdm(range(1,
                            len(dataiter) + 1),
                      desc='Extracting reults...',
                      position=0,
                      leave=True):
            batch = next(dataiter)
            data = batch['data'].to(config.device)
            label = batch['label']
            class_channels_list = []
            pred_list = []
            for j in range(100):
                with torch.no_grad():
                    logits_cls, class_channel = net.classify(data, steps=20, return_class_channel=True)
                    class_channels_list.append(class_channel.detach().cpu())
                    pred = torch.argmax(logits_cls, dim=-1).detach().cpu()
                    pred_list.append(pred)
            nqm = calc_nqm_like(class_channels_list)
            print("OOD ############")
            print(nqm.mean())
            exit(0)