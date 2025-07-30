#!/usr/bin/env python3
import sys
import os
from time import sleep

import torchvision

from openood.networks.nca_classification_head import NCA_WITH_HEAD

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)

from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network


from tqdm import tqdm
import numpy as np
import torch
import os
import os.path as osp


import torch  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import v2  # type: ignore[import-untyped]

import numpy as np


from tqdm import tqdm


# load config files for cifar10 baseline
config_files = [
    './configs/datasets/cifar10/cifar10.yml',
    './configs/datasets/cifar10/cifar10_ood.yml',
    './configs/networks/react_nca_head.yml',
    './configs/pipelines/test/test_ood.yml',
    './configs/preprocessors/base_preprocessor.yml',
    './configs/postprocessors/msp.yml',
]
config = config.Config(*config_files)

# modify config
config.num_workers = 1
config.save_output = False
config.parse_refs()

# get dataloader
id_loader_dict = get_dataloader(config)
#ood_loader_dict = get_ood_dataloader(config)
# init network
#nca = get_network(config.network).cuda()
# init ood evaluator
#evaluator = get_evaluator(config)


def eval_selfclass_pathmnist(
    hidden_channels: int,
    gpu,
    batch_size: int,
    gpu_index,
):
    device = torch.device("cuda:%d" % gpu_index if gpu_index >= 0 else "cpu")

    num_classes = 10
    nca = NCA_WITH_HEAD(
        device,
        num_image_channels=3,
        num_hidden_channels=hidden_channels,
        num_output_channels=num_classes,
        num_classes=num_classes,
        num_learned_filters=0,
        use_alive_mask=False,
        fire_rate=0.8,
        steps=20,
        filter_padding="circular",
        pad_noise=True,
    )
    nca.load_state_dict(
        torch.load("results/" + "Model with HeadTrue_c.hidden_50_steps_20class_cifar10.best.pth",
                   weights_only=True,
                   map_location=device)
    )
    nca.to(device)
    print(nca)
    nca.eval()


    pred = []
    gt = []

    print(id_loader_dict['test'])
    dataiter = iter(id_loader_dict['test'])

    for sample in tqdm(dataiter):
        z = sample
        x = z['data']
        y = z['label']
        x = x.float().to(device)
        steps = 20
        y_prob = nca.forward(x, steps)

        y = y.squeeze()
        pred.extend(torch.argmax(y_prob, dim=1).detach().cpu().numpy().tolist())
        gt.extend(y.cpu().numpy().tolist())
    pred = np.array(pred)
    gt = np.array(gt)
    from openood.evaluators.metrics import compute_all_metrics, acc
    accc = acc(pred, gt)

    #accuracy = (pred == gt).sum().item() / len(gt)
    print(f"Accuracy: {accc}")



def main(hidden_channels, gpu: bool, gpu_index: int, batch_size: int):
    eval_selfclass_pathmnist(
        hidden_channels=hidden_channels,
        gpu=gpu,
        batch_size=batch_size,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main(50, True, 0, 64)
