#!/usr/bin/env python3
import sys
import os
from time import sleep

import torchvision

from openood.networks.nca_classification_head import NCA_WITH_HEAD

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(root_dir)




import torch  # type: ignore[import-untyped]
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import v2  # type: ignore[import-untyped]

import numpy as np


from tqdm import tqdm


T = transforms.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float, scale=True),
        v2.ConvertImageDtype(dtype=torch.float32),
        v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomVerticalFlip(),
    ]
)


def eval_selfclass_pathmnist(
    hidden_channels: int,
    gpu,
    batch_size: int,
    gpu_index,
):
    device = torch.device("cuda:%d" % gpu_index if gpu_index >= 0 else "cpu")


    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=T)
    loader_test = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

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
        filter_padding="circular",
        pad_noise=False,
    )
    nca.load_state_dict(
        torch.load("results/" + "Model with HeadTrue_c.hidden_30_gc_False_noise_True_AM_False_steps_60class_cifar10.best.pth",
                   weights_only=True,
                   map_location=device)
    )

    model_parameters = filter(lambda p: p.requires_grad, nca.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Trainable parameters: {params}")
    print(f"That is {4 * params / 1000} kB")

    nca = nca.to(device)
    nca.eval()
    pred = []
    gt = []

    for sample in tqdm(iter(loader_test)):
        x, y = sample
        x = pad_input(x, nca, noise=True)
        x = x.float().to(device)
        #x = x.float().permute(0, 2, 3, 1).to(device)
        steps = 60
        y_prob = nca.classify(x, steps)

        y = y.squeeze()
        pred.extend(torch.argmax(y_prob, dim=1).detach().cpu().numpy().tolist())
        gt.extend(y.cpu().numpy().tolist())
    print(pred, gt)
    #pred = torch.tensor(pred, device=device)
    #gt = torch.tensor(gt, device=device)
    pred = np.array(pred)
    gt = np.array(gt)
    from openood.evaluators.metrics import compute_all_metrics, acc
    accc = acc(pred, gt)

    #accuracy = (pred == gt).sum().item() / len(gt)
    print(f"Accuracy: {accc}")

import torch.nn.functional as F  # type: ignore[import-untyped]
def pad_input(x: torch.Tensor, nca: "NCA_WITH_HEAD", noise: bool = True) -> torch.Tensor:
    """
    Pads input tensor along channel dimension to match the expected number of
    channels required by the NCA model. Pads with either Gaussian noise or zeros,
    depending on "noise" parameter. Gaussian noise has mean of 0.5 and sigma 0.225.

    :param x [torch.Tensor]: Input image tensor, BCWH.
    :param nca [BasicNCAModel]: NCA model definition.
    :param noise [bool]: Whether to pad with noise. Otherwise zeros. Defaults to True.

    :returns: Input tensor, BCWH, padded along the channel dimension.
    """
    if x.shape[1] < nca.num_channels:
        x = F.pad(
            x, (0, 0, 0, 0, 0, nca.num_channels - x.shape[1], 0, 0), mode="constant"
        )
        if noise:
            x[
                :,
                nca.num_image_channels : nca.num_image_channels
                + nca.num_hidden_channels,
                :,
                :,
            ] = torch.normal(
                0.5,
                0.225,
                size=(x.shape[0], nca.num_hidden_channels, x.shape[2], x.shape[3]),
            )
    return x




def main(hidden_channels, gpu: bool, gpu_index: int, batch_size: int):
    eval_selfclass_pathmnist(
        hidden_channels=hidden_channels,
        gpu=gpu,
        batch_size=batch_size,
        gpu_index=gpu_index,
    )


if __name__ == "__main__":
    main(30, True, 0, 64)
