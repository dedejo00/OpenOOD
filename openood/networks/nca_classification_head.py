from __future__ import annotations
import logging
from typing import Callable, Optional, Dict, Tuple, Any
from os import PathLike

import numpy as np

import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]



class ClassificationNCAHead(nn.Module):
    """
    Abstract base class for NCA models.
    """

    def __init__(
        self,
        device: torch.device,
        num_image_channels: int,
        num_hidden_channels: int,
        num_output_channels: int,
        fire_rate: float = 0.8,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        immutable_image_channels: bool = True,
        num_learned_filters: int = 0,
        dx_noise: float = 0.0,
        filter_padding: str = "circular",
        use_laplace: bool = False,
        kernel_size: int = 3,
        pad_noise: bool = True,
        save_cell_state: bool = False,
        use_temporal_encoding:bool = True,
        steps: int = 20
    ):
        """
        Constructor.

        :param device [device]: Pytorch device descriptor.
        :param num_image_channels [int]: Number of channels reserved for input image.
        :param num_hidden_channels [int]: Number of hidden channels (communication channels).
        :param num_output_channels [int]: Number of output channels.
        :param fire_rate [float]: Fire rate for stochastic weight update. Defaults to 0.8.
        :param hidden_size [int]: Number of neurons in hidden layer. Defaults to 128.
        :param use_alive_mask [bool]: Whether to use alive masking during training. Defaults to False.
        :param immutable_image_channels [bool]: If image channels should be fixed during inference, which is the case for most segmentation or classification problems. Defaults to True.
        :param num_learned_filters [int]: Number of learned filters. If zero, use two sobel filters instead. Defaults to 2.
        :param dx_noise [float]:
        :param filter_padding [str]: Padding type to use. Might affect reliance on spatial cues. Defaults to "circular".
        :param use_laplace [bool]: Whether to use Laplace filter (only if num_learned_filters == 0)
        :param kernel_size [int]: Filter kernel size (only for learned filters)
        :param pad_noise [bool]: Whether to pad input image tensor with noise in hidden / output channels
        :param autostepper [Optional[AutoStepper]]: AutoStepper object to select number of time steps based on activity
        """
        super(ClassificationNCAHead, self).__init__()

        self.device = device
        self.to(device)

        self.num_image_channels = num_image_channels
        self.num_hidden_channels = num_hidden_channels
        self.num_output_channels = num_output_channels
        self.num_channels = (
            num_image_channels + num_hidden_channels + num_output_channels
        )
        self.fire_rate = fire_rate
        self.hidden_size = hidden_size
        self.use_alive_mask = use_alive_mask
        self.immutable_image_channels = immutable_image_channels
        self.num_learned_filters = num_learned_filters
        self.use_laplace = use_laplace
        self.dx_noise = dx_noise
        self.pad_noise = pad_noise

        self.plot_function: Optional[Callable] = None
        self.validation_metric: Optional[str] = None
        self.filters: list | nn.ModuleList = []
        self.use_temporal_encoding = use_temporal_encoding

        self.save_cell_state = save_cell_state
        self.latest_cell_state = None
        self.steps = steps

        if num_learned_filters > 0:
            self.num_filters = num_learned_filters
            filters = []
            for _ in range(num_learned_filters):
                filters.append(
                    nn.Conv2d(
                        self.num_channels,
                        self.num_channels,
                        kernel_size=kernel_size,
                        stride=1,
                        padding=(kernel_size // 2),
                        padding_mode=filter_padding,
                        groups=self.num_channels,
                        bias=False,
                    ).to(self.device)
                )
            self.filters = nn.ModuleList(filters)
        else:
            sobel_x = np.outer([1, 2, 1], [-1, 0, 1]) / 8.0
            sobel_y = sobel_x.T
            laplace = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
            self.filters.append(sobel_x)
            self.filters.append(sobel_y)
            if self.use_laplace:
                self.filters.append(laplace)
            self.num_filters = len(self.filters)

        input_vector_size = self.num_channels * (self.num_filters + 1)
        if self.use_temporal_encoding:
            input_vector_size += 1
        self.network = nn.Sequential(
            nn.Linear(
                input_vector_size,
                self.hidden_size * 2,
                bias=True,
            ),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.num_channels, bias=False),
        ).to(device)

        # initialize final layer with 0
        with torch.no_grad():
            self.network[-1].weight.data.fill_(0)

        pool_size = 8
        self.classifierHead = nn.Sequential(
            nn.Linear(
                int(self.num_output_channels) * pool_size * pool_size, 256, bias=True
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128, bias=True),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(),
            nn.Linear(128, self.num_output_channels, bias=False),
        ).to(device)

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input. Intended to be overwritten by subclass, if preprocessing
        is necessary.

        :param x [torch.Tensor]: Input tensor to preprocess.

        :returns: Processed tensor.
        """
        return x

    def __alive(self, x):
        mask = (
            F.max_pool2d(
                x[:, 3, :, :],
                kernel_size=3,
                stride=1,
                padding=1,
            )
            > 0.1
        )
        return mask

    def _perceive(self, x, step):
        def _perceive_with(x, weight):
            if isinstance(weight, nn.Conv2d):
                return weight(x)
            # if using a hard coded filter matrix.
            # this is done in the original Growing NCA paper, but learned filters typically
            # work better.
            conv_weights = torch.from_numpy(weight.astype(np.float32)).to(self.device)
            conv_weights = conv_weights.view(1, 1, 3, 3).repeat(
                self.num_channels, 1, 1, 1
            )
            return F.conv2d(x, conv_weights, padding=1, groups=self.num_channels)

        perception = [x]
        perception.extend([_perceive_with(x, w) for w in self.filters])
        if self.use_temporal_encoding:
            perception.append(
                torch.mul(
                    torch.ones((x.shape[0], 1, x.shape[2], x.shape[3])), step / 100
                ).to(self.device)
            )
        y = torch.cat(perception, 1)
        return y

    def _update(self, x: torch.Tensor, step: int):
        """
        :param x [torch.Tensor]: Input tensor, BCWH
        """
        assert x.shape[1] == self.num_channels

        # Perception
        dx = self._perceive(x, step)

        # Compute delta from FFNN network
        dx = dx.permute(0, 2, 3, 1)  # B C W H --> B W H C
        dx = self.network(dx)

        # Stochastic weight update
        fire_rate = self.fire_rate
        stochastic = torch.rand([dx.size(0), dx.size(1), dx.size(2), 1]) < fire_rate
        stochastic = stochastic.float().to(self.device)
        dx = dx * stochastic

        dx += self.dx_noise * torch.randn([dx.size(0), dx.size(1), dx.size(2), 1]).to(
            self.device
        )

        if self.immutable_image_channels:
            dx[..., : self.num_image_channels] *= 0

        dx = dx.permute(0, 3, 1, 2)  # B W H C --> B C W H
        x = x + dx

        # Alive masking
        if self.use_alive_mask:
            life_mask = self.__alive(x)
            life_mask = life_mask
            x = x.permute(1, 0, 2, 3)  # B C W H --> C B W H
            x = x * life_mask.float()
            x = x.permute(1, 0, 2, 3)  # C B W H --> B C W H
        return x


    def predict(self, image: torch.Tensor, steps: int = 100) -> tuple[Any, Any]:
        """
        :param image [torch.Tensor]: Input image, BCWH.

        :returns [torch.Tensor]: Output image, BWHC
        """
        assert steps >= 1
        assert image.shape[1] <= self.num_channels
        self.eval()
        with torch.no_grad():
            x = image.clone()
            x = pad_input(x, self, noise=self.pad_noise)
            x = self.prepare_input(x)
            x = self.forward_head(x, steps=steps)  # type: ignore[assignment]
            return x

    def forward_head(
        self,
        x: torch.Tensor,
        steps: int = 1,
        return_steps: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, int]:
        """
        :param x [torch.Tensor]: Input image, padded along the channel dimension, BCWH.
        :param steps [int]: Time steps in forward pass.
        :param return_steps [bool]: Whether to return number of steps we took.

        :returns: Output image, BWHC
        """
        for step in range(steps):
            x = self._update(x, step)
        x = x.permute(0, 2, 3, 1)
        class_channels = x[
            ..., self.num_image_channels + self.num_hidden_channels :
        ]
        class_channels = class_channels.permute(0, 3, 1, 2)
        max = F.adaptive_avg_pool2d(class_channels, (8, 8))
        max = max.view(max.size(0), -1)
        x = self.classifierHead(max)
        x = F.log_softmax(x, dim=1)
        if return_steps:
            return x, steps
        return x


    def classify(self, image: torch.Tensor, steps: int = 100) -> torch.Tensor:
        """
        :param image [torch.Tensor]: Input image, BCWH.

        :returns [torch.Tensor]: Output image, BWHC
        """
        x = image.clone()
        x = pad_input(x, self, noise=self.pad_noise)
        x = self.prepare_input(x)
        return self.forward_head(x, steps)


    def forward(self, x, return_feature=False, return_feature_list=False):
        x = pad_input(x, self, noise=self.pad_noise)
        x = self.prepare_input(x)
        for step in range(self.steps):
            x = self._update(x, step)
        x = x.permute(0, 2, 3, 1)
        class_channels = x[
            ..., self.num_image_channels + self.num_hidden_channels :
        ]
        class_channels = class_channels.permute(0, 3, 1, 2)
        max = F.adaptive_avg_pool2d(class_channels, (8, 8))
        max = max.view(max.size(0), -1)
        feat = self.classifierHead[0:4](max)
        x = self.classifierHead[4:7](feat)
        #x = self.classifierHead(max)
        x = F.log_softmax(x, dim=1)
        if return_feature:
            return x, feat
        return x

    def get_fc_layer(self):
        return self.classifierHead[4:7]


def pad_input(x: torch.Tensor, nca: "BasicNCAModel", noise: bool = True) -> torch.Tensor:
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
