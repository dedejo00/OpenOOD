from __future__ import annotations
import logging
from typing import Callable, Optional, Dict, Tuple

import numpy as np

import torch  # type: ignore[import-untyped]
import torch.nn as nn  # type: ignore[import-untyped]
import torch.nn.functional as F  # type: ignore[import-untyped]


class AutoStepper:
    """
    Helps selecting number of timesteps based on NCA activity.
    """

    def __init__(
            self,
            min_steps: int = 10,
            max_steps: int = 100,
            plateau: int = 5,
            verbose: bool = False,
            threshold: float = 1e-2,
    ):
        """
        Constructor.

        :param min_steps [int]: Minimum number of timesteps to always execute. Defaults to 10.
        :param max_steps [int]: Terminate after maximum number of steps. Defaults to 100.
        :param plateau [int]: _description_. Defaults to 5.
        :param verbose [bool]: Whether to communicate. Defaults to False.
         threshold (float, optional): _description_. Defaults to 1e-2.
        """
        assert min_steps >= 1
        assert plateau >= 1
        assert max_steps > min_steps
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.plateau = plateau
        self.verbose = verbose
        self.threshold = threshold


class NCA_WITH_HEAD(nn.Module):
    """
    Abstract base class for NCA models.
    """

    def __init__(
            self,
            device: torch.device,
            num_image_channels: int,
            num_hidden_channels: int,
            num_output_channels: int,
            num_classes: int,
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
            autostepper: Optional[AutoStepper] = None,
            pixel_wise_loss: bool = False,
            threshold_activations_react: float = None,
            steps: int = 20,
            use_temporal_encoding: bool = True
    ):
        """
        Constructor.

        :param device [device]: Pytorch device descriptor.
        :param num_image_channels [int]: Number of channels reserved for input image.
        :param num_hidden_channels [int]: Number of hidden channels (communication channels).
        :param num_output_channels [int]: Number of output channels.
        :param fire_rate [float]: Fire rate for stochastic weight update. Defaults to 0.5.
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
        super(NCA_WITH_HEAD, self).__init__()

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
        self.autostepper = autostepper

        self.plot_function: Optional[Callable] = None
        self.filters: list | nn.ModuleList = []

        self.num_classes = num_classes
        self.pixel_wise_loss = pixel_wise_loss
        self.validation_metric = "accuracy_micro"
        self.use_temporal_encoding = use_temporal_encoding

        # ReAct
        self.threshold_activations_react = threshold_activations_react

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

        self.meta: dict = {}

        alpha = torch.tensor([1.0, 1.0, 1.0, 5.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0]).to(device)  # Emphasize classes cat and dog
        gamma = 2.5
        self.focal_loss = FocalLoss(alpha, gamma)

        pool_size = 8
        self.classifierHead = nn.Sequential(
            nn.Linear(
                int(self.num_classes) * pool_size* pool_size, 256, bias=True
            ),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128, bias=True),
            nn.Dropout(p=0.7),
            nn.LeakyReLU(),
            nn.Linear(128, self.num_classes, bias=False),
        ).to(device)


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
            ..., self.num_image_channels + self.num_hidden_channels:
        ]
        class_channels = class_channels.permute(0, 3, 1, 2)
        max = F.adaptive_avg_pool2d(class_channels, (8, 8))
        max = max.view(max.size(0), -1)

        if self.threshold_activations_react:
            x = self.classifierHead(max)
            #feature = self.classifierHead[0:3](max)
            #feature = feature.clip(max=self.threshold_activations_react)
            #x = self.classifierHead[3:6](feature)
        else:
            x = self.classifierHead(max)


        x = F.log_softmax(x, dim=1)
        if return_steps:
            return x, steps
        return x

    def loss(self, image: torch.Tensor, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return the classification loss. For pixel-wise ("self-classifying") problems,
        such as self-classifying MNIST, we compute the Cross-Entropy loss.
        For image-wise classification, MSE loss is returned.

        :param image [torch.Tensor]: Input image, BWHC.
        :param label [torch.Tensor]: Ground truth.

        :returns: Dictionary of identifiers mapped to computed losses.
        """
        loss = F.cross_entropy(image, label)
        return {
            "total": loss,
            "classification": loss,
        }

    def classify(self, image: torch.Tensor, steps: int = 100) -> torch.Tensor:
        """
        :param image [torch.Tensor]: Input image, BCWH.

        :returns [torch.Tensor]: Output image, BWHC
        """
        x = image.clone()
        x = pad_input(x, self, noise=self.pad_noise)
        x = self.prepare_input(x)
        return self.forward_head(x, steps)


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
    def _update(self, x: torch.Tensor, step: int) -> torch.Tensor:
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

    def forward(self, x, return_feature=False, return_feature_list=False):
        """
        Forward Methode Compatibility OpenOOD
        Args:
            x:
            return_feature:
            return_feature_list:

        Returns:
        """
        return self.classify(x, self.steps)

    def loss(self, image: torch.Tensor, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return the classification loss. For pixel-wise ("self-classifying") problems,
        such as self-classifying MNIST, we compute the Cross-Entropy loss.
        For image-wise classification, MSE loss is returned.

        :param image [torch.Tensor]: Input image, BWHC.
        :param label [torch.Tensor]: Ground truth.

        :returns: Dictionary of identifiers mapped to computed losses.
        """
        loss = F.cross_entropy(image,label)
        return {
            "total": loss,
            "classification": loss,
        }

    def classify(self, image: torch.Tensor, steps: int = 100) -> torch.Tensor:
        """
        :param image [torch.Tensor]: Input image, BCWH.

        :returns [torch.Tensor]: Output image, BWHC
        """
        x = image.clone()
        x = pad_input(x, self, noise=self.pad_noise)
        x = self.prepare_input(x)
        return self.forward_head(x, steps)

    def predict(self, image: torch.Tensor, steps: int = 100) -> torch.Tensor:
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
                nca.num_image_channels: nca.num_image_channels
                                        + nca.num_hidden_channels,
                :,
                :,
            ] = torch.normal(
                0.5,
                0.225,
                size=(x.shape[0], nca.num_hidden_channels, x.shape[2], x.shape[3]),
            )
    return x

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.

        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss