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


class BasicNCAModel(nn.Module):
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
        fire_rate: float = 0.5,
        hidden_size: int = 128,
        use_alive_mask: bool = False,
        immutable_image_channels: bool = True,
        num_learned_filters: int = 2,
        dx_noise: float = 0.0,
        filter_padding: str = "circular",
        use_laplace: bool = False,
        kernel_size: int = 3,
        pad_noise: bool = False,
        autostepper: Optional[AutoStepper] = None,
        pixel_wise_loss: bool = False,
        threshold_activations_react: float = None,
        threshold_cell_states_react: float = None
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
        super(BasicNCAModel, self).__init__()

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

        #ReAct
        self.threshold_activations_react = threshold_activations_react
        self.threshold_cell_states_react = threshold_cell_states_react


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

        self.network = nn.Sequential(
            nn.Linear(
                self.num_channels * (self.num_filters + 1), self.hidden_size, bias=True
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_channels, bias=False),
        ).to(device)

        # initialize final layer with 0
        with torch.no_grad():
            self.network[-1].weight.data.fill_(0)

        self.meta: dict = {}

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

    def _perceive(self, x):
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
        y = torch.cat(perception, 1)
        return y

    def _update(self, x: torch.Tensor):
        """
        :param x [torch.Tensor]: Input tensor, BCWH
        """
        assert x.shape[1] == self.num_channels

        # Perception
        dx = self._perceive(x)

        # Compute delta from FFNN network
        dx = dx.permute(0, 2, 3, 1)  # B C W H --> B W H C

        if self.threshold_activations_react:
            feature = self.network[-3:-2](dx)
            # print(dx.shape)
            feature = feature.clip(max=self.threshold_activations_react)
            #feature = feature.view(feature.size(0), -1)
            # print(feature.shape)
            dx = self.network[-2:](feature)
        else:
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

    def forward_intern(
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
        if self.autostepper is None:
            for step in range(steps):
                x = self._update(x)
                if self.threshold_cell_states_react and step == 65:
                    x = x.clip(max=self.threshold_cell_states_react)
            x = x.permute((0, 2, 3, 1))  # --> BWHC
            if return_steps:
                return x, steps
            return x

        cooldown = 0
        # invariant: auto_min_steps > 0, so both of these will be defined when used
        hidden_i: torch.Tensor | None = None
        hidden_i_1: torch.Tensor | None = None
        for step in range(self.autostepper.max_steps):
            with torch.no_grad():
                if (
                    step >= self.autostepper.min_steps
                    and hidden_i is not None
                    and hidden_i_1 is not None
                ):
                    # normalized absolute difference between two hidden states
                    score = (hidden_i - hidden_i_1).abs().sum() / (
                        hidden_i.shape[0]
                        * hidden_i.shape[1]
                        * hidden_i.shape[2]
                        * hidden_i.shape[3]
                    )
                    if score >= self.autostepper.threshold:
                        cooldown = 0
                    else:
                        cooldown += 1
                    if cooldown >= self.autostepper.plateau:
                        if self.autostepper.verbose:
                            logging.info(f"Breaking after {step} steps.")
                        x = x.permute((0, 2, 3, 1))  # --> BWHC
                        if return_steps:
                            return x, step
                        return x
            # save previous hidden state
            hidden_i_1 = x[
                :,
                self.num_image_channels : self.num_image_channels
                + self.num_hidden_channels,
                :,
                :,
            ]
            # single inference time step
            x = self._update(x)
            if self.threshold_cell_states_react and step == 65:
                x = x.clip(max=self.threshold_cell_states_react)
            # set current hidden state
            hidden_i = x[
                :,
                self.num_image_channels : self.num_image_channels
                + self.num_hidden_channels,
                :,
                :,
            ]
        x = x.permute((0, 2, 3, 1))  # --> BWHC
        if return_steps:
            return x, self.autostepper.max_steps
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
        return self.classify(x, 72, reduce=False)

    def loss(self, image: torch.Tensor, label: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Return the classification loss. For pixel-wise ("self-classifying") problems,
        such as self-classifying MNIST, we compute the Cross-Entropy loss.
        For image-wise classification, MSE loss is returned.

        :param image [torch.Tensor]: Input image, BWHC.
        :param label [torch.Tensor]: Ground truth.

        :returns: Dictionary of identifiers mapped to computed losses.
        """
        # x: B W H C
        class_channels = image[
            ..., self.num_image_channels + self.num_hidden_channels :
        ]

        # Create one-hot ground truth tensor, where all pixels of the predicted class are
        # active in the respective classification channel.
        if self.pixel_wise_loss:
            y = torch.ones((image.shape[0], image.shape[1], image.shape[2])).to(
                self.device
            )
            # if binary images are classified: mask with first image channel
            if self.num_image_channels == 1:
                mask = image[..., 0] > 0
            # TODO: mask alpha channel if available
            else:
                mask = torch.Tensor([1.0])
            for i in range(image.shape[0]):
                y[i] *= label[i]
            loss_ce = (
                F.cross_entropy(
                    class_channels.permute(0, 3, 1, 2),  # B W H C --> B C W H
                    y.long(),
                    reduction="none",
                )
                * mask
            ).mean()
            loss_classification = loss_ce
        else:
            y_pred = F.softmax(class_channels, dim=-1)  # softmax along channel dim
            y_pred = torch.mean(y_pred, dim=1)  # average W
            y_pred = torch.mean(y_pred, dim=1)  # average H
            loss_mse = (
                F.mse_loss(
                    y_pred.float(),
                    F.one_hot(label.squeeze(), num_classes=self.num_classes).float(),
                    reduction="none",
                )
            ).mean()
            loss_classification = loss_mse

        loss = loss_classification
        return {
            "total": loss,
            "classification": loss_classification,
        }

    def get_meta_dict(self) -> dict:
        return dict(
            device=str(self.device),
            num_image_channels=self.num_image_channels,
            num_hidden_channels=self.num_hidden_channels,
            num_output_channels=self.num_output_channels,
            fire_rate=self.fire_rate,
            hidden_size=self.hidden_size,
            use_alive_mask=self.use_alive_mask,
            immutable_image_channels=self.immutable_image_channels,
            num_learned_filters=self.num_learned_filters,
            dx_noise=self.dx_noise,
            **self.meta,
        )

    def finetune(self):
        """
        Prepare model for fine tuning by freezing everything except the final layer,
        and setting to "train" mode.
        """
        self.train()
        if self.num_learned_filters != 0:
            for filter in self.filters:
                filter.requires_grad_ = False
        for layer in self.network[:-1]:
            layer.requires_grad_ = False

    """
    def metrics(self, pred: torch.Tensor, label: torch.Tensor) -> Dict[str, float]:
        
        Return dict of standard evaluation metrics.

        :param pred [torch.Tensor]: Predicted image.
        :param label [torch.Tensor]: Ground truth label.
        
        accuracy_macro_metric = MulticlassAccuracy(
            average="macro", num_classes=self.num_classes
        )
        accuracy_micro_metric = MulticlassAccuracy(
            average="micro", num_classes=self.num_classes
        )
        auroc_metric = MulticlassAUROC(num_classes=self.num_classes)
        f1_metric = MulticlassF1Score(num_classes=self.num_classes)

        y_prob = pred[..., -self.num_output_channels :]
        y_true = label.squeeze()
        accuracy_macro_metric.update(y_prob, y_true)
        accuracy_micro_metric.update(y_prob, y_true)
        auroc_metric.update(y_prob, y_true)
        f1_metric.update(y_prob, y_true)

        accuracy_macro = accuracy_macro_metric.compute().item()
        accuracy_micro = accuracy_micro_metric.compute().item()
        auroc = auroc_metric.compute().item()
        f1 = f1_metric.compute().item()
        return {
            "accuracy_macro": accuracy_macro,
            "accuracy_micro": accuracy_micro,
            "f1": f1,
            "auroc": auroc,
        }"""

    def get_meta_dict(self) -> dict:
        meta = super().get_meta_dict()
        meta.update(
            dict(
                num_classes=self.num_classes,
                pixel_wise_loss=self.pixel_wise_loss,
            )
        )
        return meta

    def classify(
        self, image: torch.Tensor, steps: int = 100, reduce: bool = False
    ) -> torch.Tensor:
        """
        Predict classification for an input image.

        :param image [torch.Tensor]: Input image.
        :param steps [int]: Inference steps. Defaults to 100.
        :param reduce [bool]: Return a single softmax probability. Defaults to False.

        :returns [torch.Tensor]: Single class index or vector of logits.
        """
        with torch.no_grad():
            x = image.clone()
            x = self.predict(x, steps=steps)
            hidden_channels = x[
                ...,
                self.num_image_channels : self.num_image_channels
                + self.num_hidden_channels,
            ]

            class_channels = x[
                ..., self.num_image_channels + self.num_hidden_channels :
            ]

            # mask inactive pixels
            for i in range(image.shape[0]):
                mask = torch.max(hidden_channels[i]) > 0.1
                class_channels[i] *= mask

            # if binary classification (e.g. self classifying MNIST),
            # mask away pixels with the binary image used as a mask
            if self.num_image_channels == 1:
                for i in range(image.shape[0]):
                    mask = image[i, ..., 0]
                    for c in range(self.num_classes):
                        class_channels[i, :, :, c] *= mask

            # Average over all pixels if a single categorial prediction is desired
            y_pred = F.softmax(class_channels, dim=-1)
            y_pred = torch.mean(y_pred, dim=1)
            y_pred = torch.mean(y_pred, dim=1)

            # If reduce enabled, reduce to a single scalar.
            # Otherwise, return logits of all channels as a vector.
            if reduce:
                y_pred = torch.argmax(y_pred, dim=-1)
                return y_pred
            return y_pred

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
            x = self.forward_intern(x, steps=steps)  # type: ignore[assignment]
            return x

    def validate(
        self, image: torch.Tensor, label: torch.Tensor, steps: int
    ) -> Tuple[Dict[str, float], torch.Tensor]:
        """
        :param image [torch.Tensor]: Input image, BCWH
        :param label [torch.Tensor]: Ground truth label
        :param steps [int]: Inference steps

        :returns [Tuple[float, torch.Tensor]]: Validation metric, predicted image BWHC
        """
        pred = self.classify(image.to(self.device), steps=steps)
        metrics = self.metrics(pred, label.to(self.device))
        return metrics, pred

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
