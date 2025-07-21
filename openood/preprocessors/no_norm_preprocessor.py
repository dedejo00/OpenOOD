import torchvision.transforms as tvs_trans

from openood.utils.config import Config

from .transform import Convert, interpolation_modes, normalization_dict


class NoNormPreProcessor():
    """For train dataset standard transformation."""
    def __init__(self, config: Config):
        self.pre_size = config.dataset.pre_size
        self.image_size = config.dataset.image_size
        self.interpolation = interpolation_modes[config.dataset.interpolation]

        if 'imagenet' in config.dataset.name:
            self.transform = tvs_trans.Compose([
                tvs_trans.RandomResizedCrop(self.image_size,
                                            interpolation=self.interpolation),
                tvs_trans.RandomHorizontalFlip(0.5),
                tvs_trans.ToTensor(),
            ])
        elif 'aircraft' in config.dataset.name or 'cub' in config.dataset.name:
            self.transform = tvs_trans.Compose([
                tvs_trans.Resize(self.pre_size,
                                 interpolation=self.interpolation),
                tvs_trans.RandomCrop(self.image_size),
                tvs_trans.RandomHorizontalFlip(),
                tvs_trans.ColorJitter(brightness=32. / 255., saturation=0.5),
                tvs_trans.ToTensor(),
            ])
        else:
            self.transform = tvs_trans.Compose([
                Convert('RGB'),
                tvs_trans.Resize(self.pre_size,
                                 interpolation=self.interpolation),
                tvs_trans.CenterCrop(self.image_size),
                tvs_trans.RandomHorizontalFlip(),
                tvs_trans.RandomCrop(self.image_size, padding=4),
                tvs_trans.ToTensor(),
            ])

    def setup(self, **kwargs):
        pass

    def __call__(self, image):
        return self.transform(image)
