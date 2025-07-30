# necessary imports
import torch
from openood.utils import config
from openood.networks import get_network
from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32 # just a wrapper around the ResNet

config_files = [
    './configs/networks/react_nca_head.yml',
]

config = config.Config(*config_files)

# load the model
net = get_network(config.network).to(config.device)
net.eval()

#@title choose an implemented postprocessor
postprocessor_name = "temp_scaling"

evaluator = Evaluator(
    net,
    id_name='cifar10',                     # the target ID dataset
    data_root='./data',                    # change if necessary
    config_root=None,                      # see notes above
    preprocessor=None,                     # default preprocessing for the target ID dataset
    postprocessor_name=postprocessor_name, # the postprocessor to use
    postprocessor=None,                    # if you want to use your own postprocessor
    batch_size=64,                        # for certain methods the results can be slightly affected by batch size
    shuffle=False,
    num_workers=2)                         # could use more num_workers outside colab

metrics = evaluator.eval_ood(fsood=False)

print('Components within evaluator.metrics:\t', evaluator.metrics.keys())
print('Components within evaluator.scores:\t', evaluator.scores.keys())
print('')
print('The predicted ID class of the first 5 samples of CIFAR-100:\t', evaluator.scores['ood']['near']['cifar100'][0][:5])
print('The OOD score of the first 5 samples of CIFAR-100:\t', evaluator.scores['ood']['near']['cifar100'][1][:5])