# necessary imports
import torch
from openood.utils import config
from openood.networks import get_network
from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32 # just a wrapper around the ResNet

config_files = [
    './configs/networks/nca_head.yml',
]

config = config.Config(*config_files)

# load the model
net = get_network(config.network).to(config.device)
net.eval()

#@title choose an implemented postprocessor
postprocessor_name = "ash"

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

print('ID Accuracy: ', (evaluator.metrics['id_acc']))

print('Components within evaluator.metrics:\t', evaluator.metrics.keys())
print('Components within evaluator.scores:\t', evaluator.scores.keys())
print('')
print('The predicted ID class of the first 5 samples of CIFAR-100:\t', evaluator.scores['ood']['near']['cifar100'][0][:5])
print('The OOD score of the first 5 samples of CIFAR-100:\t', evaluator.scores['ood']['near']['cifar100'][1][:5])

"""
cifar100    96.09  47.02    47.66     47.24 63.51
tin         96.03  46.37    51.04     42.76 63.51
nearood     96.06  46.70    49.35     45.00 63.51
mnist       99.91   6.76     6.03     73.49 63.51
svhn        96.78  41.35    21.77     67.98 63.51
texture     98.99  42.01    55.12     32.80 63.51
places365   92.93  51.58    22.02     79.41 63.51
farood      97.15  35.42    26.24     63.42 63.51
ID Accuracy:  63.511111111111106
Components within evaluator.metrics:     dict_keys(['id_acc', 'csid_acc', 'ood', 'fsood'])
Components within evaluator.scores:      dict_keys(['id', 'csid', 'ood', 'id_preds', 'id_labels', 'csid_preds', 'csid_labels'])

The predicted ID class of the first 5 samples of CIFAR-100:      [5 5 5 5 6]
The OOD score of the first 5 samples of CIFAR-100:       [2.328 2.309 2.316 2.319 2.338]
"""
