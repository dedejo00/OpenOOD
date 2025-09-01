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
postprocessor_name = "react"

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
           FPR@95  AUROC  AUPR_IN  AUPR_OUT   ACC
cifar100    74.02  64.56    69.45     59.33 63.82
tin         73.10  64.40    72.38     55.34 63.82
nearood     73.56  64.48    70.92     57.33 63.82
mnist       81.32  66.13    23.40     93.62 63.82
svhn        65.68  65.61    55.40     79.45 63.82
texture     77.10  65.70    76.68     52.23 63.82
places365   72.09  66.03    45.06     85.69 63.82
farood      74.05  65.87    50.14     77.75 63.82
ID Accuracy:  63.82222222222222
Components within evaluator.metrics:     dict_keys(['id_acc', 'csid_acc', 'ood', 'fsood'])
Components within evaluator.scores:      dict_keys(['id', 'csid', 'ood', 'id_preds', 'id_labels', 'csid_preds', 'csid_labels'])

The predicted ID class of the first 5 samples of CIFAR-100:      [8 8 8 8 9]
The OOD score of the first 5 samples of CIFAR-100:       [1.109 0.936 0.542 1.331 0.912]
"""