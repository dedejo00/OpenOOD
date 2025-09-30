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
postprocessor_name = "odin"

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
NCA WITH HEAD:
           FPR@95  AUROC  AUPR_IN  AUPR_OUT   ACC
cifar100    69.74  70.14    73.68     64.99 63.48
tin         70.67  69.63    75.88     60.58 63.48
nearood     70.21  69.89    74.78     62.78 63.48
mnist       86.29  53.94    17.99     88.04 63.48
svhn        37.39  85.83    78.65     92.06 63.48
texture     86.70  55.29    68.42     40.07 63.48
places365   67.17  72.33    51.20     88.78 63.48
farood      69.39  66.85    54.06     77.24 63.48
ID Accuracy:  63.477777777777774
Components within evaluator.metrics:     dict_keys(['id_acc', 'csid_acc', 'ood', 'fsood'])
Components within evaluator.scores:      dict_keys(['id', 'csid', 'ood', 'id_preds', 'id_labels', 'csid_preds', 'csid_labels'])

The predicted ID class of the first 5 samples of CIFAR-100:      [8 8 8 8 9]
The OOD score of the first 5 samples of CIFAR-100:       [0.111 0.108 0.104 0.109 0.105]
"""