# necessary imports
import torch
from openood.utils import config
from openood.networks import get_network
from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32 # just a wrapper around the ResNet

config_files = [
    #'./configs/networks/nca.yml',
    './configs/networks/nca_head.yml',
]

config = config.Config(*config_files)

# load the model
net = get_network(config.network).to(config.device)
net.eval()

#@title choose an implemented postprocessor
postprocessor_name = "msp"

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


"""NCA WITHOUT HEAD
           FPR@95  AUROC  AUPR_IN  AUPR_OUT   ACC
cifar100    78.91  65.03    67.54     60.77 61.07
tin         76.78  65.52    71.40     57.49 61.07
nearood     77.84  65.28    69.47     59.13 61.07
mnist       91.76  40.93    12.56     83.54 61.07
svhn        62.83  72.70    60.41     84.60 61.07
texture     87.96  59.66    70.04     43.93 61.07
places365   78.81  65.98    38.83     86.56 61.07
farood      80.34  59.81    45.46     74.66 61.07
Components within evaluator.metrics:     dict_keys(['id_acc', 'csid_acc', 'ood', 'fsood'])
Components within evaluator.scores:      dict_keys(['id', 'csid', 'ood', 'id_preds', 'id_labels', 'csid_preds', 'csid_labels'])

The predicted ID class of the first 5 samples of CIFAR-100:      [8 8 8 8 9]
The OOD score of the first 5 samples of CIFAR-100:       [0.169 0.121 0.153 0.186 0.215]
"""

"""NCA WITH HEAD
           FPR@95  AUROC  AUPR_IN  AUPR_OUT   ACC
cifar100    73.30  67.22    71.09     62.06 63.19
tin         72.07  67.66    74.38     58.69 63.19
nearood     72.68  67.44    72.73     60.38 63.19
mnist       83.04  52.62    20.47     86.89 63.19
svhn        56.08  75.41    65.47     85.70 63.19
texture     78.47  61.44    74.18     43.71 63.19
places365   72.22  69.60    46.55     87.92 63.19
farood      72.45  64.77    51.67     76.06 63.19
Components within evaluator.metrics:     dict_keys(['id_acc', 'csid_acc', 'ood', 'fsood'])
Components within evaluator.scores:      dict_keys(['id', 'csid', 'ood', 'id_preds', 'id_labels', 'csid_preds', 'csid_labels'])

The predicted ID class of the first 5 samples of CIFAR-100:      [8 8 2 8 9]
The OOD score of the first 5 samples of CIFAR-100:       [0.901 0.506 0.312 0.526 0.742]
"""