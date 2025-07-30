from openood.utils import config
from openood.datasets import get_dataloader, get_ood_dataloader
from openood.evaluators import get_evaluator

from openood.networks import get_network
from tqdm import tqdm
import numpy as np
import torch
import os
import os.path as osp



# load config files for cifar10 baseline
config_files = [
    './configs/datasets/cifar10/cifar10.yml',
    './configs/datasets/cifar10/cifar10_ood.yml',
    './configs/networks/react_nca_head.yml',
    './configs/pipelines/test/test_ood.yml',
    './configs/preprocessors/base_preprocessor.yml',
    './configs/postprocessors/msp.yml',
]
config = config.Config(*config_files)

# modify config
config.num_workers = 1
config.save_output = False
config.parse_refs()

# get dataloader
id_loader_dict = get_dataloader(config)
ood_loader_dict = get_ood_dataloader(config)
# init network
net = get_network(config.network).to(config.device)
# init ood evaluator
evaluator = get_evaluator(config)

def save_arr_to_dir(arr, dir):
    # Ensure the directory exists
    os.makedirs(os.path.dirname(dir), exist_ok=True)
    # Save the array to the file
    with open(dir, 'wb') as f:
        np.save(f, arr)

save_root = f'./results/{config.exp_name}'

# save id (test & val) results
net.eval()
modes = ['test', 'val']
for mode in modes:
    dl = id_loader_dict[mode]
    dataiter = iter(dl)

    logits_list = []
    feature_list = []
    label_list = []

    for i in tqdm(range(1,
                        len(dataiter) + 1),
                  desc='Extracting reults...',
                  position=0,
                  leave=True):
        batch = next(dataiter)
        data = batch['data'].to(config.device)
        label = batch['label']
        with torch.no_grad():
            logits_cls = net.forward_threshold(data, threshold=-10.886123)
        logits_list.append(logits_cls.data.to('cpu').numpy())
        #feature_list.append(feature.data.to('cpu').numpy())
        label_list.append(label.numpy())

    logits_arr = np.concatenate(logits_list)
    #feature_arr = np.concatenate(feature_list)
    label_arr = np.concatenate(label_list)

    save_arr_to_dir(logits_arr, osp.join(save_root, 'id', f'{mode}_logits.npy'))
    #save_arr_to_dir(feature_arr, osp.join(save_root, 'id', f'{mode}_feature.npy'))
    save_arr_to_dir(label_arr, osp.join(save_root, 'id', f'{mode}_labels.npy'))

# save ood results
net.eval()
ood_splits = ['nearood', 'farood']
for ood_split in ood_splits:
    for dataset_name, ood_dl in ood_loader_dict[ood_split].items():
        dataiter = iter(ood_dl)

        logits_list = []
        feature_list = []
        label_list = []

        for i in tqdm(range(1,
                            len(dataiter) + 1),
                      desc='Extracting reults...',
                      position=0,
                      leave=True):
            batch = next(dataiter)
            data = batch['data'].to(config.device)
            label = batch['label']

            with torch.no_grad():
                logits_cls = net.forward_threshold(data, threshold=-10.886123)
            logits_list.append(logits_cls.data.to('cpu').numpy())
            #feature_list.append(feature.data.to('cpu').numpy())
            label_list.append(label.numpy())

        logits_arr = np.concatenate(logits_list)
        #feature_arr = np.concatenate(feature_list)
        label_arr = np.concatenate(label_list)

        save_arr_to_dir(logits_arr, osp.join(save_root, ood_split, f'{dataset_name}_logits.npy'))
        #save_arr_to_dir(feature_arr, osp.join(save_root, ood_split, f'{dataset_name}_feature.npy'))
        save_arr_to_dir(label_arr, osp.join(save_root, ood_split, f'{dataset_name}_labels.npy'))

# load logits, feature, label for this benchmark
results = dict()
# for id
modes = ['val', 'test']
results['id'] = dict()
for mode in modes:
    results['id'][mode] = dict()
    #results['id'][mode]['feature'] = np.load(osp.join(save_root, 'id', f'{mode}_feature.npy'))
    results['id'][mode]['logits'] = np.load(osp.join(save_root, 'id', f'{mode}_logits.npy'))
    results['id'][mode]['labels'] = np.load(osp.join(save_root, 'id', f'{mode}_labels.npy'))

# for ood
split_types = ['nearood', 'farood']
for split_type in split_types:
    results[split_type] = dict()
    dataset_names = config['ood_dataset'][split_type].datasets
    for dataset_name in dataset_names:
        results[split_type][dataset_name] = dict()
        #results[split_type][dataset_name]['feature'] = np.load(osp.join(save_root, split_type, f'{dataset_name}_feature.npy'))
        results[split_type][dataset_name]['logits'] = np.load(osp.join(save_root, split_type, f'{dataset_name}_logits.npy'))
        results[split_type][dataset_name]['labels'] = np.load(osp.join(save_root, split_type, f'{dataset_name}_labels.npy'))



# build msp method (pass in pre-saved logits)
def msp_postprocess(logits):
    score = torch.softmax(logits, dim=1)
    conf, pred = torch.max(score, dim=1)
    return pred, conf

def print_nested_dict(dict_obj, indent = 0):
    ''' Pretty Print nested dictionary with given indent level
    '''
    # Iterate over all key-value pairs of dictionary
    for key, value in dict_obj.items():
        # If value is dict type, then print nested dict
        if isinstance(value, dict):
            print(' ' * indent, key, ':', '{')
            print_nested_dict(value, indent + 2)
            print(' ' * indent, '}')
        else:
            print(' ' * indent, key, ':', value.shape)



print_nested_dict(results)

from openood.evaluators.metrics import compute_all_metrics, acc

def eval_id(postprocess_results):
    for mode_res in postprocess_results['id']:
        print(postprocess_results['id'][mode_res])
        accuracy = acc(postprocess_results['id'][mode_res][0], postprocess_results['id'][mode_res][2])
        print("ID Accuracy: {:.2f}".format(accuracy * 100), flush=True)

# get pred, conf, gt from MSP postprocessor (can change to your custom_postprocessor here)
postprocess_results = dict()
# id
modes = ['val', 'test']
postprocess_results['id'] = dict()
for mode in modes:
    pred, conf = msp_postprocess(torch.from_numpy(results['id'][mode]['logits']))
    pred, conf = pred.numpy(), conf.numpy()
    gt = results['id'][mode]['labels']
    postprocess_results['id'][mode] = [pred, conf, gt]

eval_id(postprocess_results)

# ood
split_types = ['nearood', 'farood']
for split_type in split_types:
    postprocess_results[split_type] = dict()
    dataset_names = config['ood_dataset'][split_type].datasets
    for dataset_name in dataset_names:
        pred, conf = msp_postprocess(torch.from_numpy(results[split_type][dataset_name]['logits']))
        pred, conf = pred.numpy(), conf.numpy()
        gt = results[split_type][dataset_name]['labels']
        gt = -1 * np.ones_like(gt)   # hard set to -1 here
        postprocess_results[split_type][dataset_name] = [pred, conf, gt]

def print_all_metrics(metrics):
    [fpr, auroc, aupr_in, aupr_out, accuracy] = metrics
    print('FPR@95: {:.2f}, AUROC: {:.2f}'.format(100 * fpr, 100 * auroc),
            end=' ',
            flush=True)
    print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(
        100 * aupr_in, 100 * aupr_out),
            flush=True)
    print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
    print(u'\u2500' * 70, flush=True)




def eval_ood(postprocess_results):
    [id_pred, id_conf, id_gt] = postprocess_results['id']['test']
    split_types = ['nearood', 'farood']

    for split_type in split_types:
        metrics_list = []
        print(f"Performing evaluation on {split_type} datasets...")
        dataset_names = config['ood_dataset'][split_type].datasets

        for dataset_name in dataset_names:
            [ood_pred, ood_conf, ood_gt] = postprocess_results[split_type][dataset_name]

            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])
            print(f'Computing metrics on {dataset_name} dataset...')

            ood_metrics = compute_all_metrics(conf, label, pred)
            print_all_metrics(ood_metrics)
            metrics_list.append(ood_metrics)
        print('Computing mean metrics...', flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0)
        print_all_metrics(metrics_mean)

eval_ood(postprocess_results)


""" threshold=0.672239 percentile 90
ID Accuracy: 61.19
Performing evaluation on nearood datasets...
Computing metrics on cifar100 dataset...
FPR@95: 81.39, AUROC: 63.35 AUPR_IN: 65.48, AUPR_OUT: 59.66
ACC: 61.19
──────────────────────────────────────────────────────────────────────
Computing metrics on tin dataset...
FPR@95: 76.88, AUROC: 66.94 AUPR_IN: 72.36, AUPR_OUT: 59.46
ACC: 61.19
──────────────────────────────────────────────────────────────────────
Computing mean metrics...
FPR@95: 79.13, AUROC: 65.14 AUPR_IN: 68.92, AUPR_OUT: 59.56
ACC: 61.19
──────────────────────────────────────────────────────────────────────
Performing evaluation on farood datasets...
Computing metrics on mnist dataset...
FPR@95: 78.86, AUROC: 57.89 AUPR_IN: 27.12, AUPR_OUT: 89.92
ACC: 61.19
──────────────────────────────────────────────────────────────────────
Computing metrics on svhn dataset...
FPR@95: 64.91, AUROC: 70.33 AUPR_IN: 57.82, AUPR_OUT: 83.43
ACC: 61.19
──────────────────────────────────────────────────────────────────────
Computing metrics on texture dataset...
FPR@95: 83.99, AUROC: 63.32 AUPR_IN: 73.37, AUPR_OUT: 48.29
ACC: 61.19
──────────────────────────────────────────────────────────────────────
Computing metrics on place365 dataset...
FPR@95: 83.13, AUROC: 63.36 AUPR_IN: 34.12, AUPR_OUT: 85.50
ACC: 61.19
──────────────────────────────────────────────────────────────────────
Computing mean metrics...
FPR@95: 77.72, AUROC: 63.73 AUPR_IN: 48.10, AUPR_OUT: 76.78
ACC: 61.19
──────────────────────────────────────────────────────────────────────

"""


""" threshold=-1.886123 percentile 50
ID Accuracy: 61.46
Performing evaluation on nearood datasets...
Computing metrics on cifar100 dataset...
FPR@95: 81.69, AUROC: 63.12 AUPR_IN: 65.33, AUPR_OUT: 59.29
ACC: 61.46
──────────────────────────────────────────────────────────────────────
Computing metrics on tin dataset...
FPR@95: 76.87, AUROC: 66.72 AUPR_IN: 72.28, AUPR_OUT: 59.00
ACC: 61.46
──────────────────────────────────────────────────────────────────────
Computing mean metrics...
FPR@95: 79.28, AUROC: 64.92 AUPR_IN: 68.80, AUPR_OUT: 59.14
ACC: 61.46
──────────────────────────────────────────────────────────────────────
Performing evaluation on farood datasets...
Computing metrics on mnist dataset...
FPR@95: 78.96, AUROC: 57.80 AUPR_IN: 27.04, AUPR_OUT: 89.82
ACC: 61.46
──────────────────────────────────────────────────────────────────────
Computing metrics on svhn dataset...
FPR@95: 64.68, AUROC: 70.25 AUPR_IN: 57.77, AUPR_OUT: 83.27
ACC: 61.46
──────────────────────────────────────────────────────────────────────
Computing metrics on texture dataset...
FPR@95: 84.20, AUROC: 63.18 AUPR_IN: 73.40, AUPR_OUT: 47.61
ACC: 61.46
──────────────────────────────────────────────────────────────────────
Computing metrics on place365 dataset...
FPR@95: 82.76, AUROC: 63.33 AUPR_IN: 34.17, AUPR_OUT: 85.43
ACC: 61.46
──────────────────────────────────────────────────────────────────────
Computing mean metrics...
FPR@95: 77.65, AUROC: 63.64 AUPR_IN: 48.10, AUPR_OUT: 76.53
ACC: 61.46
──────────────────────────────────────────────────────────────────────

"""