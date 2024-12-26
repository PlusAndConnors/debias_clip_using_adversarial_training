import argparse
import os
from datetime import datetime
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

from base import AlternatingOptimizer, Use_original, Unlearn
from clip import clip
from datasets import Clipfeature, Clipfeature_check, initialize_data
from defaults import _C as cfg
from network import load_base_model
from network.clip import evaluate_clip
from utils import initialize_experiment

args = argparse.Namespace(
    root='./datasets',
    config_file="configs/debias_waterbird_mode.yaml",
    lr=1e-4,
    epochs=10,
    nolabel=False,
    mode='sud',
    bs=256
)

cfg.merge_from_file(args.config_file)
cfg.merge_from_list(['lr', args.lr])
cfg.num_workers = 2
random_seed = cfg.seed
torch.manual_seed(random_seed)
np.random.seed(random_seed)

print('tau_i: %.2f tau_z_i: %.2f tau_t: %.2f tau_z_t: %.2f' % (cfg.tau_i, cfg.tau_z_i, cfg.tau_t, cfg.tau_z_t))

# ----------------------------------------------------------
# loading model /label for zero-shot testing
# ----------------------------------------------------------
base_model_args = cfg.load_base_model.split('_')
base_model_components = load_base_model(base_model_args, cfg, clip=clip)
base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions = base_model_components
uo = Use_original(base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions, cfg)
if cfg.dataset == 'waterbirds':
    text_descriptions = ['This is a picture of a landbird.', 'This is a picture of a waterbird.']
else:
    text_descriptions = ['A photo of a celebrity with dark hair.', 'A photo of a celebrity with blond hair.']
# ----------------------------------------------------------
# loading model /label for original data
# ----------------------------------------------------------
args.dataset, args.arch, args.num_workers, args.verbose = cfg.dataset, cfg.load_base_model.split('_')[-1], 2, False
args.load_base_model, args.device, args.embeddings_dir = cfg.load_base_model, cfg.device, 'new_test_'
os.makedirs(args.embeddings_dir, exist_ok=True)
load_dataloaders = initialize_data(args)
dataloaders_base = load_dataloaders(args, train_shuffle=False, transform=base_transform)

train_loader_base, val_loader_base, test_loader_base = dataloaders_base
splits = ['train', 'val', 'test']

initialize_experiment(cfg)
os.makedirs('new_test', exist_ok=True)
dataset_embeddings = {}
for dix, split in enumerate(splits):
    dataset_embeddings[split] = get_dataset_embeddings(base_model, dataloaders_base[dix], args, split=split)

# ----------------------------------------------------------
# loading train / test sets
# ----------------------------------------------------------

trainset = Clipfeature_check('train', cfg)
traindata = DataLoader(trainset, batch_size=trainset.__len__(), shuffle=False)
for i, (imfeat_train, textfeat_train, labels_train_y, labels_train_s, labels_train_y_gt) in enumerate(traindata):
    y_train = ((labels_train_y_gt + 1) / 2)[:, 1].int()
    s_train = ((labels_train_s + 1) / 2)[:, 1].int()

testset = Clipfeature_check('test', cfg)
testdata = DataLoader(testset, batch_size=testset.__len__(), shuffle=False)
# ----------------------------------------------------------

query_embeddings = get_embeddings(text_descriptions, base_model, cfg, normalize=True, verbose=False)
text_embeddings = query_embeddings.float().to(cfg.device)

for i, (_, _, labels_train_y, labels_train_s, labels_train_y_gt) in enumerate(traindata):
    iter = args.epochs if cfg.nolabel else args.epochs

    model = Unlearn(cfg, uo, train_loader_base, val_loader_base, get_zeroshot_predictions, no_label=cfg.nolabel)
    model.info(text_embeddings, labels_train_y, test_loader_base, testdata)
    # model.train_sud_loss(iter, True) if cfg.mode == 'sud' else model.train(iter)

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def make_plot(ori_feature, adv_feature, ori_prediction, adv_prediction):
    batch, dim = ori_feature.shape
    markers = ['o', '^']
    sample_features = torch.cat((ori_feature.cpu().detach(), adv_feature.cpu().detach()), dim=0)
    pca_2d = PCA(n_components=2)
    reduced_features = pca_2d.fit_transform(sample_features)
    reduced_samples_2d = pca_2d.transform(sample_features)

    ori_feature_pca = reduced_samples_2d[:batch, :]
    adv_feature_pca = reduced_samples_2d[batch:, :]
    y = y_train[:batch]
    s = s_train[:batch]
    styles = {
        'class_0_best_right': {'marker': 'o', 'facecolor': 'blue', 'edgecolor': 'none'},
        'class_0_best_wrong': {'marker': 'o', 'facecolor': 'blue', 'edgecolor': 'none'},
        'class_0_worst_right': {'marker': 'o', 'facecolor': 'none', 'edgecolor': 'blue'},
        'class_0_worst_wrong': {'marker': 'o', 'facecolor': 'none', 'edgecolor': 'blue'},
        'class_1_best_right': {'marker': '^', 'facecolor': 'none', 'edgecolor': 'red'},
        'class_1_best_wrong': {'marker': '^', 'facecolor': 'none', 'edgecolor': 'red'},
        'class_1_worst_right': {'marker': '^', 'facecolor': 'red', 'edgecolor': 'none'},
        'class_1_worst_wrong': {'marker': '^', 'facecolor': 'red', 'edgecolor': 'none'},
        'adv_best': {'edgecolor': 'orange', 'linewidth': 1.5},
        'adv_worst': {'edgecolor': 'purple', 'linewidth': 1.5}
    }

    # Calculate group indices
    group_indices_ori = {
        'class_0_best_right': np.array(
            [(yi == False) and (si == False) and (oi == False) for yi, si, oi in zip(y, s, ori_prediction)]),
        'class_0_best_wrong': np.array(
            [(yi == False) and (si == False) and (oi == True) for yi, si, oi in zip(y, s, ori_prediction)]),
        'class_0_worst_right': np.array(
            [(yi == False) and (si == True) and (oi == False) for yi, si, oi in zip(y, s, ori_prediction)]),
        'class_0_worst_wrong': np.array(
            [(yi == False) and (si == True) and (oi == True) for yi, si, oi in zip(y, s, ori_prediction)]),
        'class_1_best_right': np.array(
            [(yi == True) and (si == True) and (oi == True) for yi, si, oi in zip(y, s, ori_prediction)]),
        'class_1_best_wrong': np.array(
            [(yi == True) and (si == True) and (oi == False) for yi, si, oi in zip(y, s, ori_prediction)]),
        'class_1_worst_right': np.array(
            [(yi == True) and (si == False) and (oi == True) for yi, si, oi in zip(y, s, ori_prediction)]),
        'class_1_worst_wrong': np.array(
            [(yi == True) and (si == False) and (oi == False) for yi, si, oi in zip(y, s, ori_prediction)])
    }

    # Calculate group indices for adv_feature_pca
    group_indices_adv = {
        'class_0_best_right': np.array(
            [(yi == False) and (si == False) and (ai == False) for yi, si, ai in zip(y, s, adv_prediction)]),
        'class_0_best_wrong': np.array(
            [(yi == False) and (si == False) and (ai == True) for yi, si, ai in zip(y, s, adv_prediction)]),
        'class_0_worst_right': np.array(
            [(yi == False) and (si == True) and (ai == False) for yi, si, ai in zip(y, s, adv_prediction)]),
        'class_0_worst_wrong': np.array(
            [(yi == False) and (si == True) and (ai == True) for yi, si, ai in zip(y, s, adv_prediction)]),
        'class_1_best_right': np.array(
            [(yi == True) and (si == True) and (ai == True) for yi, si, ai in zip(y, s, adv_prediction)]),
        'class_1_best_wrong': np.array(
            [(yi == True) and (si == True) and (ai == False) for yi, si, ai in zip(y, s, adv_prediction)]),
        'class_1_worst_right': np.array(
            [(yi == True) and (si == False) and (ai == True) for yi, si, ai in zip(y, s, adv_prediction)]),
        'class_1_worst_wrong': np.array(
            [(yi == True) and (si == False) and (ai == False) for yi, si, ai in zip(y, s, adv_prediction)])
    }

    plt.figure(figsize=(10, 8))
    used_labels = set()

    def plot_with_overlay(data, indices, marker_style, label_text, overlay_color='green'):
        if label_text not in used_labels:
            plt.scatter(data[indices, 0], data[indices, 1],
                        marker=marker_style['marker'],
                        facecolors=marker_style['facecolor'],
                        edgecolors=marker_style['edgecolor'],
                        label=label_text)
            used_labels.add(label_text)
        else:
            plt.scatter(data[indices, 0], data[indices, 1],
                        marker=marker_style['marker'],
                        facecolors=marker_style['facecolor'],
                        edgecolors=marker_style['edgecolor'])
        if 'wrong' in label_text:
            plt.scatter(data[indices, 0], data[indices, 1],
                        marker='s', s=150, facecolors='none', edgecolors=overlay_color, alpha=0.5)

    for key, indices in group_indices_ori.items():
        base_key = '_'.join(key.split('_')[:3])
        label_text = f'{base_key.replace("_", " ")} ori_group'
        plot_with_overlay(ori_feature_pca, indices, styles[key], label_text)

    for key in ['class_0_best_right', 'class_0_best_wrong', 'class_1_best_right', 'class_1_best_wrong']:
        indices = group_indices_adv[key]
        base_key = '_'.join(key.split('_')[:3])
        label_text = f'{base_key.replace("_", " ")} adv_group'
        plt.scatter(adv_feature_pca[indices, 0], adv_feature_pca[indices, 1],
                    marker=styles[key]['marker'],
                    facecolors='none',
                    edgecolors=styles['adv_best']['edgecolor'] if 'best' in key else styles['adv_worst']['edgecolor'],
                    linewidths=styles['adv_best']['linewidth'] if 'best' in key else styles['adv_worst']['linewidth'],
                    label=label_text if label_text not in used_labels else '')
        if label_text not in used_labels:
            used_labels.add(label_text)
        if 'wrong' in key:
            plt.scatter(adv_feature_pca[indices, 0], adv_feature_pca[indices, 1],
                        marker='s', s=150, facecolors='none', edgecolors='green', alpha=0.5)

    # for key in ['class_0_worst_right', 'class_0_worst_wrong', 'class_1_worst_right', 'class_1_worst_wrong']:
    #     indices = group_indices[key]
    #     base_key = '_'.join(key.split('_')[:3])  # Extract the base key (e.g., 'class_0_best')
    #     label_text = f'{base_key.replace("_", " ")} adv_group'
    #     plt.scatter(adv_feature_pca[indices, 0], adv_feature_pca[indices, 1],
    #                 marker=styles[key]['marker'],
    #                 facecolors='none',
    #                 edgecolors=styles['adv_best']['edgecolor'] if 'best' in key else styles['adv_worst']['edgecolor'],
    #                 linewidths=styles['adv_best']['linewidth'] if 'best' in key else styles['adv_worst']['linewidth'],
    #                 label=label_text if label_text not in used_labels else '')
    #     used_labels.add(label_text)
    #     # Overlay green square for incorrect predictions
    #     if 'wrong' in key:
    #         plt.scatter(adv_feature_pca[indices, 0], adv_feature_pca[indices, 1],
    #                     marker='s', s=150, facecolors='none', edgecolors='green', alpha=0.5)

    plt.legend()
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('[PCA] Clip waterbird Vision Output Features')
    plt.show()


for step, (batch_x, batch_y, _) in enumerate(model.train_loader):
    ori_f, adv_f = model.get_feature_with_attack(batch_x, batch_y, load_epoch=10, alpha=0.3, beta=0.9)
    adv_prediction = torch.argmax( 100 * adv_f.float() @ model.debias_text_embedding.t(), dim=-1)
    ori_prediction = torch.argmax( 100 * ori_f.float() @ model.debias_text_embedding.t(), dim=-1)
    make_plot(ori_f, adv_f, ori_prediction.cpu(), adv_prediction.cpu())
    break