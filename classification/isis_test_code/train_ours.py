import argparse
import os
from datetime import datetime
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

from base import AlternatingOptimizer, Use_original, Cifar_test
from clip import clip
from datasets import Clipfeature, Clipfeature_check, initialize_data
from defaults import _C as cfg
from network import load_base_model
from network.clip import evaluate_clip
from utils import initialize_experiment


def main():
    parser = argparse.ArgumentParser(description="fairerclip")
    parser.add_argument("--config-file", default="configs/debias_waterbird_mode.yaml", metavar="FILE",
                        help="path to config file", type=str)
    parser.add_argument('--lr', default=0.001, type=int)
    parser.add_argument("--opts", help="Modify config options using the command-line", default=None,
                        nargs='+', )
    # mode 'base': Fairer, 'at': can(ours)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
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
    args.load_base_model, args.device, args.embeddings_dir = cfg.load_base_model, cfg.device, 'new_test'
    load_dataloaders = initialize_data(args)
    dataloaders_base = load_dataloaders(args, train_shuffle=False, transform=base_transform)

    train_loader_base, val_loader_base, test_loader_base = dataloaders_base

    # ----------------------------------------------------------
    # loading train / test sets
    # ----------------------------------------------------------
    trainset = Clipfeature_check('train', cfg)
    traindata = DataLoader(trainset, batch_size=trainset.__len__(), shuffle=False)

    testset = Clipfeature_check('test', cfg)
    testdata = DataLoader(testset, batch_size=testset.__len__(), shuffle=False)
    # ----------------------------------------------------------

    query_embeddings = get_embeddings(text_descriptions, base_model, cfg, normalize=True, verbose=False)
    text_embeddings = query_embeddings.float().to(cfg.device)

    for i, (imfeat_train, textfeat_train, labels_train_y, labels_train_s, labels_train_y_gt) in enumerate(traindata):

        group_1_minus1 = (labels_train_y[:, 0] == 1) & (labels_train_s[:, 0] == -1)
        group_minus1_1 = (labels_train_y[:, 0] == -1) & (labels_train_s[:, 0] == 1)

        indices_group_1_minus1 = torch.nonzero(group_1_minus1).squeeze()
        indices_group_minus1_1 = torch.nonzero(group_minus1_1).squeeze()

        selected_indices_1_minus1 = indices_group_1_minus1[torch.randperm(len(indices_group_1_minus1))[:1]]
        selected_indices_minus1_1 = indices_group_minus1_1[torch.randperm(len(indices_group_minus1_1))[:1]]

        selected_indices = torch.cat((selected_indices_1_minus1, selected_indices_minus1_1))
        mask = torch.ones(len(labels_train_y), dtype=torch.bool)
        mask[selected_indices] = False
        imfeat_train, textfeat_train, labels_train_y, labels_train_s, labels_train_y_gt = (
            imfeat_train[mask], textfeat_train[mask], labels_train_y[mask], labels_train_s[mask], labels_train_y_gt[mask])
        iter = cfg.iters if cfg.nolabel else 1

        model = AlternatingOptimizer(cfg, uo) if True else AlternatingOptimizer(cfg)
        model.main(imfeat_train, labels_train_y, labels_train_s, labels_train_y, labels_train_s, text_embeddings, iter,
                   get_zeroshot_predictions, cfg)
        # test

    splits = ['train', 'val', 'test']

    initialize_experiment(cfg)
    os.makedirs(args.embeddings_dir, exist_ok=True)
    dataset_embeddings = {}
    for dix, split in enumerate(splits):
        dataset_embeddings[split] = get_dataset_embeddings(base_model, dataloaders_base[dix], args, split=split)

    for i, (imfeat_test, textfeat_test, labels_test_y, labels_test_s, labels_test_y_gt) in enumerate(testdata):
        debias_image, debias_text = model.get_feat(imfeat_test, textfeat_test) # 768
        text_embeddings_debias = model.get_textfeat(text_embeddings)

        dataset_predictions = get_zeroshot_predictions(debias_image, text_embeddings_debias, cfg, temperature=100.)
        print('result for testing set:')
        avg_acc, robust_acc, groups_acc = evaluate_clip(dataset_predictions, labels_test_y_gt, labels_test_s,
                                                        verbose=True)
        # cifar_test_ = Cifar_test(model.text_model, model.image_model, cfg, base_transform, mode=cfg.mode)
        # cifar_test_.set_info(uo, model) if cfg.mode == 'fairer' else cifar_test_.set_info(uo)
        # cifar_test_.cifar_test()


def evaluate_waterbirds_predictions(predictions, dataloader):
    targets = dataloader.dataset.targets_all['target']
    spurious = dataloader.dataset.targets_all['spurious']

    try:
        predictions = predictions.numpy()
    except:
        pass
    correct_by_group = [[0, 0], [0, 0]]
    total_by_group = [[0, 0], [0, 0]]
    accs_by_group = [[0, 0], [0, 0]]
    correct = predictions == targets
    for t in [0, 1]:
        for s in [0, 1]:
            ix = np.where(np.logical_and(targets == t,
                                         spurious == s))[0]
            correct_by_group[t][s] += np.sum(correct[ix])
            total_by_group[t][s] += len(ix)
            accs_by_group[t][s] = np.sum(correct[ix]) / len(ix)

    # Average accuracy
    avg_acc = (
            correct_by_group[0][0] +
            correct_by_group[0][1] +
            correct_by_group[1][0] +
            correct_by_group[1][1]
    )
    avg_acc = avg_acc * 100 / np.sum(np.array(total_by_group))

    # Adjust average accuracy
    adj_avg_acc = (
            accs_by_group[0][0] * 3498 +
            accs_by_group[0][1] * 184 +
            accs_by_group[1][0] * 56 +
            accs_by_group[1][1] * 1057
    )
    adj_avg_acc = adj_avg_acc * 100 / (3498 + 184 + 56 + 1057)

    accs_by_group = np.array(accs_by_group).flatten() * 100

    worst_acc = np.min(accs_by_group)

    return worst_acc, adj_avg_acc, avg_acc, accs_by_group

if __name__ == "__main__":
    main()
