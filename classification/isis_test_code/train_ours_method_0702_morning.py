import argparse
import os
from datetime import datetime
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

from base_backup_0702_morning import AlternatingOptimizer, Use_original, Unlearn
from clip import clip
from datasets import Clipfeature, Clipfeature_check, initialize_data
from defaults import _C as cfg
from network import load_base_model
from network.clip import evaluate_clip
from utils import initialize_experiment


def main():
    parser = argparse.ArgumentParser(description="fairerclip")
    parser.add_argument("--config-file", default="configs/debias_waterbird_mode.yaml", metavar="FILE",
                        help="path to config file", type=str) # debias_waterbird_mode.yaml
    parser.add_argument('--lr', default=1e-4, type=int)
    parser.add_argument('--epochs', default=0, type=int)
    parser.add_argument('--att_mode', default=None, type=str)  # mse or bafa
    parser.add_argument('--learn_mode', default=None, type=str)  # bias or None or bafa
    parser.add_argument('--model', default="clip_ViTL14", type=str)  # bias or None or bafa
    parser.add_argument("--opts", help="Modify config options using the command-line", default=None,
                        nargs='+', )
    # mode 'base': Fairer, 'at': can(ours)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(['lr', args.lr])
    cfg.merge_from_list(['load_base_model', args.model])
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
    os.makedirs(args.embeddings_dir, exist_ok=True)
    load_dataloaders = initialize_data(args)
    dataloaders_base = load_dataloaders(args, train_shuffle=False, transform=base_transform)

    train_loader_base, val_loader_base, test_loader_base = dataloaders_base
    splits = ['train', 'val', 'test']

    initialize_experiment(cfg)
    os.makedirs('new_test', exist_ok=True)
    dataset_embeddings = {}
    # for dix, split in enumerate(splits):
    #     dataset_embeddings[split] = get_dataset_embeddings(base_model, dataloaders_base[dix], args, split=split)

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

    for i, (_, _, labels_train_y, labels_train_s, labels_train_y_gt) in enumerate(traindata):
        iter = args.epochs if cfg.nolabel else args.epochs

        model = Unlearn(cfg, uo, train_loader_base, val_loader_base, get_zeroshot_predictions, no_label=cfg.nolabel)
        model.info(text_embeddings, labels_train_y, test_loader_base, testdata)
        # model.cifar_test2()
        model.train_sud_loss(iter, True, args.att_mode, args.learn_mode) if cfg.mode == 'sud' else model.train(iter)
        # test
        model.test()
        model.cifar_test2()

if __name__ == "__main__":
    main()
