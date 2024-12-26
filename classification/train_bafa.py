import argparse
import os
from datetime import datetime
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

from base import Use_original, Collaboration
from clip import clip
from datasets import Clipfeature, Clipfeature_check, initialize_data
from defaults import _C as cfg
from network import load_base_model
from network.clip import evaluate_clip
from utils import initialize_experiment, update_args
from copy import deepcopy as dc


def main():
    parser = argparse.ArgumentParser(description="fairerclip")
    parser.add_argument("--config-file", default="configs/debias_waterbird_mode.yaml", metavar="FILE",
                        help="path to config file", type=str)  # debias_waterbird_mode.yaml
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--iter', default=15, type=int)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--att_mode', default='ori', type=str)  # mse or bafa or nothing
    parser.add_argument('--learn_mode', default='proj', type=str)  # linear proj vpt lora
    parser.add_argument('--txt_learn_mode', default='linear', type=str)  # linear proj
    parser.add_argument('--model', default="clip_ViTL14", type=str)  # bias or None or bafa
    parser.add_argument('--target_layer', default=1, type=int)  # bias or None or bafa
    parser.add_argument("--opts", help="Modify config options using the command-line", default=None,
                        nargs='+', )
    # mode 'base': Fairer, 'at': can(ours)
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)

    if args.learn_mode == 'vpt':
        args.lr = args.lr * 5

    cfg.merge_from_list(
        ['lr', args.lr, 'load_base_model', args.model, 'att_mode', args.att_mode, 'learn_mode', args.learn_mode,
         'embeddings_dir', 'new_test', 'txt_learn_mode', args.txt_learn_mode, 'target_layer', args.target_layer])
    cfg.num_workers = 2
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    update_args(args, cfg)

    random_seed = cfg.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

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

    os.makedirs(cfg.embeddings_dir, exist_ok=True)
    load_dataloaders = initialize_data(args)
    dataloaders_base = load_dataloaders(args, train_shuffle=False, transform=base_transform)
    # args_b = dc(args)
    # args_b.bs = 10000
    # dataloaders_10k_batch = load_dataloaders(args_b, train_shuffle=False, transform=base_transform)
    # train_loader_10k, _, _ = dataloaders_10k_batch

    train_loader_base, val_loader_base, test_loader_base = dataloaders_base
    initialize_experiment(cfg)

    testset = Clipfeature_check('test', cfg)
    testinfo = DataLoader(testset, batch_size=testset.__len__(), shuffle=False)

    '''
    * augmentation
    Img loss => PGD, CE loss
    Txt loss => PGD, CE loss
    
    * Collaboration - txt and img encoder : T = 1, 5, 10, repeat
    
    * Learning
    Img model => Projection(default) .
    Txt model => last layer(default) .
    Img loss => similarity loss
    txt loss => similarity loss, CE method
    '''
    model = Collaboration(cfg, uo, train_loader_base, val_loader_base, test_loader_base, testinfo, )
    # args.epochs if cfg.nolabel else args.epochs
    # model.exper_set_txt(train_loader_10k)
    if args.iter > 0:
        model.exper_set(False, cfg.att_mode, cfg.learn_mode, text_descriptions)
        model.clip_img_feat(data_t=cfg.dataset)
    else:
        model.clip_img_feat(False, data_t=cfg.dataset)
    t_type = 'txt'
    # if txt:
    #     model.just_txt_tuning(args.epochs)
    # elif img:
    #     model.img_tuning_sud(args.epochs)

    for i in range(args.iter):
        print('===================', 'iter :', i, ' : ', t_type, 'times', '===========================')
        if cfg.att_mode == 'nothing':  # or i < 10:
            if t_type == 'txt':
                if args.nolabel:
                    model.txt_tuning_nolabel(args.epochs, i, just_tun=True)
                else:
                    model.just_txt_tuning(args.epochs, i, just_tun=True)
                t_type = 'img'
            else:
                if args.nolabel:
                    model.img_tuning_nolabel(args.epochs, i, just_tun=True)
                else:
                    model.img_tuning_label(args.epochs, i, just_tun=True)
                t_type = 'txt'
        else:
            if t_type == 'txt':
                if args.nolabel:
                    model.txt_tuning_nolabel(args.epochs, i)
                else:
                    model.just_txt_tuning(args.epochs, i)
                t_type = 'img'
            else:
                if args.nolabel:
                    model.img_tuning_nolabel(args.epochs, i)
                else:
                    model.img_tuning_label(args.epochs, i)
                t_type = 'txt'
        model.test()
    model.test()
    model.cifar_test2()

    # model.imagenet_test()


if __name__ == "__main__":
    main()
