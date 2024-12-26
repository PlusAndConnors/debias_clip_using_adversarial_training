import argparse
import os
from datetime import datetime
from os.path import join

import numpy as np
import torch
from torch.utils.data import DataLoader

from trainer.can_base import Use_openai_clip, Collaborative_adversarial_network
from base import AlternatingOptimizer, Use_original, Unlearn
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
    parser.add_argument('--lr', default=0.0001, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--bound', default=0.6, type=float)
    parser.add_argument('--step', default=0.5, type=int)
    parser.add_argument('--iter', default=10, type=int)
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
    base = Use_openai_clip(cfg)
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
    dataloaders_base = load_dataloaders(args, train_shuffle=True, transform=base.base_transform, huggingface=True)

    train_loader_base, val_loader_base, test_loader_base = dataloaders_base
    splits = ['train', 'val', 'test']

    initialize_experiment(cfg)
    os.makedirs('new_test', exist_ok=True)
    dataset_embeddings = {}
    for dix, split in enumerate(splits):
        dataset_embeddings[split] = base.get_dataset_embeddings(dataloaders_base[dix], args, split=split)

    # ----------------------------------------------------------
    # loading train / test sets
    # ----------------------------------------------------------
    testset = Clipfeature_check('test', cfg)
    testdata = DataLoader(testset, batch_size=testset.__len__(), shuffle=False)
    # ----------------------------------------------------------

    query_embeddings = base.get_txt_feature(text_descriptions)
    text_embeddings = query_embeddings.float().to(cfg.device)

    num_iters = args.epochs

    optimizer = torch.optim.SGD(student_layer.parameters(), lr=self.cfg.lr, momentum=0.9)
    optimizer.zero_grad()
    model = Collaborative_adversarial_network(base, cfg, args)
    for epo in range(num_iters):
        for step, (batch_x, batch_y, _) in enumerate(train_loader_base):
            bias_batch_x = model.attack_vision(batch_x, text_embeddings, batch_y)
            bias_text_embedding = model.attack_text(batch_x, text_embeddings, batch_y, bias_batch_x)

            loss = model.rebuild_both(batch_x, text_embeddings, batch_y, bias_text_embedding)
            loss.backward(retain_graph=True)
            optimizer.step()
        if not epo % 3:
            for step, (batch_x, batch_y, _) in enumerate(val_loader_base):
                pass


    for imfeat_test, _, _, labels_test_s, labels_test_y_gt in testdata:
        debias_image = base.get_img_feature(test_loader_base)
        text_embeddings_debias = base.get_txt_feature(text_embeddings)

        dataset_predictions_debias_vl = base.get_zeroshot_predictions(imfeat_test, text_embeddings, 100)
        print('original vision use, acc')
        avg_acc, robust_acc, groups_acc = evaluate_clip(dataset_predictions_debias_vl, labels_test_y_gt, labels_test_s, verbose=True)
        print('our use, acc')
        avg_acc, robust_acc, groups_acc = evaluate_clip(dataset_predictions_debias_vl, labels_test_y_gt, labels_test_s,
                                                        verbose=True)


if __name__ == "__main__":
    main()
