import os
import clip
import torch
from torchvision.datasets import CIFAR100, CIFAR10
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
                        help="path to config file", type=str)  # debias_waterbird_mode.yaml
    parser.add_argument('--lr', default=1e-4, type=int)
    parser.add_argument('--epochs', default=100, type=int)
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


    load_dataloaders = initialize_data(args)
    dataloaders_base = load_dataloaders(args, train_shuffle=False, transform=base_transform)
    train_loader_base, val_loader_base, test_loader_base = dataloaders_base


def test_cifar(model, get_embeddings, device, cfg, transform, mode=10, down=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load('ViT-B/32', device)

    # Download the dataset
    data = CIFAR100(root=os.path.expanduser("datasets/cifar100"), download=down, train=False) \
        if mode == 100 else CIFAR100(root=os.path.expanduser("datasets/cifar10"), download=down, train=False, transform=transform)
    test_loader = DataLoader(data, batch_size=128, shuffle=False)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in data.classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    dataloader = DataLoader(dataset,  shuffle=False, batch_size=128, num_workers=2)

    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
    query_embeddings = get_embeddings(text_inputs, model, cfg, normalize=True, verbose=False)
    # Prepare the inputs
    image, class_id = cifar100[3637]
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

    # Calculate features
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

    # Pick the top 5 most similar labels for the image
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    values, indices = similarity[0].topk(5)

    # Print the result
    print("\nTop predictions:\n")
    for value, index in zip(values, indices):
        print(f"{cifar100.classes[index]:>16s}: {100 * value.item():.2f}%")




if __name__ == "__main__":
    main()
