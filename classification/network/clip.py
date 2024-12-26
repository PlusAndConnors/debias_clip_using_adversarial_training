import os
import numpy as np
import torch
from tqdm import tqdm
from clip import clip
from utils.logging import summarize_acc


def get_embeddings(text, clip_model, args, normalize=True, verbose=True):
    if verbose:
        desc = '-> Text descriptions for zero-shot classification:'
        print('-' * len(desc))
        print(desc)
        num_display = 5
        for d in text[:num_display]:
            print(f'   - {d}')
        if len(text) > num_display:
            print(f'     ...')
            for d in text[-num_display:]:
                print(f'   - {d}')
        print('-' * len(desc))
    if 'clip' in args.load_base_model or 'cloob' in args.load_base_model:
        text_tokens = clip.tokenize(text)
    elif 'slip' in args.load_base_model:
        slip_tokenizer = CLIPSimpleTokenizer()
        text_tokens = slip_tokenizer(text)
        text_tokens = text_tokens.view(-1, 77).contiguous()
    clip_model.to(args.device)
    clip_model.eval()
    with torch.no_grad():
        text_tokens = text_tokens.to(args.device)
        text_embeddings = clip_model.encode_text(text_tokens).float().cpu()
        if normalize:
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
    clip_model.cpu()
    return text_embeddings


def get_dataset_embeddings(model, dataloader, args, split='train'):
    return get_clip_embeddings(model, dataloader, args, split)


def get_clip_embeddings(model, dataloader, args,
                        split='train', verbose=False):
    verbose = True if args.verbose else False

    dataset = args.dataset.replace('_iid', '').split('_min')[0]
    embedding_fname = f'd={dataset}-s={split}-m={args.load_base_model}.pt'
    embedding_path = os.path.join(args.embeddings_dir, embedding_fname)
    try:
        if os.path.exists(embedding_path):
            if verbose:
                print(f'-> Retrieving image embeddings from {embedding_path}!')
            embeddings = torch.load(embedding_path)
            return embeddings
        else:
            if verbose:
                print(f'-> Image embeddings from {embedding_path} not found.')
    except:
        pass

    model.to(args.device)
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for ix, data in enumerate(tqdm(dataloader,
                                       desc=f'Computing {args.load_base_model} image embeddings for {split} split')):
            inputs, labels, data_ix = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            try:
                embeddings = model.encode_image(inputs).float().cpu()
                all_embeddings.append(embeddings)
                inputs = inputs.cpu()
                labels = labels.cpu()
            except Exception as e:
                import pdb;
                pdb.set_trace()
    model.cpu()

    # Save to disk
    torch.save(torch.cat(all_embeddings), embedding_path)
    if verbose:
        print(f'-> Saved image embeddings to {embedding_path}!')

    return torch.cat(all_embeddings)


def evaluate_clip(clip_predictions, target, group, verbose=False, return_adj = False):
    """
    General method for classification validation
    Args:
    - clip_predictions (np.array): predictions
    - dataloader (torch.utils.data.DataLoader): (unshuffled) dataloader
    """
    targets_t = target.numpy().astype(int).argmax(1)
    targets_s = group.numpy().astype(int).argmax(1)

    correct_by_groups = np.zeros([len(np.unique(targets_t)), len(np.unique(targets_s))])
    auroc_by_groups = np.zeros([len(np.unique(targets_t)), len(np.unique(targets_s))])
    total_by_groups = np.zeros(correct_by_groups.shape)
    losses_by_groups = np.zeros(correct_by_groups.shape)

    correct = (clip_predictions == targets_t)
    for ix, y in enumerate(targets_t):
        s = targets_s[ix]
        correct_by_groups[int(y)][int(s)] += correct[ix].item()
        total_by_groups[int(y)][int(s)] += 1

    avg_acc, robust_acc, groups_acc, adj = summarize_acc(correct_by_groups, total_by_groups, return_groups=True,
                                                    stdout=verbose, return_adj=True)
    # adj_avg_acc = (
    #         correct_by_groups[0][0] / total_by_groups[0][0] +
    #         correct_by_groups[0][1] / total_by_groups[0][1] +
    #         correct_by_groups[1][0] / total_by_groups[1][0] +
    #         correct_by_groups[1][1] / total_by_groups[1][1]
    # )
    # adj_avg_acc = adj_avg_acc * 100 / (3498 + 184 + 56 + 1057)

    # adj_avg_acc = (
    #         correct_by_groups[0][0] +
    #         correct_by_groups[0][1] +
    #         correct_by_groups[1][0] +
    #         correct_by_groups[1][1]
    # )
    # adj_avg_acc = adj_avg_acc * 100 / total_by_groups.sum()
    if return_adj:
        return avg_acc, robust_acc, groups_acc, adj
    return avg_acc, robust_acc, groups_acc


def classify_with_embeddings(image_embeddings, text_embeddings, args, temperature=100.):
    with torch.no_grad():
        _image_embeddings = (image_embeddings /
                             image_embeddings.norm(dim=-1, keepdim=True))

        _text_embeddings = (text_embeddings /
                            text_embeddings.norm(dim=-1, keepdim=True))

        cross = _image_embeddings @ _text_embeddings.T
        text_probs = (temperature * cross).softmax(dim=-1)
        _, predicted = torch.max(text_probs.data, 1)

    return predicted.cpu().numpy()


def get_zeroshot_predictions(key_embeddings, text_embeddings, args, temperature=100.):
    predictions = classify_with_embeddings(
        key_embeddings, text_embeddings, args,
        temperature=100.
    )
    return predictions


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
