import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR100, CIFAR10
import os
import xml.etree.ElementTree as ET
from PIL import Image
class Cifar_test:
    def __init__(self, text_model, image_model, cfg, basetransform=None):
        self.text_model = text_model
        self.image_model = image_model
        self.cfg = cfg
        self.transform = None
        if basetransform is not None:
            self.transform = basetransform

    def zeroshot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates]  # Format with class
                class_embeddings = self.text_model.txt_inference(texts)  # Embed with text encoder
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.cfg.device)
        return zeroshot_weights

    def accuracy(self, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

    def zero_shot_evaluation(self, dataset, class_names, templates):
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        zeroshot_weights = self.zeroshot_classifier(class_names, templates)

        top1, top5, n = 0., 0., 0.
        with torch.no_grad():
            for images, target in tqdm(loader):
                images = images.to(self.cfg.device)
                target = target.to(self.cfg.device)

                # Predict
                image_features = self.image_model.img_inference(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ zeroshot_weights

                # Measure accuracy
                acc1, acc5 = self.accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100
        return top1, top5

    def cifar_test(self):
        # Load the datasets
        root = os.path.expanduser("~/.cache")
        test_cifar100 = CIFAR100(root, download=True, train=False, transform=self.transform)
        test_cifar10 = CIFAR10(root, download=True, train=False, transform=self.transform)

        # Prompt templates for CIFAR-10 and CIFAR-100
        templates = [
            'a photo of a {}.', 'a blurry photo of a {}.', 'a black and white photo of a {}.',
            'a low contrast photo of a {}.', 'a high contrast photo of a {}.', 'a bad photo of a {}.',
            'a good photo of a {}.', 'a photo of a small {}.', 'a photo of a big {}.', 'a photo of the {}.',
            'a blurry photo of the {}.', 'a black and white photo of the {}.', 'a low contrast photo of the {}.',
            'a high contrast photo of the {}.', 'a bad photo of the {}.', 'a good photo of the {}.',
            'a photo of the small {}.', 'a photo of the big {}.',
        ]

        # Evaluate on CIFAR-100
        cifar100_top1, cifar100_top5 = self.zero_shot_evaluation(test_cifar100, test_cifar100.classes, templates)
        print(f"CIFAR-100 Zero-shot CLIP model Top-1 accuracy: {cifar100_top1:.2f}%")
        print(f"CIFAR-100 Zero-shot CLIP model Top-5 accuracy: {cifar100_top5:.2f}%")

        # Evaluate on CIFAR-10
        cifar10_top1, cifar10_top5 = self.zero_shot_evaluation(test_cifar10, test_cifar10.classes, templates)
        print(f"CIFAR-10 Zero-shot CLIP model Top-1 accuracy: {cifar10_top1:.2f}%")
        print(f"CIFAR-10 Zero-shot CLIP model Top-5 accuracy: {cifar10_top5:.2f}%")


def parse_imagenet_xml(annotation_path):
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    class_name = root.find("object").find("name").text
    return class_name


# ImageNet validation dataset class
class ImageNetValDataset(Dataset):
    def __init__(self, img_dir, anno_dir, transform=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.img_paths = sorted(os.listdir(self.img_dir))
        self.anno_paths = sorted(os.listdir(self.anno_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        anno_path = os.path.join(self.anno_dir, self.anno_paths[idx])

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load label (from XML)
        label = parse_imagenet_xml(anno_path)

        return image, label


class Imagenet_test:
    def __init__(self, text_model, image_model, cfg):
        self.text_model = text_model
        self.image_model = image_model
        self.cfg = cfg

    def zeroshot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates]  # Format with class
                class_embeddings = self.text_model.get_embedding(texts)  # Embed with text encoder
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.cfg.device)
        return zeroshot_weights

    def accuracy(self, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

    def zero_shot_evaluation(self, dataset, class_names, templates):
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        zeroshot_weights = self.zeroshot_classifier(class_names, templates)

        top1, top5, n = 0., 0., 0.
        with torch.no_grad():
            for images, target in tqdm(loader):
                images = images.to(self.cfg.device)
                target = target.to(self.cfg.device)

                # Predict
                image_features = self.image_model.encod(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                logits = 100. * image_features @ zeroshot_weights

                # Measure accuracy
                acc1, acc5 = self.accuracy(logits, target, topk=(1, 5))
                top1 += acc1
                top5 += acc5
                n += images.size(0)

        top1 = (top1 / n) * 100
        top5 = (top5 / n) * 100
        return top1, top5

    def imagenet_test(self, img_dir, anno_dir, class_names, templates):
        # Load ImageNet validation dataset
        val_dataset = ImageNetValDataset(img_dir, anno_dir, transform=self.text_model.base_transform)

        # Evaluate on ImageNet validation set
        imagenet_top1, imagenet_top5 = self.zero_shot_evaluation(val_dataset, class_names, templates)
        print(f"ImageNet Zero-shot CLIP model Top-1 accuracy: {imagenet_top1:.2f}%")
        print(f"ImageNet Zero-shot CLIP model Top-5 accuracy: {imagenet_top5:.2f}%")
