from transformers import CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from copy import deepcopy as dc
from can.attack import debias_vl
from attack_code import PGD


class Collaborative_adversarial_network:
    def __init__(self, base, cfg, args):
        # base = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.cfg = cfg
        self.base = dc(base)
        # vision
        lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["module_name"], lora_dropout=0.1, bias="none")
        self.lora_model = get_peft_model(base, lora_config)

        # text
        self.optimizer = torch.optim.Adam(self.lora_model.parameters(), lr=cfg.lr)  # "lr": 1e-4
        self.debias_vl = debias_vl
        self.reset_t()
        self.adv = PGD(self.lora_model, args.bound, args.step, args.iter, device=cfg.device, P=self.P, no_label=cfg.nolabel, using=self.text_model)

    def attack_vision(self, batch_x, text_embeddings, batch_y):
        pass

    def attack_text(self, batch_x, text_embeddings, batch_y, bias_batch_x):
        pass

    def rebuild_both(self, batch_x, text_embeddings, batch_y, bias_batch_x=None):  # + bias_text_embedding=None
        device = self.cfg.device
        img_feature = self.lora_model.vision_model(batch_x)
        txt_feature = self.lora_model.text_encoder(text_embeddings)
        debias_txt_feature = self.debias_vl(txt_feature)

        ori_logit = self.debias_model.get_zeroshot_predictions(img_feature, txt_feature, temperature=100)
        debias_logit = self.debias_model.get_zeroshot_predictions(img_feature, debias_txt_feature, temperature=100)
        adv_logit = self.debias_model.get_zeroshot_predictions(bias_batch_x, txt_feature, temperature=100)

        best_group = batch_y == torch.argmax(ori_logit, dim=-1)
        bias_group = batch_y != torch.argmax(ori_logit, dim=-1)
        att_succ = (batch_y[best_group] != torch.argmax(adv_logit, dim=-1)[best_group])
        att_fail = (batch_y[best_group] == torch.argmax(adv_logit, dim=-1)[best_group])

        # CE loss
        ce = torch.tensor(0, device=device).float()
        ce_loss_att, ce_loss_bias, ce_loss_base = ce.clone(), ce.clone(), ce.clone()
        if torch.any(att_succ):
            ce_loss_att = F.cross_entropy(adv_logit[best_group][att_succ], batch_y[best_group][att_succ])
            ce_loss_base1 = F.cross_entropy(ori_logit[best_group][att_succ], batch_y[best_group][att_succ])
            ce_loss_base += ce_loss_base1

        if torch.any(att_fail):
            ce_loss_base2 = F.cross_entropy(ori_logit[best_group][att_fail], batch_y[best_group][att_fail])
            ce_loss_base += ce_loss_base2 * 0.5

        if torch.any(bias_group):
            ce_loss_bias = F.cross_entropy(ori_logit[bias_group], batch_y[bias_group])
        # if torch.any(best_group):
        #     ce_loss_base = F.cross_entropy(ori_logit_debias_s[best_group], batch_y[best_group])
        ce_loss = (ce_loss_att + ce_loss_bias + ce_loss_base) * 0.1
        if not ce_loss:
            return None
        else:
            self.att_loss_t += ce_loss_att.item()
            self.bias_loss_t += ce_loss_bias.item()
            loss = ce_loss_att + ce_loss_bias + ce_loss_base
            return loss

    def reset_t(self):
        self.att_loss_t, self.bias_loss_t = 0, 0

    def check_t(self):
        return self.att_loss_t, self.bias_loss_t


class Use_openai_clip:
    def __init__(self, cfg):
        # Initialize CLIP model and processor
        self.base_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.base_transform = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.get_embeddings_ = self.base_model.get_text_features
        self.get_dataset_embeddings_ = self.base_model.get_image_features
        self.cfg = cfg

    def get_embedding(self, text):
        inputs = self.base_transform(text=text, return_tensors="pt", padding=True)
        embeddings = self.get_embeddings_(**inputs)
        return embeddings

    def get_dataset_embeddings(self, dataloader, args, split):
        dataset = args.dataset.replace('_iid', '').split('_min')[0]
        embedding_fname = f'd={dataset}-s={split}-m={args.load_base_model}.pt'
        os.makedirs(args.embeddings_dir, exist_ok=True)
        embedding_path = os.path.join(args.embeddings_dir, embedding_fname)
        if os.path.exists(embedding_path):
            embeddings = torch.load(embedding_path)
            return embeddings
        self.base_model.to(args.device)
        self.base_model.eval()
        all_embeddings = []
        with torch.no_grad():
            for ix, data in enumerate(
                    tqdm(dataloader, desc=f'Computing {args.load_base_model} image embeddings for {split} split')):
                inputs, labels, data_ix = data
                inputs = inputs.to(args.device)
                labels = labels.to(args.device)
                embeddings = self.get_img_feature(inputs).float().cpu()
                all_embeddings.append(embeddings)
        torch.save(torch.cat(all_embeddings), embedding_path)
        return embeddings

    def get_txt_feature(self, text):
        with torch.no_grad():
            inputs = self.base_transform(text=text, return_tensors="pt", padding=True)
            embeddings = self.get_embeddings_(**inputs)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    def get_img_feature(self, img):
        with torch.no_grad():
            embeddings = self.get_dataset_embeddings_(img['pixel_values'].squeeze(1))
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    def get_zeroshot_predictions(self, im_feature, txt_feature, temperature=100):
        # Normalize the features
        im_feature = im_feature / im_feature.norm(dim=-1, keepdim=True)
        txt_feature = txt_feature / txt_feature.norm(dim=-1, keepdim=True)

        # Compute the dot product
        logit = temperature * torch.matmul(im_feature, txt_feature.t())
        return torch.argmax(F.softmax(logit))
