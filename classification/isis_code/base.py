import build_kernel as bk
import torch
from tqdm import tqdm
import adversarial_training as at
import torch.nn.functional as F
from copy import deepcopy as dc
from can.attack.debias_vl import debais_vl_s_c, debias_vl, bias_vl
from can.attack.adversarial_attack import PGD
from torch import nn
from can.utils.tools import get_parameter_count
from network.clip import evaluate_clip


class AlternatingOptimizer:
    def __init__(self, opts, uo=None):
        self.mode = opts.mode
        if uo is not None:
            if self.mode == 'base':
                self.image_model = dc(uo)
                self.text_model = dc(uo)
            elif self.mode == 'debias_vl':
                self.image_model = dc(uo)
                uo.debias()
                self.text_model = dc(uo)
            elif self.mode == 'at':
                model = AttackTarget(opts, uo.base_model)
                self.text_model = dc(model)
                model.set_target()
                self.image_model = dc(model)

            elif self.mode == 'fairer':
                self.image_model = bk.KernelMethodY(opts, 'image')
                self.text_model = bk.KernelMethodY(opts, 'text')
            else:
                self.image_model = bk.KernelMethodY(opts, 'image')
                self.text_model = bk.KernelMethodY(opts, 'text')

    def datasetting(self, dataloader):
        self.data_loader = dataloader

    def main(self, X_I, Y_I, S_I, Y_D, S_D, text_embeddings, num_iters, get_zeroshot_predictions, cfg):

        if self.mode == 'base' or self.mode == 'debias_vl':
            pass
        else:
            self.image_model.solver(X=X_I, Y=Y_I, S=S_I, Z=None)
            y_binary = ((Y_I + 1) / 2)[:, 1].int()
            X_T = text_embeddings[y_binary]

            for iter in tqdm(range(num_iters)):

                # Updating the pseudo-labels
                if iter > 0:
                    debias_image_train, debias_text_train = self.get_feat(X_I, X_T)
                    text_embeddings_debias = self.get_textfeat(text_embeddings)
                    dataset_predictions_train = get_zeroshot_predictions(debias_image_train, text_embeddings_debias,
                                                                         cfg, temperature=100.)
                    Y_D = (torch.nn.functional.one_hot(torch.from_numpy(dataset_predictions_train.astype(int)),
                                                       num_classes=2)) * 2 - 1
                    Y_I = Y_D

                    y_binary = ((Y_I + 1) / 2)[:, 1].int()
                    X_T = text_embeddings[y_binary]

                Z_I = self.image_model.encod(X_I)  # Z_I encoder output
                self.text_model.solver(X=X_T, Y=Y_D, S=S_D, Z=Z_I)

                Z_D = self.text_model.encod(X_T)
                self.image_model.solver(X=X_I, Y=Y_I, S=S_I, Z=Z_D)

                print(f'Training {iter + 1}/{num_iters} done!')

    def get_feat(self, X_I, X_D):
        Z_D = self.text_model.encod(X_D)
        Z_I = self.image_model.encod(X_I)

        return Z_I, Z_D

    def get_textfeat(self, X_D):
        Z_D = self.text_model.encod(X_D)
        return Z_D

    def get_imgfeat(self, X_I):
        Z_I = self.image_model.encod(X_I)
        return Z_I


class Unlearn:
    def __init__(self, opts, uo, train_loader_base, val_loader_base, get_zeroshot_predictions, no_label=False):
        self.mode = opts.mode
        self.cfg = opts
        self.P = uo.debias()

        model = AttackTarget(opts, uo.base_model.float().to(opts.device))
        self.text_model = dc(uo)
        self.image_model = dc(model)

        self.train_loader = train_loader_base
        self.val_loader = val_loader_base
        self.len_t = len(val_loader_base.dataset)

        self.get_zeroshot_predictions = get_zeroshot_predictions
        self.init_set()
        self.no_label = no_label
        self.bias_sample_check = True
        self.cls_num_list = [0, 1]

    def rebuilding(self):
        import torch
        from transformers import CLIPProcessor, CLIPModel

        model_url = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
        model_path = "ViT-L-14.pt"
        torch.hub.download_url_to_file(model_url, model_path)

        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        from peft import get_peft_model, LoraConfig
        lora_config = LoraConfig(
            target_modules=["vision_model.encoder.layers"],
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none"
        )

    def info(self, text_embeddings, Y_I, test_loader_base=None, testdata=None):
        self.text_embeddings = text_embeddings
        self.y_binary = ((Y_I + 1) / 2)[:, 1].int()
        self.X_T = text_embeddings[self.y_binary]
        self.debias_text_embedding = self.text_model.encod(self.text_embeddings)
        self.bias_text_embedding = self.text_model.get_bias_text(self.text_embeddings)
        self.bias_em_embedding = self.text_model.spurious_embeddings.float().to(self.cfg.device),
        self.mix = self.text_model.candidate_embeddings.float().to(self.cfg.device)
        self.test_loader_base = test_loader_base
        self.testdata = testdata

    def kd_loss(self, outputs, labels, teacher_outputs, mask, alpha=0.5, T=1):
        kl_loss = self.kl_loss_(outputs, teacher_outputs, T)
        ce_loss = F.cross_entropy(outputs[mask], labels[mask])
        KD_loss = kl_loss * (alpha * T * T) + ce_loss * (1. - alpha)
        return KD_loss, kl_loss, ce_loss

    def kd_loss_new(self, outputs, teacher_outputs, ori_logit, adv_logit, teacher_logit, best_group, bias_group,
                    att_bias_group, alpha=0.95, beta=0.98):
        # teacher logit is debias_logit (I @ P(T) ) or batch_y
        sub = torch.tensor(0, device=self.cfg.device).float()
        loss_adv, loss_bias, mse_loss = dc(sub), dc(sub), dc(sub)
        if torch.any(best_group):
            feat_num = outputs[best_group].size(0)
            for i in range(feat_num):
                mse_loss += F.mse_loss(outputs[best_group][i], teacher_outputs[best_group][i],
                                       reduction="none").mean()
            if self.no_label:
                if torch.any(att_bias_group):
                    loss_adv = self.kl_loss_(adv_logit[best_group][att_bias_group],
                                             teacher_logit[best_group][att_bias_group])
            else:
                if torch.any(att_bias_group):
                    loss_adv = F.cross_entropy(adv_logit[best_group][att_bias_group],
                                               teacher_logit[best_group][att_bias_group])
        # ce_loss = F.cross_entropy(outputs[bias_group], labels[bias_group])
        if torch.any(bias_group):
            if self.no_label:
                loss_bias = self.kl_loss_(ori_logit[bias_group], teacher_logit[bias_group], T)
            else:
                loss_bias = F.cross_entropy(ori_logit[bias_group], teacher_logit[bias_group])

        if torch.isnan(loss_bias).any():
            print('*' * 50, "Tensor contains NaN values.")
        KD_loss = mse_loss * (alpha) + (loss_bias * (beta) + loss_adv * (1. - beta)) * (1. - alpha)
        return KD_loss, mse_loss, loss_adv, loss_bias

    def sud_loss(self, outputs, teacher_outputs, ori_logit, y_anchor, best_group, bias_group, beta=0.8, mask=None):
        # teacher logit is debias_logit (I @ P(T) ) or batch_y
        if mask is not None:
            outputs, teacher_outputs = outputs[mask], teacher_outputs[mask]
            ori_logit, y_anchor = ori_logit[mask], y_anchor[mask]
        sub = torch.tensor(0, device=self.cfg.device).float()
        ce_loss, mse_loss = dc(sub), dc(sub)
        if torch.any(best_group):
            feat_num = outputs[best_group].size(0)
            for i in range(feat_num):
                mse_loss += F.mse_loss(outputs[best_group][i], teacher_outputs[best_group][i], reduction="none").mean()
            # mse_loss = F.cosine_similarity(outputs[best_group], teacher_outputs[best_group].detach(), dim=-1).mean()
        if torch.any(bias_group):
            ce_loss = F.cross_entropy(ori_logit[bias_group], y_anchor[bias_group])
        sud_loss = mse_loss * (beta) + ce_loss * (1. - beta)
        if torch.isnan(sud_loss).any():
            print('*' * 50, "Tensor contains NaN values.")
        return sud_loss, mse_loss * (beta), ce_loss * (1. - beta)

    def sud_loss_info(self, outputs, teacher_outputs, ori_logit, y_anchor, best_group, bias_group, beta=0.8, mask=None):
        # teacher logit is debias_logit (I @ P(T) ) or batch_y
        if mask is not None:
            outputs, teacher_outputs = outputs[mask], teacher_outputs[mask]
            ori_logit, y_anchor = ori_logit[mask], y_anchor[mask]
        sub = torch.tensor(0, device=self.cfg.device).float()
        debias_loss, retain_loss = dc(sub), dc(sub)
        if torch.any(best_group):
            feat_num = outputs[best_group].size(0)
            for i in range(feat_num):
                retain_loss += F.mse_loss(outputs[best_group][i], teacher_outputs[best_group][i],
                                          reduction="none").mean()
            # retain_loss = self.supcon_infonce(outputs[best_group], teacher_outputs[best_group])
        if torch.any(bias_group):
            # debias_loss = F.cross_entropy(ori_logit[bias_group], y_anchor[bias_group])
            debias_loss = self.supcon_infonce_bal(outputs[bias_group], teacher_outputs[bias_group],
                                                  y_anchor[bias_group], self.debias_text_embedding)
        sud_loss = retain_loss * (beta) + debias_loss * (1. - beta)
        if torch.isnan(sud_loss).any():
            print('*' * 50, "Tensor contains NaN values.")
        return sud_loss, retain_loss * (beta), debias_loss * (1. - beta)

    def kd_loss_info(self, outputs, teacher_outputs, ori_logit, adv_logit, debais_t, ori_t, best_group, bias_group,
                     att_bias_group, alpha=0.9, beta=0.5):
        # teacher logit is debias_logit (I @ P(T) ) or batch_y
        sub = torch.tensor(0, device=self.cfg.device).float()
        loss_adv, loss_bias, mse_loss = dc(sub), dc(sub), dc(sub)
        if torch.any(best_group):
            feat_num = outputs[best_group].size(0)
            for i in range(feat_num):
                mse_loss += F.mse_loss(outputs[best_group][i], teacher_outputs[best_group][i],
                                       reduction="none").mean()
            if self.no_label:
                if torch.any(att_bias_group):
                    loss_adv = self.kl_loss_(adv_logit[best_group][att_bias_group],
                                             teacher_logit[best_group][att_bias_group])
            else:
                if torch.any(att_bias_group):
                    loss_adv = self.infonce_loss(adv_logit[best_group][att_bias_group],
                                                 debais_t[best_group][att_bias_group],
                                                 ori_t[best_group][att_bias_group])
        # ce_loss = F.cross_entropy(outputs[bias_group], labels[bias_group])
        if torch.any(bias_group):
            if self.no_label:
                loss_bias = self.kl_loss_(ori_logit[bias_group], teacher_logit[bias_group], T)
            else:
                loss_bias = self.infonce_loss(ori_logit[bias_group], debais_t[bias_group], ori_t[bias_group])

        if torch.isnan(loss_bias).any():
            print('*' * 50, "Tensor contains NaN values.")
        KD_loss = mse_loss * (alpha) + (loss_bias * (beta) + loss_adv * (1. - beta)) * (1. - alpha)
        return KD_loss, mse_loss, loss_adv, loss_bias

    def kl_loss_(self, outputs, teacher_outputs, T=1):
        return nn.KLDivLoss()(F.log_softmax(outputs + 1e-10 / T, dim=1), F.softmax(teacher_outputs / T, dim=1))

    def infonce_loss(self, anchor, negative, positive, temper=0.01):
        anchor = F.normalize(anchor, dim=-1)
        if len(positive.shape) > 2:
            pos_similarity = torch.tensor(0, device=self.cfg.device).float()
            for pos_feature in positive:
                pos_feature = F.normalize(pos_feature, dim=-1)
                pos_similarity += torch.exp(torch.sum(anchor * pos_feature, dim=-1) / temper).mean()
            pos_similarity = pos_similarity / len(positive)
        else:
            positive = F.normalize(positive, dim=-1)
            pos_similarity = torch.exp(torch.sum(anchor * positive, dim=-1) / temper)

        if len(negative.shape) > 2:
            neg_similarity = torch.tensor(0, device=self.cfg.device).float()
            for neg_feature in negative:
                neg_feature = F.normalize(neg_feature, dim=-1)
                neg_similarity += torch.exp(torch.sum(anchor * neg_feature, dim=-1) / temper).mean()
            neg_similarity = neg_similarity / len(negative)
        else:
            negatives = F.normalize(negative, dim=-1)
            neg_similarity = torch.exp(torch.sum(anchor * negatives, dim=-1) / temper)

        loss = -torch.log(pos_similarity / (pos_similarity + neg_similarity)).mean()
        return loss

    def pairwise_distances(self, x):
        # x should be two dimensional
        instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
        return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()

    def GaussianKernelMatrix(self, x, sigma=1):
        pairwise_distances_ = self.pairwise_distances(x)
        return torch.exp(-pairwise_distances_ / sigma)

    def covariance(self, x, y):
        x_mean = x - x.mean(dim=0, keepdim=True)
        y_mean = y - y.mean(dim=0, keepdim=True)
        cov = (x_mean * y_mean).mean(dim=0)
        return cov

    def HSIC(self, x, y, s_x=1, s_y=1):
        m, _ = x.shape  # batch size
        K = self.GaussianKernelMatrix(x, s_x)
        L = self.GaussianKernelMatrix(y, s_y)
        H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
        H = H.float().cuda()
        HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
        return HSIC

    def supcon_infonce(self, features, feature_t, labels=None, text_feature=None, temperature=0.1):
        device = self.cfg.device

        features_cat = torch.cat([features.unsqueeze(1), feature_t.unsqueeze(1)], dim=1)
        batch_size, contrast_count, _ = features_cat.shape
        contrast_feature = torch.cat(torch.unbind(features_cat, dim=1), dim=0)
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T), temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if labels is not None:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = torch.eye(batch_size, device=device)

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.ones_like(mask)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (temperature / 0.07) * mean_log_prob_pos if labels is not None else -mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()

        return loss

    def supcon_infonce_bal(self, features, feature_t, labels=None, text_feature=None, temperature=0.1, balance=True):
        device = self.cfg.device

        features_cat = torch.cat([features.unsqueeze(1), feature_t.unsqueeze(1)], dim=1)
        batch_size, contrast_count, _ = features_cat.shape
        contrast_feature = torch.cat(torch.unbind(features_cat, dim=1), dim=0)
        if balance:
            contrast_feature_ = torch.cat([contrast_feature, text_feature], dim=0)
            anchor_dot_contrast = torch.div(contrast_feature_[:2 * batch_size].mm(contrast_feature_.T), temperature)
        else:
            anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        if balance:
            labels = labels.contiguous().view(-1, 1)
            targets_centers = torch.arange(len(self.cls_num_list), device=device).view(-1, 1)
            labels = torch.cat([labels.repeat(2, 1), targets_centers], dim=0)
            mask = torch.eq(labels[:2 * batch_size], labels.T).float().to(device)
            logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                        torch.arange(batch_size * 2).view(-1, 1).to(device), 0)
            mask = mask * logits_mask
        else:
            if labels is not None:
                labels = labels.contiguous().view(-1, 1)
                mask = torch.eq(labels, labels.T).float().to(device)
                mask = mask.repeat(contrast_count, contrast_count)
            else:
                mask = torch.eye(batch_size, device=device)
            mask = mask.repeat(contrast_count, contrast_count)
            logits_mask = torch.ones_like(mask)
            logits_mask.fill_diagonal_(0)
            mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        if balance:
            batch_cls_count = torch.eye(len(self.cls_num_list), device=device)[labels].sum(dim=0).squeeze()
            per_ins_weight = torch.tensor([batch_cls_count[i] for i in labels], device=device).view(1, -1).expand(
                2 * batch_size, 2 * batch_size + len(self.cls_num_list)) - mask
            exp_logits = exp_logits.div(per_ins_weight)

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = -mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()

        return loss

    def init_set(self):
        self.image_model.eval()
        self.student_layer = self.image_model.target_layer  # 1024*768
        print(f'unlearn target parameter : {get_parameter_count(self.student_layer)}')

    def train(self, num_iters):
        logit_scale, batch, device, check_unlearned = 100, 128, self.cfg.device, True
        student_layer = self.student_layer
        optimizer = torch.optim.SGD(student_layer.parameters(), lr=self.cfg.lr, momentum=0.9)
        teacher_layer = dc(student_layer)
        alpha = 0.995
        for epo in range(num_iters):
            ce_loss_att_track, ce_loss_bias_track, kl_loss_track, kl_loss_bias_track = 0, 0, 0, 0
            # adv = PGD(teacher_layer, 0.4, 0.5, 5, False, True, True, device, self.P)
            adv = PGD(student_layer, 0.4, 0.5, 20, False, True, True, device,
                      self.P, no_label=self.no_label, using=self.text_model)
            optimizer.zero_grad()
            teacher_layer.eval()
            for step, (batch_x, batch_y, _) in enumerate(self.train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                x = self.image_model.get_img_feature(batch_x)
                teacher_layer.eval()
                ori_feature_s = student_layer(x)

                # ori_logit_debias = logit_scale * ori_feature_t.float() @ self.debias_text_embedding.t()

                ori_logit_base = logit_scale * ori_feature_s.float() @ self.text_embeddings.t()
                ori_logit_y_base = torch.argmax(ori_logit_base, dim=-1)

                with torch.no_grad():
                    ori_feature_f = student_layer(x)
                    ori_logit_debias_f = logit_scale * ori_feature_f.float() @ self.debias_text_embedding.t()
                    ori_logit_y_debias = torch.argmax(ori_logit_debias_f, dim=-1)
                    ori_feature_t = teacher_layer(x)
                    ori_logit_debias_t = logit_scale * ori_feature_t.float() @ self.debias_text_embedding.t()
                    ori_logit_base_t = logit_scale * ori_feature_t.float() @ self.text_embeddings.t()

                if self.no_label:
                    maybe_gt = torch.argmax(ori_logit_debias_t, dim=-1)
                    already_true = torch.argmax(ori_logit_base_t, dim=-1) == maybe_gt
                    target = torch.argmax(ori_logit_base_t, dim=-1) != torch.argmax(ori_logit_debias_t, dim=-1)
                    # 1. best_group to be bias_group using attack for robust training <- bias attack
                    best_group = (ori_logit_y_debias == ori_logit_y_base) & target
                    # 2. logit of bias group to be debias logit using kl_d
                    bias_group = ori_logit_y_debias != ori_logit_y_base & target

                    # 3. logit of attacked group to make bias
                    x_adv = adv.perturb_bias_p(x, self.text_embeddings, self.debias_text_embedding, target_y=batch_y,
                                               model=student_layer, device=device, group=target, method='proj')
                    adv_feature = student_layer(x_adv)
                    adv_logit = logit_scale * adv_feature.float() @ self.text_embeddings.t()
                    adv_logit_debias = logit_scale * adv_feature.float() @ self.debias_text_embedding.t()
                    adv_logit_y = torch.argmax(adv_logit, dim=-1)
                    att_bias_group = (maybe_gt[already_true] != adv_logit_y[already_true])
                    sub = torch.tensor(0, device=self.cfg.device).float()
                    loss_adv, loss_bias, mse_loss = dc(sub), dc(sub), dc(sub)
                    ce = torch.tensor(0, device=device).float()
                    ce_loss_att, ce_loss_bias, ce_loss_base = ce.clone(), ce.clone(), ce.clone()

                    if torch.any(already_true):
                        feat_num = ori_feature_s[already_true].size(0)
                        for i in range(feat_num):
                            mse_loss += F.mse_loss(ori_feature_s[already_true][i], ori_feature_t[already_true][i],
                                                   reduction="none").mean()
                        if torch.any(att_bias_group):
                            ce_loss_att = F.cross_entropy(adv_logit[already_true][att_bias_group],
                                                          ori_logit_y_debias[already_true][att_bias_group])
                    if torch.any(bias_group):
                        # ce_loss_bias = F.cross_entropy(ori_logit_base[bias_group], ori_logit_y_debias[bias_group],label_smoothing=0.7)
                        loss_bias = F.cross_entropy(ori_logit_base[bias_group], ori_logit_y_debias[bias_group],
                                                    reduction='none')
                        pt = torch.exp(-loss_bias)
                        ce_loss_bias = ((1 - pt) ** 3 * loss_bias).mean()
                        # print(ce_loss_bias)
                    loss = (7 * ce_loss_att + 0.01 * ce_loss_bias + 15 * mse_loss) * 0.1
                    # infonce_loss = self.infonce_loss(ori_feature_s[bias_group], self.bias_em_embedding[0].unsqueeze(1).expand(-1, bias_group.sum(), -1), ori_feature_t[bias_group], temper=0.01)
                    # infonce_loss = self.infonce_loss(ori_feature_s[bias_group], self.text_embeddings, self.debias_text_embedding)
                    # print('ce loss1', ce_loss1.item(), 'ce loss2', ce_loss2.item())

                    if torch.isnan(ori_feature_s).any():
                        print('*' * 50, "Tensor contains NaN values.")
                    if not loss:
                        continue
                    ce_loss_att_track += 5 * ce_loss_att.item()
                    ce_loss_bias_track += 0.02 * ce_loss_bias.item()
                    kl_loss_track += 10 * mse_loss.item()

                else:
                    best_group = batch_y == ori_logit_y_base  # matrix un-aware bias
                    # 2. logit of bias group to be debias logit using kl_d
                    bias_group = batch_y != ori_logit_y_base  # matrix aware bias - mixed
                    true_count = torch.sum(best_group).item() - torch.sum(bias_group).item()
                    # attacked feature : bias gradient augmentation
                    x_adv = adv.perturb_bias_p(x, self.text_embeddings, self.debias_text_embedding, target_y=batch_y,
                                               model=teacher_layer, device=device, group=best_group, method='proj')
                    adv_feature = student_layer(x_adv)
                    adv_logit = logit_scale * adv_feature.float() @ self.text_embeddings.t()
                    adv_logit_debias = logit_scale * adv_feature.float() @ self.debias_text_embedding.t()
                    adv_logit_y = torch.argmax(adv_logit, dim=-1)
                    adv_logit_debias_y = torch.argmax(adv_logit_debias, dim=-1)

                    # 3. logit of attacked group to make bias
                    att_bias_group = (batch_y[best_group] != adv_logit_y[best_group])
                    att_bias_group_fail = (batch_y[best_group] == adv_logit_y[best_group])
                    # att_bias_group = (ori_logit_y_base[best_group] != adv_logit_y[best_group]) == (
                    #         ori_logit_y_debias[best_group] == adv_logit_debias_y[best_group])

                    # CE loss
                    # ce = torch.tensor(0, device=device).float()
                    # ce_loss_att, ce_loss_bias, ce_loss_base = ce.clone(), ce.clone(), ce.clone()
                    # if torch.any(att_bias_group):
                    #     ce_loss_att = F.cross_entropy(adv_logit[best_group][att_bias_group],
                    #                                   batch_y[best_group][att_bias_group])
                    #     ce_loss_base1 = F.cross_entropy(ori_logit_base[best_group][att_bias_group],
                    #                                     batch_y[best_group][att_bias_group])
                    #     ce_loss_base += ce_loss_base1
                    # if torch.any(att_bias_group_fail):
                    #     ce_loss_base2 = F.cross_entropy(ori_logit_base[best_group][att_bias_group_fail],
                    #                                     batch_y[best_group][att_bias_group_fail])
                    #     ce_loss_base += ce_loss_base2 * 0.5
                    # # ce_loss1 = F.cross_entropy(adv_logit[best_group], batch_y[best_group])
                    # if torch.any(bias_group):
                    #     ce_loss_bias = F.cross_entropy(ori_logit_base[bias_group], batch_y[bias_group])
                    # loss = (ce_loss_att + ce_loss_bias + ce_loss_base) * 0.1

                    loss, kl_loss, ce_loss_att, ce_loss_bias \
                        = self.kd_loss_new(ori_feature_s, ori_feature_t, ori_logit_base, adv_logit, batch_y,
                                           best_group, bias_group, att_bias_group)
                    # ori_txt_debias = self.debias_text_embedding[batch_y]
                    # ori_txt_ = self.text_embeddings[batch_y]
                    # loss, kl_loss, ce_loss_att, ce_loss_bias \
                    #     = self.kd_loss_info(ori_feature_s, ori_feature_t, ori_logit_base, adv_logit, batch_y,
                    #                        best_group, bias_group, att_bias_group)
                    if not loss:
                        continue
                    # print('ce loss1', ce_loss1.item(), 'ce loss2', ce_loss2.item())
                    ce_loss_att_track += ce_loss_att.item()
                    ce_loss_bias_track += ce_loss_bias.item()
                    kl_loss_track += kl_loss.item()
                loss.backward(retain_graph=True)
                optimizer.step()
                if not self.no_label:
                    for mean_param, param in zip(teacher_layer.parameters(), student_layer.parameters()):
                        mean_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)
                self.image_model.target_layer = student_layer
                student_layer.zero_grad()

            # Check that unlearning is done.
            if not epo % 3:
                for step, (batch_x, batch_y, _) in enumerate(self.val_loader):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    # debias_image = self.get_can_embedding(self.val_loader)
                    # text_embeddings_debias =self.get_textfeat(self.text_embeddings)
                    # dataset_predictions_debias_vl = self.get_zeroshot_predictions(debias_image, text_embeddings_debias, self.cfg,
                    #                                                          temperature=100.)
                    # self.get_zeroshot_predictions()
                    x = self.image_model.get_img_feature(batch_x)
                    teacher_layer = dc(student_layer)
                    teacher_layer.eval()
                    ori_feature = teacher_layer(x)
                    if torch.isnan(ori_feature).any():
                        print('*' * 50, "Tensor contains NaN values.")
                    ori_logit = torch.argmax(logit_scale * ori_feature.float() @ self.debias_text_embedding.t(), dim=-1)
                    ori_logit_base = torch.argmax(logit_scale * ori_feature.float() @ self.text_embeddings.t(), dim=-1)
                    mask = batch_y != ori_logit  # ori_logit != ori_logit_base  # == bias group
                    student_layer.zero_grad()
                    check_unlearned += mask.sum()
                    del x
                if not check_unlearned:
                    break
                un_c = int(check_unlearned)
                print(f'biased = {un_c} -- {round(un_c * 100 / self.len_t, 2)}% is biased bias_group')
                check_unlearned = 0
            print(
                f'epoch: {epo} | ce_loss_att_track : {round(ce_loss_att_track, 2)} | ce_loss_bias_track: {round(ce_loss_bias_track, 2)}'
                f' | kl_loss_track : {round(kl_loss_track, 2)} | | kl_loss_bias_track : {round(kl_loss_bias_track, 2)}')
        self.image_model.target_layer = student_layer

    def train_sud_loss(self, num_iters, model_save=False, att_mode=None, learn_mode=None):
        logit_scale, batch, device, check_unlearned = 100, 128, self.cfg.device, True
        student_layer = self.student_layer
        optimizer = torch.optim.SGD(student_layer.parameters(), lr=self.cfg.lr, momentum=0.9)
        # teacher_layer = dc(student_layer)

        for epo in range(num_iters):
            teacher_layer = dc(student_layer)
            mse_ori_, ce_ori_, mse_adv_, ce_adv_ = 0, 0, 0, 0
            # adv = PGD(teacher_layer, 0.4, 0.5, 5, False, True, True, device, self.P)
            # 0.3 0.05 5
            adv = PGD(student_layer, 0.3, 0.05, 5, False, True, True, device,
                      self.P, no_label=self.no_label, using=self.text_model, mode=att_mode)
            optimizer.zero_grad()
            teacher_layer.eval()
            for step, (batch_x, batch_y, _) in enumerate(self.train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                x = self.image_model.get_img_feature(batch_x)
                ori_feature = student_layer(x)
                ori_logit = logit_scale * ori_feature.float() @ self.text_embeddings.t()
                y_predict = torch.argmax(ori_logit, dim=-1)
                debias_logit = logit_scale * student_layer(x) @ self.debias_text_embedding.t()
                y_debias_predict = torch.argmax(debias_logit, dim=-1)

                with torch.no_grad():
                    ori_feature_t = teacher_layer(x)
                    ori_logit_t = logit_scale * ori_feature.float() @ self.text_embeddings.t()
                    y_predict_t = torch.argmax(ori_logit, dim=-1)

                y_best_group = y_debias_predict == y_predict  # (student)
                x_adv = adv.perturb_bafa(x, self.text_embeddings, self.debias_text_embedding, target_y=batch_y,
                                         model=student_layer, device=device, y_best_group=y_best_group)
                adv_feature = student_layer(x_adv)
                adv_logit = F.softmax(logit_scale * adv_feature.float() @ self.text_embeddings.t(), dim=-1)
                y_adv_predict = torch.argmax(adv_logit, dim=-1)
                adv_logit_debias = F.softmax(logit_scale * adv_feature.float() @ self.debias_text_embedding.t(), dim=-1)
                y_adv_predict_debias = torch.argmax(adv_logit_debias, dim=-1)
                with torch.no_grad():
                    adv_feature_t = teacher_layer(x_adv)
                beta = 0.9
                if self.no_label:
                    y_standard = y_predict
                    # if label not changed ==> ignore. teacher feature == student feature
                    adv_best_group = y_adv_predict_debias[y_best_group] == y_predict[y_best_group]
                    # if label changed ==> attacked. ori_y == attack_loss
                    adv_bias_group = y_adv_predict_debias[y_best_group] != y_predict[y_best_group]

                    loss_adv, mse_adv, ce_adv = self.sud_loss(adv_feature, adv_feature_t, adv_logit, y_standard,
                                                              adv_best_group, adv_bias_group, beta=0.8,
                                                              mask=y_best_group)
                    # sub = torch.tensor(0, device=self.cfg.device).float()
                    # loss_adv, mse_adv = dc(sub), dc(sub)
                    # if torch.any(adv_bias_group):
                    #     loss_adv = F.cross_entropy(adv_logit[y_best_group][adv_bias_group], y_standard[y_best_group][adv_bias_group]) * 0.05
                    ce_adv = loss_adv

                else:
                    y_standard = batch_y
                    # adv_best_group = y_adv_predict[y_best_group] == y_predict[y_best_group]  # if label not changed ==> ignore. teacher feature == student feature
                    # adv_bias_group = y_adv_predict[y_best_group] != y_predict[y_best_group]  # if label changed ==> attacked. ori_y == attack_loss
                    adv_best_group = y_adv_predict_debias == y_predict  # if label not changed ==> ignore. teacher feature == student feature
                    adv_bias_group = y_adv_predict_debias != y_predict  # if label changed ==> attacked. ori_y == attack_loss
                    # if torch.any(adv_bias_group):
                    #     adv_feature_t[adv_bias_group] = adv_feature[adv_bias_group] # ori_bias_group
                    loss_adv, mse_adv, ce_adv = self.sud_loss(adv_feature, adv_feature_t, adv_logit_debias, y_standard,
                                                              adv_best_group, adv_bias_group,
                                                              beta=beta)  # , mask=y_best_group
                    # true_ = y_predict == y_debias_predict
                    # loss_ori, mse_ori, ce_ori = self.sud_loss(adv_feature, adv_feature_t, adv_logit_debias, y_standard,
                    #                                           adv_best_group, true_, beta=0)

                # best group : KL | bias_group : CE
                ori_best_group = y_debias_predict == y_standard
                # ori_bias_group = y_debias_predict != y_standard
                # loss_ori, mse_ori, ce_ori = self.sud_loss(ori_feature, ori_feature_t, debias_logit, y_standard,
                #                                           ori_best_group, ori_bias_group, beta=beta)

                # loss_ori, mse_ori, ce_ori = self.sud_loss(ori_feature, ori_feature_t, debias_logit, y_standard,
                #                                                ori_best_group, ori_bias_group, beta=beta)
                # test
                ori_best_group = ori_best_group & (y_predict == y_debias_predict)
                ori_bias_group = ~ori_best_group
                loss_ori, mse_ori, ce_ori = self.sud_loss(ori_feature, ori_feature_t, debias_logit, y_standard,
                                                          ori_best_group, ori_bias_group, beta=beta)

                if self.no_label:
                    alpha = 0.99
                    loss = loss_ori * 100 * alpha + loss_adv * (1 - alpha)
                else:
                    alpha = 0.7  # 0.6  # 0.95(best) # 0.3 is best?
                    loss = loss_ori * alpha + loss_adv * (1 - alpha)  # 10 * alpha and (1-alpha) | best

                if torch.isnan(ori_feature).any():
                    print('*' * 50, "Tensor contains NaN values.")
                if not loss:
                    continue

                mse_ori_ += mse_ori.item()
                ce_ori_ += ce_ori.item()
                mse_adv_ += mse_adv.item()
                ce_adv_ += ce_adv.item()

                loss.backward(retain_graph=True)
                optimizer.step()
                self.image_model.target_layer = student_layer
                student_layer.zero_grad()

            # Check that unlearning is done.
            if not epo % 3:
                pre_acc, our_acc, pre_acc_de, our_acc_de, att_suc, att_suc_de = 0, 0, 0, 0, 0, 0
                for step, (batch_x, batch_y, _) in enumerate(self.val_loader):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    # debias_image = self.get_can_embedding(self.val_loader)
                    # self.get_zeroshot_predictions()
                    x = self.image_model.get_img_feature(batch_x)
                    with torch.no_grad():
                        ori = student_layer(x)
                        tea = teacher_layer(x)
                    before_y = torch.argmax(logit_scale * tea @ self.text_embeddings.t(), dim=-1)
                    after_y = torch.argmax(logit_scale * ori @ self.text_embeddings.t(), dim=-1)

                    before_y_de = torch.argmax(logit_scale * tea @ self.debias_text_embedding.t(), dim=-1)
                    after_y_de = torch.argmax(logit_scale * ori @ self.debias_text_embedding.t(), dim=-1)

                    y_best_group = before_y_de == before_y
                    x_adv = adv.perturb_bafa(x, self.text_embeddings, self.debias_text_embedding, target_y=batch_y,
                                             model=student_layer, device=device, y_best_group=y_best_group)

                    with torch.no_grad():
                        adv_ = student_layer(x_adv)
                    att_y = torch.argmax(logit_scale * adv_ @ self.text_embeddings.t(), dim=-1)
                    att_y_de = torch.argmax(logit_scale * adv_ @ self.debias_text_embedding.t(), dim=-1)

                    att_suc += (att_y == batch_y).sum()
                    pre_acc += (before_y == batch_y).sum()
                    our_acc += (after_y == batch_y).sum()
                    att_suc_de += (att_y_de == batch_y).sum()
                    pre_acc_de += (before_y_de == batch_y).sum()
                    our_acc_de += (after_y_de == batch_y).sum()
                    if torch.isnan(ori).any():
                        print('*' * 50, "Tensor contains NaN values.")
                    student_layer.zero_grad()
                    del x
                if not att_suc and not att_suc_de:
                    print(att_suc)
                    break
                print(f'val - att_suc = {att_suc} | pre_acc = {pre_acc} | our_acc = {our_acc} | '
                      f'att_suc_de = {att_suc_de} | pre_acc_de = {pre_acc_de} | our_acc_de = {our_acc_de} | ')
            if not epo % 5:
                if model_save:
                    import os
                    name = 'save_for_new_groups'
                    os.makedirs(name, exist_ok=True)
                    torch.save(student_layer.state_dict(), f'{name}/student_layer{epo}_{alpha}_{beta}_{att_mode}.pth')
                if epo >= 10:
                    self.test(False)
            print(
                f'epoch: {epo} | mse_ori : {round(mse_ori_, 2)} | ce_ori: {round(ce_ori_, 2)}'
                f' | mse_adv : {round(mse_adv_, 2)} | | ce_adv : {round(ce_adv_, 2)}')
        self.image_model.target_layer = student_layer

    def train_toy(self, num_iters):
        logit_scale, batch, device, check_unlearned = 100, 128, self.cfg.device, True
        student_layer = self.student_layer
        optimizer = torch.optim.SGD(student_layer.parameters(), lr=self.cfg.lr, momentum=0.9)
        teacher_layer = dc(student_layer)
        for epo in range(num_iters):
            mse_ori_, ce_ori_, mse_adv_, ce_adv_ = 0, 0, 0, 0
            # adv = PGD(teacher_layer, 0.4, 0.5, 5, False, True, True, device, self.P)
            # 0.3 0.05 5
            optimizer.zero_grad()
            teacher_layer.eval()
            for step, (batch_x, batch_y, _) in enumerate(self.train_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                x = self.image_model.get_img_feature(batch_x)
                ori_feature = teacher_layer(x)
                target_feature = student_layer(x)
                ori_logit = logit_scale * ori_feature.float().detach @ self.text_embeddings.t()
                debias_logit = logit_scale * target_feature @ self.debias_text_embedding.t()
                final_logit = ori_logit - debias_logit

                loss = F.cross_entropy(final_logit, batch_y)
                loss.backward(retain_graph=True)
                optimizer.step()
                self.image_model.target_layer = student_layer
                student_layer.zero_grad()

            # Check that unlearning is done.
            if not epo % 3:
                acc = 0
                for step, (batch_x, batch_y, _) in enumerate(self.val_loader):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    x = self.image_model.get_img_feature(batch_x)
                    ori_feature = teacher_layer(x)
                    target_feature = student_layer(x)
                    ori_logit = logit_scale * ori_feature.float().detach @ self.text_embeddings.t()
                    debias_logit = logit_scale * target_feature @ self.debias_text_embedding.t()
                    final_logit = ori_logit - debias_logit
                    final_y = torch.argmax(final_logit, dim=-1)
                    acc = (final_y == batch_y).sum()
                    student_layer.zero_grad()
                    del x
                print(f'val - acc = {acc}')
            if not epo % 5:
                if epo >= 10:
                    self.test(False)
            print(
                f'epoch: {epo} | loss : {round(loss.item(), 2)} ')
        self.image_model.target_layer = student_layer

    def get_feat(self, X_I, X_D=None):
        Z_D = None if X_D is None else self.text_model.encod(X_D)
        Z_I = self.image_model.encod(X_I.to(self.cfg.device))

        return Z_I, Z_D

    def get_can_embedding(self, loader_base):
        all_embeddings = []
        for batch_x, batch_y, _ in loader_base:
            all_embeddings.append(self.image_model.encod(batch_x.to(self.cfg.device)).float())
        return torch.cat(all_embeddings)

    def get_textfeat(self, X_D):
        Z_D = self.text_model.encod(X_D)
        return Z_D

    def load_student(self, epoch=10, alpha=None, beta=None, mode=None, name=None):
        if alpha is None and beta is None:
            student_layer_state_dict = torch.load(f'{name}/student_layer{epoch}.pth')
        elif mode is None:
            student_layer_state_dict = torch.load(f'{name}/student_layer{epoch}_{alpha}_{beta}.pth')
        else:
            student_layer_state_dict = torch.load(f'{name}/student_layer{epoch}_{alpha}_{beta}_{mode}.pth')
        self.student_layer.load_state_dict(student_layer_state_dict)

    def get_feature_with_attack(self, input, gt, load_epoch=None, alpha=None, beta=None, mode=None, name=None):
        if load_epoch is not None:
            if name is None:
                name = 'save_for_test'
            self.load_student(load_epoch, alpha, beta, mode, name)
        x = self.image_model.get_img_feature(input.to(self.cfg.device))
        adv = PGD(self.student_layer, 0.3, 0.5, 20, False, True, False,
                  no_label=self.no_label, device=self.cfg.device, mode=mode)
        x_adv = adv.perturb_bafa(x, self.text_embeddings, self.debias_text_embedding, target_y=gt.to(self.cfg.device),
                                 model=self.student_layer, device=self.cfg.device)
        ori_feature = self.student_layer(x)
        adv_feature = self.student_layer(x_adv)
        return ori_feature, adv_feature

    def get_feature_with_orginal_attack(self, input, gt):
        self.load_student(0, 0.5, 0.9, 'bafa_kl', 'save_for_new_group')
        x = self.image_model.get_img_feature(input.to(self.cfg.device))
        adv = PGD(self.student_layer, 0.3, 0.5, 20, False, True, True, no_label=self.no_label, device=self.cfg.device)
        x_adv = adv.perturb_bafa(x, self.text_embeddings, self.debias_text_embedding, target_y=gt.to(self.cfg.device),
                                 model=self.student_layer, device=self.cfg.device, mode='ori')
        ori_feature = self.student_layer(x)
        adv_feature = self.student_layer(x_adv)
        return ori_feature, adv_feature

    def test(self, all_acc=True):
        for imfeat_test, _, _, labels_test_s, labels_test_y_gt in self.testdata:
            debias_image = self.get_can_embedding(self.test_loader_base)
            if all_acc:
                dataset_predictions_debias_vl = (
                    self.text_model.get_zeroshot_predictions(imfeat_test, self.debias_text_embedding, self.cfg,
                                                             temperature=100.))
                dataset_predictions_origin_text = (
                    self.text_model.get_zeroshot_predictions(debias_image, self.text_embeddings, self.cfg,
                                                             temperature=100.))
                print('original vision use, acc')
                evaluate_clip(dataset_predictions_debias_vl, labels_test_y_gt, labels_test_s, verbose=True)
                print('original text acc')
                evaluate_clip(dataset_predictions_origin_text, labels_test_y_gt, labels_test_s, verbose=True)
            dataset_predictions = self.text_model.get_zeroshot_predictions(debias_image, self.debias_text_embedding,
                                                                           self.cfg, temperature=100.)
            print('result for debias text & vision set:')
            evaluate_clip(dataset_predictions, labels_test_y_gt, labels_test_s, verbose=True)


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, student_model):
        for name, param in student_model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.shadow[name]


class Use_original:
    def __init__(self, base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions,
                 cfg):
        self.base_model = base_model
        self.base_transform = base_transform
        self.get_embeddings = get_embeddings
        self.get_dataset_embeddings = get_dataset_embeddings
        self.get_zeroshot_predictions = get_zeroshot_predictions
        self.cfg = cfg
        self.P = None

    def get_embedding(self, x):
        x = self.get_embeddings(x, self.base_model, self.cfg, normalize=True, verbose=False)
        return x

    def encod(self, X):
        if self.P is not None:
            X = torch.matmul(X, self.P.T)
            out = F.normalize(X, dim=-1)
        else:
            out = X
        return out

    def get_bias_text(self, X):
        if self.Pb is not None:
            out = []
            for pb in self.Pb:
                out_ = F.normalize(torch.matmul(X, pb.T), dim=-1)
                out.append(out_)
            return torch.stack(out)

    def debias(self):
        spurious_prompt, candidate_prompt, self.S, B, text = debais_vl_s_c(self.cfg)
        self.candidate_embeddings = self.get_embedding(candidate_prompt)
        self.spurious_embeddings = self.get_embedding(spurious_prompt)
        P = debias_vl(self.spurious_embeddings, self.candidate_embeddings, self.S)
        self.Pb = bias_vl(self.spurious_embeddings, self.candidate_embeddings, B).to(self.cfg.device)
        self.P = P.to(self.cfg.device)
        return P


class AttackTarget(nn.Module):
    def __init__(self, cfg=None, model=None):
        super(AttackTarget, self).__init__()
        self.vis_mode = cfg.load_base_model.lower()
        self.image_encoder = model.visual
        self.dtype = model.dtype
        self.target_layer = None
        self.set_target()

    def set_target(self):
        if 'vit' in self.vis_mode:
            self.target_layer = Target(self.image_encoder.proj, 'vit')
            self.image_encoder.proj = None
        else:
            self.target_layer = Target(self.image_encoder.attnpool)
            self.image_encoder.attnpool.c_
            # self.image_encoder.attnpool = nn.Identity()

    def get_img_feature(self, img):
        with torch.no_grad():
            image_features = self.image_encoder(img)
        return image_features

    def encod(self, X):
        if self.target_layer is not None:
            X = self.get_img_feature(X)
            out = self.target_layer(X)

        else:
            out = X
        return out


class Target(nn.Module):
    def __init__(self, fc, name='RN'):
        super(Target, self).__init__()
        self.fc = fc
        self.name = name

    def forward(self, x):
        output = x @ self.fc.to(x.dtype) if 'vit' in self.name else self.fc(x)
        output = output / output.norm(dim=-1, keepdim=True)
        return output


import os
import clip
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10, ImageNet
from tqdm import tqdm


class Cifar_test:
    def __init__(self, text_model, image_model, cfg, transform=None, mode='base'):
        self.text_model = text_model
        self.image_model = image_model
        self.cfg = cfg
        self.transform = transform
        self.mode = mode

    def zeroshot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates]  # Format with class
                class_embeddings = self.uo.get_embedding(texts)  # Embed with text encoder
                # if not self.mode == 'fairer':
                #     class_embeddings = self.text_model.encod(class_embeddings.to(self.cfg.device)).to(self.cfg.device)
                class_embeddings /= (class_embeddings.norm(dim=-1, keepdim=True) + 1e-10)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= (class_embedding.norm() + 1e-10)
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.cfg.device)
        return zeroshot_weights

    def set_info(self, uo, model=None):
        self.uo = uo
        self.img_model = dc(uo.base_model)
        self.img_model.eval()
        self.img_model = self.img_model.float().to(self.cfg.device)
        self.fairer = dc(model)

    def accuracy(self, output, target, topk=(1,)):
        pred = output.topk(max(topk), 1, True, True)[1].t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # (5,128) or (1,128)
        return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

    def zero_shot_evaluation(self, dataset, class_names, templates):
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        zeroshot_weights = self.zeroshot_classifier(class_names, templates)
        if self.mode == 'fairer':
            fairer_weights = self.fairer.get_textfeat(zeroshot_weights.t())

        top1, top5, n = 0., 0., 0.

        with torch.no_grad():

            for images, target in tqdm(loader):
                images = images.to(self.cfg.device)
                target = target.to(self.cfg.device)

                self.img_model = self.img_model.to(self.cfg.device)
                image_features = self.img_model.encode_image(images)

                if self.mode == 'fairer':
                    image_features, _ = self.fairer.get_feat(image_features.float(), zeroshot_weights.t()[target])
                    logits = 100. * image_features @ fairer_weights.T
                    acc1, acc5 = self.accuracy(logits, target, topk=(1, 5))
                    top1 += acc1
                    top5 += acc5
                    n += images.size(0)
                else:
                    # Predict
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    logits = 100. * image_features.float() @ zeroshot_weights
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
        root10, root100, rootimgnet = map(os.path.expanduser, ["./datasets/cifar10", "./datasets/cifar100","./datasets/imagenet"])
        self.transform = self.transform or self.uo.base_transform
        download10, download100 = map(lambda p: not os.path.exists(p) and os.makedirs(p, exist_ok=True) is None,
                                      [root10, root100])
        test_cifar100 = CIFAR100(root100, download=download100, train=False, transform=self.transform)
        test_cifar10 = CIFAR10(root10, download=download10, train=False, transform=self.transform)

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


        # # ImageNet
        imagenet_templates = [
            'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
            'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
            'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
            'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
            'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
            'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.',
            'a painting of a {}.', 'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.',
            'a cropped photo of a {}.', 'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.',
            'a blurry photo of the {}.', 'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.',
            'a {} in a video game.', 'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.',
            'a photo of a {}.', 'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.',
            'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.',
            'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
            'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.', 'a sketch of the {}.',
            'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.', 'a jpeg corrupted photo of the {}.',
            'a good photo of a {}.',  'a plushie {}.',  'a photo of the nice {}.', 'a photo of the small {}.',
            'a photo of the weird {}.',  'the cartoon {}.', 'art of the {}.', 'a drawing of the {}.',
            'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.', 'a dark photo of a {}.',
            'itap of a {}.', 'graffiti of the {}.', 'a toy {}.', 'itap of my {}.', 'a photo of a cool {}.',
            'a photo of a small {}.', 'a tattoo of the {}.', ]
        from imagenetv2_pytorch import ImageNetV2Dataset
        imagenet = ImageNetV2Dataset(transform=self.transform)
        # loader = torch.utils.data.DataLoader(images, batch_size=32, num_workers=2)
        # test_cifar10 = CIFAR10(root10, download=download10, train=False, transform=self.transform)
        imagenet_top1, imagenet_top5 = self.zero_shot_evaluation(imagenet, imagenet.classes, imagenet_templates)
        print(f"Imagenet Zero-shot CLIP model Top-1 accuracy: {imagenet_top1:.2f}%")
        print(f"Imagenet Zero-shot CLIP model Top-5 accuracy: {imagenet_top5:.2f}%")