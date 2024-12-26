from copy import deepcopy as dc

import torch.nn.functional as F
from torch import nn

from can.attack.adversarial_attack_backup_0722_BAFA import PGD
from can.attack.debias_vl import debais_vl_s_c, debias_vl, bias_vl
from can.utils.tools import get_parameter_count
from network.clip import evaluate_clip


class Unlearn:
    def __init__(self, opts, uo, train_loader_base, val_loader_base, get_zeroshot_predictions, no_label=False,
                 learn_mode=None):
        self.mode = opts.mode
        self.cfg = opts
        self.uo = dc(uo)
        self.use_feature_augmentation = True
        if learn_mode == 'CAN':
            model = AttackTarget(opts, uo.base_model.float().to(opts.device), mode='txt')
            self.text_model = dc(model)
            # model = AttackTarget(opts, uo.base_model.float().to(opts.device), mode=None)
            self.image_model = Base_clip(uo.base_model)
        elif learn_mode == 'BCAU':
            model = AttackTarget(opts, uo.base_model.float().to(opts.device), mode='txt')
            self.text_model = dc(model)
            model = AttackTarget(opts, uo.base_model.float().to(opts.device), mode='img')
            self.image_model = dc(model)
        elif learn_mode is not None:
            self.P = uo.debias()
            model = AttackTarget(opts, uo.base_model.float().to(opts.device))
            self.text_model = dc(uo)
            self.image_model = dc(model)

        self.train_loader = train_loader_base
        self.val_loader = val_loader_base
        self.len_t = len(val_loader_base.dataset)

        self.get_zeroshot_predictions = get_zeroshot_predictions
        if learn_mode == 'CAN':
            self.init_set('txt')
        elif learn_mode == 'BCAU':
            self.init_set('both')
        else:
            self.init_set()
        self.no_label = no_label
        self.bias_sample_check = True
        self.cls_num_list = [0, 1]
        self.data = opts.embeddings_dir.split('/')[-1]

        self.sub = torch.tensor(0, device=self.cfg.device).float()
    def rebuilding(self):
        import torch
        from transformers import CLIPProcessor, CLIPModel

        model_url = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
        model_path = "ViT-L-14.pt"
        torch.hub.download_url_to_file(model_url, model_path)

        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        from peft import LoraConfig
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

    def info_img(self, text_embeddings, Y_I, test_loader_base=None, testdata=None, mode=None):
        self.text_embeddings = text_embeddings
        self.y_binary = ((Y_I + 1) / 2)[:, 1].int()
        self.X_T = text_embeddings[self.y_binary]
        if not mode == 'BCAU':
            self.target_image_set, self.y = self.get_can_embedding(self.train_loader, with_label=True, get_feature=True)
            self.val_image_set, self.val_y = self.get_can_embedding(self.val_loader, with_label=True, get_feature=True)
            self.y, self.val_y = self.y.to(self.cfg.device), self.val_y.to(self.cfg.device)
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
            outputs, teacher_outputs = outputs[mask], teacher_outputs[mask] if len(
                outputs) > 10 else outputs, teacher_outputs
            ori_logit, y_anchor = ori_logit[mask], y_anchor[mask]
        sub = torch.tensor(0, device=self.cfg.device).float()
        ce_loss, mse_loss = dc(sub), dc(sub)
        if beta:
            if torch.any(best_group):
                feat_num = outputs[best_group].size(0)
                for i in range(feat_num):
                    mse_loss += F.mse_loss(outputs[best_group][i], teacher_outputs[best_group][i],
                                           reduction="none").mean()
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

    def sud_loss_txt(self, outputs, teacher_outputs, ori_logit, y_anchor, best_group, bias_group, beta=0.8):
        sub = torch.tensor(0, device=self.cfg.device).float()
        ce_loss, mse_loss = dc(sub), dc(sub)
        if beta:
            if torch.any(best_group):
                feat_num = outputs.size(0)
                for i in range(feat_num):
                    mse_loss += F.mse_loss(outputs[i], teacher_outputs[i], reduction="none").mean()
        if torch.any(bias_group):
            ce_loss = F.cross_entropy(ori_logit[bias_group], y_anchor[bias_group])
        sud_loss = mse_loss * (beta) + ce_loss * (1. - beta)
        if torch.isnan(sud_loss).any():
            print('*' * 50, "Tensor contains NaN values.")
        return sud_loss, mse_loss * (beta), ce_loss * (1. - beta)

    def kl_loss_(self, outputs, teacher_outputs, T=1):
        return nn.KLDivLoss()(F.log_softmax(outputs + 1e-10 / T, dim=1), F.softmax(teacher_outputs / T, dim=1))

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

    def init_set(self, target='img'):
        if target == 'img':
            self.image_model.eval()
            self.student_layer = self.image_model.target_layer  # 1024*768
            print(f'unlearn target parameter : {get_parameter_count(self.student_layer)}')
        elif target == 'txt':
            self.text_model.eval()
            self.student_layer = self.text_model.target_layer  # 768*768
            print(f'unlearn target parameter : {get_parameter_count(self.student_layer)}')
        elif target == 'both':
            self.text_model.eval()
            self.image_model.eval()
            self.student_layer_txt = self.text_model.target_layer  # 768*768
            self.student_layer_img = self.image_model.target_layer
            print(
                f'unlearn target parameter : {get_parameter_count(self.student_layer_txt) + get_parameter_count(self.student_layer_img)}')

    def train_txt_sud_loss(self, num_iters, model_save=False, att_mode=None, learn_mode=None, pre_embedding=None):
        logit_scale, batch, device, check_unlearned, self.pre_embedding = 100, 128, self.cfg.device, True, pre_embedding
        student_layer = self.student_layer
        optimizer = torch.optim.SGD(student_layer.parameters(), lr=self.cfg.lr, momentum=0.9)
        # teacher_layer = dc(student_layer)
        bound, step, iters, num_sample = 0.5, 1e-2, 5, 10

        batch_y = self.y
        cases = ['wrong_best', 'wrong']
        case = cases[1]
        target_image_set_expanded = self.target_image_set.unsqueeze(0).expand(num_sample, -1, -1)
        for epo in range(num_iters):
            # original
            teacher_layer = dc(student_layer)
            mse_ori_, ce_ori_, mse_adv_, ce_adv_ = 0, 0, 0, 0
            # adv = PGD(teacher_layer, 0.4, 0.5, 5, False, True, True, device, self.P)
            adv = PGD(teacher_layer, bound, step, iters, False, True, True, device,
                      no_label=self.no_label, using=self.text_model, mode=att_mode, learn_mode=learn_mode)
            optimizer.zero_grad()
            teacher_layer.eval()

            text_embeddings = student_layer(pre_embedding)
            ori_logit = logit_scale * self.target_image_set @ text_embeddings.t()

            y_predict = torch.argmax(ori_logit, dim=-1)
            with torch.no_grad():
                text_embeddings_t = teacher_layer(pre_embedding)

            ori_best_group = y_predict == batch_y  # y_predict
            ori_bias_group = y_predict != batch_y
            pre_embedd_adv = adv.perturb_txt(pre_embedding, self.target_image_set, target_y=self.y, model=student_layer,
                                             device=device, num_sample=num_sample)
            text_embeddings_adv = student_layer(pre_embedd_adv)

            c, dim = text_embeddings.shape
            adv_f = text_embeddings_adv.view(-1, c, dim)
            results, len_ = [], len(adv_f)

            adv_logit_ = logit_scale * target_image_set_expanded @ adv_f.transpose(1, 2)

            total_adv_logit = adv_logit_.view(-1, 2)
            y_adv_predict = torch.argmax(total_adv_logit, dim=-1)
            expanded_binary_y = batch_y.repeat(len_)

            with torch.no_grad():
                text_embeddings_adv_t = teacher_layer(pre_embedd_adv)
            if case == 'wrong_best':  # 1. CE : GT == y_predict =/= y_adv
                adv_bias_group = (y_adv_predict == expanded_binary_y) & ori_best_group.repeat(len_)  # original
            else:  # 2. CE : GT =/= y_adv
                adv_bias_group = (y_adv_predict == expanded_binary_y)  # & ori_best_group # left > if wrong
            adv_best_group = ~adv_bias_group
            loss_adv, mse_adv, ce_adv = self.sud_loss(text_embeddings_adv, text_embeddings_adv_t, total_adv_logit,
                                                      expanded_binary_y, adv_best_group, adv_bias_group, beta=0)

            loss_ori, mse_ori, ce_ori = self.sud_loss_txt(text_embeddings, text_embeddings_t, ori_logit, batch_y,
                                                          ori_best_group, ori_bias_group, beta=0.9)

            if self.no_label:
                alpha = 0.99
                loss = loss_ori * 100 * alpha + loss_adv * (1 - alpha)
            else:
                alpha = 0.3  # 0.6  # 0.95(best) # 0.3 is best?
                loss = loss_ori * alpha + loss_adv * (1 - alpha)  # 10 * alpha and (1-alpha) | best

            if torch.isnan(ori_logit).any():
                print('*' * 50, "Tensor contains NaN values.")
            mse_ori_ += mse_ori.item()
            ce_ori_ += ce_ori.item()
            mse_adv_ += mse_adv.item()
            ce_adv_ += ce_adv.item()

            loss.backward(retain_graph=True)
            optimizer.step()
            self.text_model.target_layer = student_layer
            student_layer.zero_grad()
            if not epo % 3:
                pre_acc, our_acc, pre_acc_de, our_acc_de, att_suc, att_suc_de = 0, 0, 0, 0, 0, 0

                new_text_embeddings = student_layer(pre_embedding)
                before_logit = logit_scale * self.val_image_set @ self.text_embeddings.t()
                after_logit = logit_scale * self.val_image_set @ new_text_embeddings.t()
                before_predict = torch.argmax(before_logit, dim=-1)
                after_predict = torch.argmax(after_logit, dim=-1)

                pre_embedd_adv = adv.perturb_txt(pre_embedding, self.target_image_set, target_y=self.y,
                                                 model=student_layer, device=device)
                new_text_embeddings_adv = student_layer(pre_embedd_adv)
                att_y = torch.argmax(logit_scale * self.val_image_set @ new_text_embeddings_adv.t(), dim=-1)

                att_suc += (att_y != after_predict).sum()
                pre_acc += (before_predict == self.val_y).sum()
                our_acc += (after_predict == self.val_y).sum()
                if torch.isnan(new_text_embeddings).any():
                    print('*' * 50, "Tensor contains NaN values.")
                student_layer.zero_grad()
                del new_text_embeddings
                print(f'val - att_suc = {att_suc} | pre_acc = {pre_acc} | our_acc = {our_acc} | ')
            if not att_suc:
                break
            if not epo % 5:
                if model_save:
                    import os
                    name = f'txt_{self.data}'
                    os.makedirs(name, exist_ok=True)
                    torch.save(student_layer.state_dict(),
                               f'{name}/student_layer{epo}_{alpha}_{bound}_{step}_{iters}_{learn_mode}.pth')
                if epo >= 4:
                    self.test(False, learn_mode='CAN')
            print(
                f'epoch: {epo} | mse_ori : {round(mse_ori_, 2)} | ce_ori: {round(ce_ori_, 2)}'
                f' | mse_adv : {round(mse_adv_, 2)} | | ce_adv : {round(ce_adv_, 2)}')
        self.text_model.target_layer = student_layer

    def train_sud_loss_txt_tuning(self, num_iters, model_save=False, att_mode=None, learn_mode=None,
                                  pre_embedding=None):
        logit_scale, batch, device, check_unlearned, self.pre_embedding = 100, 128, self.cfg.device, True, pre_embedding
        student_layer_img = self.student_layer_img
        student_layer_txt = self.student_layer_txt
        optimizer = torch.optim.SGD(student_layer_txt.parameters(), lr=self.cfg.lr, momentum=0.9)
        # teacher_layer = dc(student_layer)
        bound, step, iters = 0.4, 1e-2, 5

        case = ['wrong_best', 'wrong']
        case = case[1]
        # teacher_layer = dc(student_layer)
        adv = PGD(student_layer_img, bound, step, iters, False, True, True, device,
                  no_label=self.no_label, using=self.text_model, mode=att_mode, learn_mode=learn_mode)
        with torch.no_grad():
            ori_target = student_layer_img(self.target_image_set)
            val_image_set = student_layer_img(self.val_image_set)

        for epo in range(num_iters):
            # original
            teacher_layer = dc(student_layer_txt)
            mse_ori_, ce_ori_, mse_adv_, ce_adv_ = 0, 0, 0, 0
            # adv = PGD(teacher_layer, 0.4, 0.5, 5, False, True, True, device, self.P)
            optimizer.zero_grad()
            teacher_layer.eval()
            text_embeddings = student_layer_txt(pre_embedding)

            ori_logit = logit_scale * ori_target @ text_embeddings.t()
            y_predict = torch.argmax(ori_logit, dim=-1)
            with torch.no_grad():
                text_embeddings_t = teacher_layer(pre_embedding)

            ori_best_group = y_predict == self.y  # y_predict
            ori_bias_group = y_predict != self.y
            x_adv = adv.perturb_img(self.target_image_set, text_embeddings, target_y=self.y,
                                    model=student_layer_img, device=device)
            with torch.no_grad():
                adv_feature = student_layer_img(x_adv)
            ori_logit_adv = logit_scale * adv_feature @ text_embeddings.t()
            y_adv_predict = torch.argmax(ori_logit_adv, dim=-1)
            if case == 'wrong_best':  # 1. CE : GT == y_predict =/= y_adv
                adv_bias_group = (y_adv_predict != y_predict) & ori_best_group  # original
            else:  # 2. CE : GT =/= y_adv
                adv_bias_group = (y_adv_predict != y_predict)
            adv_best_group = ~adv_bias_group

            loss_adv, mse_adv, ce_adv = self.sud_loss(text_embeddings, text_embeddings, ori_logit_adv,
                                                      self.y, adv_best_group, adv_bias_group, beta=0)

            loss_ori, mse_ori, ce_ori = self.sud_loss_txt(text_embeddings, text_embeddings_t, ori_logit, self.y,
                                                          ori_best_group, ori_bias_group, beta=0.9)
            if self.no_label:
                alpha = 0.99
                loss = loss_ori * 100 * alpha + loss_adv * (1 - alpha)
            else:
                alpha = 0.3  # 0.6  # 0.95(best) # 0.3 is best?
                loss = loss_ori * alpha + loss_adv * (1 - alpha)  # 10 * alpha and (1-alpha) | best

            if torch.isnan(ori_logit).any():
                print('*' * 50, "Tensor contains NaN values.")
            mse_ori_ += mse_ori.item()
            ce_ori_ += ce_ori.item()
            mse_adv_ += mse_adv.item()
            ce_adv_ += ce_adv.item()

            loss.backward(retain_graph=True)
            optimizer.step()
            self.text_model.target_layer = student_layer_txt
            student_layer_txt.zero_grad()
            if not epo % 3:
                pre_acc, our_acc, pre_acc_de, our_acc_de, att_suc, att_suc_de = 0, 0, 0, 0, 0, 0
                new_text_embeddings = student_layer_txt(pre_embedding)
                before_logit = logit_scale * val_image_set @ self.text_embeddings.t()
                after_logit = logit_scale * val_image_set @ new_text_embeddings.t()
                before_predict = torch.argmax(before_logit, dim=-1)
                after_predict = torch.argmax(after_logit, dim=-1)

                x_adv = adv.perturb_img(self.val_image_set, text_embeddings, target_y=self.val_y,
                                        model=student_layer_img, device=device)
                new_img_adv = student_layer_img(x_adv)
                att_y = torch.argmax(logit_scale * new_img_adv @ new_text_embeddings.t(), dim=-1)

                att_suc += (att_y != after_predict).sum()
                pre_acc += (before_predict == self.val_y).sum()
                our_acc += (after_predict == self.val_y).sum()
                if torch.isnan(new_text_embeddings).any():
                    print('*' * 50, "Tensor contains NaN values.")
                student_layer_txt.zero_grad()
                del new_text_embeddings
                print(f'val - att_suc = {att_suc} | pre_acc = {pre_acc} | our_acc = {our_acc} | ')
            if not att_suc:
                break
            if not epo % 5:
                if model_save:
                    import os
                    name = f'txt_{self.data}'
                    os.makedirs(name, exist_ok=True)
                    torch.save(student_layer_txt.state_dict(),
                               f'{name}/student_layer{epo}_{alpha}_{bound}_{step}_{iters}_{learn_mode}.pth')
                if epo >= 4:
                    self.test(False, learn_mode='CAN')
            print(
                f'epoch: {epo} | mse_ori : {round(mse_ori_, 2)} | ce_ori: {round(ce_ori_, 2)}'
                f' | mse_adv : {round(mse_adv_, 2)} | | ce_adv : {round(ce_adv_, 2)}')
        self.text_model.target_layer = student_layer_txt
    def train_sud_loss_both_tuning(self, num_iters, model_save=False, att_mode=None, learn_mode=None,
                                   pre_embedding=None):
        logit_scale, batch, device, check_unlearned, self.pre_embedding = 100, 128, self.cfg.device, True, pre_embedding
        student_layer_img, student_layer_txt = self.student_layer_img, self.student_layer_txt
        # optimizer_img = torch.optim.SGD(student_layer_img.parameters(), lr=self.cfg.lr, momentum=0.9)
        optimizer_img = torch.optim.SGD(student_layer_img.parameters(), lr=self.cfg.lr, momentum=0.9)
        optimizer_txt = torch.optim.SGD(student_layer_txt.parameters(), lr=self.cfg.lr, momentum=0.9)
        # teacher_layer = dc(student_layer)
        bound, step, iters, alpha, beta, num_sample = 0.4, 1e-2, 5, 0.6, 0.9, 1

        # teacher_layer = dc(student_layer)
        adv = PGD(student_layer_img, bound, step, iters, False, True, True, device,
                  no_label=self.no_label, using=self.text_model, mode=att_mode, learn_mode=learn_mode)

        img_epoch, text_epoch= 5, 5
        # All right -> KD
        # All wrong -> Img, txt Tuning
        # All right & Txt PGD Wrong -> Text tuning
        # All right & Img PGD Wrong -> Img tuning
        step_type = 'img'
        for epo in range(num_iters):
            teacher_layer_txt, teacher_layer_img = dc(student_layer_txt), dc(student_layer_img)
            mse_ori_, ce_ori_, ce_adv_img_, ce_adv_txt_, succ_img, succ_txt = 0, 0, 0, 0, 0, 0
            optimizer_img.zero_grad(), optimizer_txt.zero_grad()
            teacher_layer_txt.eval(), teacher_layer_img.eval()
            with torch.no_grad():
                text_embeddings_t = teacher_layer_txt(pre_embedding)
            for i in range(2):
                if step_type == 'txt':
                    for _, (batch_x, batch_y, _) in enumerate(self.train_loader):
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        ce_loss, mse_loss, ce_adv_img, ce_adv_txt = dc(self.sub), dc(self.sub), dc(self.sub), dc(self.sub)
                        text_embeddings = student_layer_txt(pre_embedding)
                        with torch.no_grad():
                            img_embeddings = student_layer_img(img_feature)
                        ori_logit = logit_scale * img_embeddings @ text_embeddings.t()
                        y_predict = torch.argmax(ori_logit, dim=-1)

                        best_group = batch_y == y_predict
                        wrong_group = ~best_group
                        if torch.any(wrong_group):
                            ce_loss = F.cross_entropy(ori_logit[wrong_group], batch_y[wrong_group])
                        for i in range(text_embeddings.size(0)):
                            mse_loss += F.mse_loss(text_embeddings, text_embeddings_t, reduction="none").mean()
                        if torch.any(best_group):
                            target_feature, target_img, target_y = (img_feature[best_group], img_embeddings[best_group],
                                                                    batch_y[best_group])
                            c, dim = text_embeddings.shape
                            pre_embedd_adv = adv.perturb_txt(pre_embedding, target_img, bound=0.01, target_y=target_y,
                                                             model=student_layer_txt, device=device, num_sample=num_sample)
                            text_embeddings_adv = student_layer_txt(pre_embedd_adv)
                            expand_target_img = target_img.detach().unsqueeze(0).expand(num_sample, -1, -1)
                            adv_f = text_embeddings_adv.view(-1, c, dim)
                            results, len_ = [], len(adv_f)

                            adv_logit_ = logit_scale * expand_target_img @ adv_f.transpose(1, 2)
                            total_adv_logit = adv_logit_.view(-1, 2)
                            y_adv_predict = torch.argmax(total_adv_logit, dim=-1)
                            expanded_binary_y = target_y.repeat(len_)
                            att_succ_txt = y_adv_predict != expanded_binary_y
                            if torch.any(att_succ_txt):
                                succ_txt += att_succ_txt.sum().item() // num_sample
                                ce_adv_txt = F.cross_entropy(total_adv_logit[att_succ_txt],
                                                             expanded_binary_y[att_succ_txt]) / num_sample
                        loss = beta * mse_loss + (1 - beta) * (alpha * ce_loss + (1 - alpha) * ce_adv_txt)
                        loss.backward(retain_graph=True)
                        optimizer_txt.step()
                        self.text_model.target_layer = student_layer_txt
                        student_layer_txt.zero_grad()
                        ce_adv_txt_ += ce_adv_txt.item()
                        ce_ori_ += ce_loss.item()
                    step_type = 'txt'
                else:
                    for _, (batch_x, batch_y, _) in enumerate(self.train_loader):
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        ce_loss, mse_loss, ce_adv_img, ce_adv_txt = dc(self.sub), dc(self.sub), dc(self.sub), dc(
                            self.sub)
                        with torch.no_grad():
                            img_feature = self.image_model.get_img_feature(batch_x)
                            img_embeddings_t = teacher_layer_img(img_feature)
                        text_embeddings = student_layer_txt(pre_embedding)
                        img_embeddings = student_layer_img(img_feature)

                        ori_logit = logit_scale * img_embeddings @ text_embeddings.t()
                        y_predict = torch.argmax(ori_logit, dim=-1)

                        best_group = batch_y == y_predict
                        wrong_group = ~best_group
                        if torch.any(wrong_group):
                            ce_loss = F.cross_entropy(ori_logit[wrong_group], batch_y[wrong_group])
                        if torch.any(best_group):
                            target_feature, target_img, target_y = (img_feature[best_group], img_embeddings[best_group],
                                                                    batch_y[best_group])
                            for i in range(target_img.size(0)):
                                mse_loss += F.mse_loss(target_img[i], img_embeddings_t[best_group][i],
                                                       reduction="none").mean()
                            x_adv = adv.perturb_img(target_feature, text_embeddings.detach(), target_y=target_y,
                                                    model=student_layer_img, device=device)
                            adv_feature = student_layer_img(x_adv)
                            adv_logit = F.softmax(logit_scale * adv_feature.float() @ text_embeddings.detach().t(), dim=-1)
                            y_adv_predict = torch.argmax(adv_logit, dim=-1)
                            att_succ_img = y_adv_predict != target_y
                            if torch.any(att_succ_img):
                                succ_img += att_succ_img.sum().item()
                                ce_adv_img = F.cross_entropy(adv_logit[att_succ_img], target_y[att_succ_img])
                            loss = beta * mse_loss + (1 - beta) * (alpha * ce_loss + (1 - alpha) * ce_adv_img)
                            loss.backward(retain_graph=True)

                            optimizer_img.step()
                            self.image_model.target_layer = student_layer_img
                            student_layer_img.zero_grad()
                            ce_ori_ += ce_loss.item()
                            ce_adv_img_ += ce_adv_img.item()
                            mse_ori_ += mse_loss.item()
                    step_type = 'txt'
            # if not epo % 3:
            #     pre_acc, our_acc, pre_acc_de, our_acc_de, att_suc, att_suc_de = 0, 0, 0, 0, 0, 0
            #     new_text_embeddings = student_layer_txt(pre_embedding)
            #     with torch.no_grad():
            #         val_image_set = student_layer_img(self.val_image_set)
            #     before_logit = logit_scale * val_image_set @ self.text_embeddings.t()
            #     after_logit = logit_scale * val_image_set @ new_text_embeddings.t()
            #     before_predict = torch.argmax(before_logit, dim=-1)
            #     after_predict = torch.argmax(after_logit, dim=-1)
            #
            #     x_adv = adv.perturb_img(self.val_image_set, text_embeddings, target_y=self.val_y,
            #                             model=student_layer_img, device=device)
            #     new_img_adv = student_layer_img(x_adv)
            #     att_y = torch.argmax(logit_scale * new_img_adv @ new_text_embeddings.t(), dim=-1)
            #
            #     att_suc += (att_y != after_predict).sum()
            #     pre_acc += (before_predict == self.val_y).sum()
            #     our_acc += (after_predict == self.val_y).sum()
            #     if torch.isnan(new_text_embeddings).any():
            #         print('*' * 50, "Tensor contains NaN values.")
            #     student_layer_txt.zero_grad()
            #     student_layer_img.zero_grad()
            #     del new_text_embeddings
            #     print(f'val - att_suc = {att_suc} | pre_acc = {pre_acc} | our_acc = {our_acc} | ')
            # if not att_suc:
            #     break
            if not epo % 5:
                if model_save:
                    import os
                    name = f'both__{self.data}'
                    os.makedirs(name, exist_ok=True)
                    torch.save(student_layer_txt.state_dict(),
                               f'{name}/student_layer{epo}_{bound}_{step}_{alpha}_{beta}_{iters}_{learn_mode}_txt.pth')
                    torch.save(student_layer_img.state_dict(),
                               f'{name}/student_layer{epo}_{bound}_{step}_{alpha}_{beta}_{iters}_{learn_mode}_img.pth')
                if epo >= 4:
                    self.test(False, learn_mode='BCAU')
            print(
                f'epoch: {epo} | mse_ori : {round(mse_ori_, 2)} | ce_ori: {round(ce_ori_, 2)}'
                f' | ce_adv_txt : {round(ce_adv_txt_, 2)} | | ce_adv_img : {round(ce_adv_img_, 2)}'
                f' | txt_att_scu : {round(succ_txt, 2)}| | img_att_scu : {round(succ_img, 2)}')
        self.text_model.target_layer = student_layer_txt
        self.image_model.target_layer = student_layer_img

    def txt_tuning(self, pre_embedding, ori_target, student_layer_img, student_layer, teacher_layer, adv, optimizer,
                   device, case='wrong', logit_scale=100):
        text_embeddings = student_layer(pre_embedding)

        ori_logit = logit_scale * ori_target @ text_embeddings.t()
        y_predict = torch.argmax(ori_logit, dim=-1)
        with torch.no_grad():
            text_embeddings_t = teacher_layer(pre_embedding)

        ori_best_group = y_predict == self.y  # y_predict
        ori_bias_group = y_predict != self.y
        x_adv = adv.perturb_img(self.target_image_set, text_embeddings, target_y=self.y,
                                model=student_layer_img, device=device)
        with torch.no_grad():
            adv_feature = student_layer_img(x_adv)
        ori_logit_adv = logit_scale * adv_feature @ text_embeddings.t()
        y_adv_predict = torch.argmax(ori_logit_adv, dim=-1)
        if case == 'wrong_best':  # 1. CE : GT == y_predict =/= y_adv
            adv_bias_group = (y_adv_predict != y_predict) & ori_best_group  # original
        else:  # 2. CE : GT =/= y_adv
            adv_bias_group = (y_adv_predict != y_predict)
        adv_best_group = ~adv_bias_group
        sub = torch.tensor(0, device=self.cfg.device).float()
        mse_adv, ce_adv = dc(sub), dc(sub)
        loss_adv, mse_adv, ce_adv = self.sud_loss(text_embeddings, text_embeddings, ori_logit_adv,
                                                  self.y, adv_best_group, adv_bias_group, beta=0)

        loss_ori, mse_ori, ce_ori = self.sud_loss_txt(text_embeddings, text_embeddings_t, ori_logit, self.y,
                                                      ori_best_group, ori_bias_group, beta=0.95)

        alpha = 0.3  # 0.6  # 0.95(best) # 0.3 is best?
        loss = loss_ori * alpha + loss_adv * (1 - alpha)  # 10 * alpha and (1-alpha) | best

        if torch.isnan(ori_logit).any():
            print('*' * 50, "Tensor contains NaN values.")
        loss.backward(retain_graph=True)
        optimizer.step()
        self.text_model.target_layer = student_layer
        student_layer.zero_grad()

        return student_layer, mse_adv, ce_adv, mse_ori, ce_ori

    def img_tuning(self, student_layer, teacher_layer, adv, optimizer, device, text_em=None, case='wrong',
                   logit_scale=100):
        if text_em is None:
            text_em = self.text_embeddings
        for _, (batch_x, batch_y, _) in enumerate(self.train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            x = self.image_model.get_img_feature(batch_x)
            ori_feature = student_layer(x)
            ori_logit = logit_scale * ori_feature.float() @ text_em.t()
            y_predict = torch.argmax(ori_logit, dim=-1)
            with torch.no_grad():
                ori_feature_t = teacher_layer(x)
            x_adv = adv.perturb_bafa(x, text_em, target_y=batch_y, model=student_layer, device=device)
            adv_feature = student_layer(x_adv)
            adv_logit = F.softmax(logit_scale * adv_feature.float() @ text_em.t(), dim=-1)
            y_adv_predict = torch.argmax(adv_logit, dim=-1)
            with torch.no_grad():
                adv_feature_t = teacher_layer(x_adv)

            ori_best_group = y_predict == batch_y  # y_predict
            ori_bias_group = y_predict != batch_y  # y_predict or y_debias_predict != y_standard

            if case == 'wrong_best':  # 1. CE : GT == y_predict =/= y_adv
                adv_bias_group = (y_adv_predict != y_predict) & ori_best_group  # original
            else:  # 2. CE : GT =/= y_adv
                adv_bias_group = (y_adv_predict != y_predict)  # & ori_best_group # left > if wrong
            adv_best_group = ~adv_bias_group
            loss_adv, mse_adv, ce_adv = self.sud_loss(adv_feature, adv_feature_t, adv_logit, batch_y,
                                                      adv_best_group, adv_bias_group, beta=0)
            loss_ori, mse_ori, ce_ori = self.sud_loss(ori_feature, ori_feature_t, ori_logit, batch_y,
                                                      ori_best_group, ori_bias_group, beta=0.9)

            if self.no_label:
                alpha = 0.99
                loss = loss_ori * 100 * alpha + loss_adv * (1 - alpha)
            else:
                alpha = 0.3  # 0.6  # 0.95(best) # 0.3 is best?
                loss = loss_ori * alpha + loss_adv * (1 - alpha)  # 10 * alpha and (1-alpha) | best

            if torch.isnan(ori_feature).any():
                print('*' * 50, "Tensor contains NaN values.")
            if not loss:
                continue
            loss.backward(retain_graph=True)
            optimizer.step()
            self.image_model.target_layer = student_layer
            student_layer.zero_grad()

        return student_layer, mse_adv, ce_adv, mse_ori, ce_ori

    def train_sud_loss(self, num_iters, model_save=False, att_mode=None, learn_mode=None):  # final
        logit_scale, batch, device, check_unlearned = 100, 128, self.cfg.device, True
        student_layer = self.student_layer
        optimizer = torch.optim.SGD(student_layer.parameters(), lr=self.cfg.lr, momentum=0.9)
        # teacher_layer = dc(student_layer)
        bound, step, iters = 0.4, 1e-2, 5
        # test
        # teacher_layer = dc(student_layer)
        for epo in range(num_iters):
            # original
            teacher_layer = dc(student_layer)
            mse_ori_, ce_ori_, mse_adv_, ce_adv_ = 0, 0, 0, 0
            # adv = PGD(teacher_layer, 0.4, 0.5, 5, False, True, True, device, self.P)
            adv = PGD(teacher_layer, bound, step, iters, False, True, True, device,
                      self.P, no_label=self.no_label, using=self.text_model, mode=att_mode, learn_mode=learn_mode)
            optimizer.zero_grad()
            teacher_layer.eval()
            for _, (batch_x, batch_y, _) in enumerate(self.train_loader):
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
                beta = 0
                y_standard = batch_y

                ori_best_group = y_predict == y_standard  # y_predict
                ori_bias_group = y_predict != y_standard  # y_predict or y_debias_predict != y_standard
                if self.no_label:
                    # if label not changed ==> ignore. teacher feature == student feature
                    adv_best_group = y_adv_predict_debias[y_best_group] == y_predict[y_best_group]
                    # if label changed ==> attacked. ori_y == attack_loss
                    adv_bias_group = y_adv_predict_debias[y_best_group] != y_predict[y_best_group]

                    loss_adv, mse_adv, ce_adv = self.sud_loss(adv_feature, adv_feature_t, adv_logit, y_standard,
                                                              adv_best_group, adv_bias_group, beta=0.8,
                                                              mask=y_best_group)
                    ce_adv = loss_adv

                else:
                    if learn_mode is not None and learn_mode == 'bias':
                        y_worst_group = ~y_best_group
                        adv_bias_group = y_adv_predict != y_predict
                        loss_adv, mse_adv, ce_adv = self.sud_loss(adv_feature, adv_feature_t, adv_logit_debias, batch_y,
                                                                  y_best_group, y_worst_group, beta=0)
                        loss_adv2, mse_adv_, ce_adv_ = self.sud_loss(adv_feature, adv_feature_t, adv_logit_debias,
                                                                     batch_y,
                                                                     ~adv_bias_group, adv_bias_group, beta=0)
                        loss_adv += loss_adv2
                    else:

                        case = ['wrong_best', 'wrong']
                        case = case[1]
                        if case == 'wrong_best':  # 1. CE : GT == y_predict =/= y_adv
                            adv_bias_group = (y_adv_predict != y_predict) & ori_best_group  # original
                        else:  # 2. CE : GT =/= y_adv
                            adv_bias_group = (y_adv_predict != y_predict)  # & ori_best_group # left > if wrong
                        # adv_bias_group = (y_adv_predict != y_predict) & ~ori_best_group # right > if wrong & bias
                        adv_best_group = ~adv_bias_group
                        # adv_best_group = y_adv_predict[y_best_group] == y_predict[y_best_group]  # if label not changed ==> ignore. teacher feature == student feature
                        # adv_bias_group = y_adv_predict[y_best_group] != y_predict[y_best_group]  # if label changed ==> attacked. ori_y == attack_loss
                        # adv_best_group = y_adv_predict == y_predict  # if label not changed ==> ignore. teacher feature == student feature
                        # adv_bias_group = y_adv_predict != y_predict  # if label changed ==> attacked. ori_y == attack_loss
                        # if torch.any(adv_bias_group):
                        #     adv_feature_t[adv_bias_group] = adv_feature[adv_bias_group] # ori_bias_group
                        loss_adv, mse_adv, ce_adv = self.sud_loss(adv_feature, adv_feature_t,
                                                                  adv_logit if learn_mode is not None and learn_mode == 'bafa' else adv_logit_debias,
                                                                  y_standard, adv_best_group, adv_bias_group, beta=beta)
                        # , mask=y_best_group
                    # true_ = y_predict == y_debias_predict
                    # loss_ori, mse_ori, ce_ori = self.sud_loss(adv_feature, adv_feature_t, adv_logit_debias, y_standard,
                    #                                           adv_best_group, true_, beta=0)

                # best group : KL | bias_group : CE

                # loss_ori, mse_ori, ce_ori = self.sud_loss(ori_feature, ori_feature_t, debias_logit, y_standard,
                #                                           ori_best_group, ori_bias_group, beta=beta)

                # loss_ori, mse_ori, ce_ori = self.sud_loss(ori_feature, ori_feature_t, debias_logit, y_standard,
                #                                                ori_best_group, ori_bias_group, beta=beta)
                # test
                # ori_best_group = ori_best_group & (y_predict == y_debias_predict) # GT and UNChanged........
                # ori_bias_group = ~ori_best_group # Wrong or Changed
                loss_ori, mse_ori, ce_ori = self.sud_loss(ori_feature, ori_feature_t,
                                                          ori_logit if learn_mode is not None and learn_mode == 'bafa' else debias_logit,
                                                          y_standard, ori_best_group, ori_bias_group, beta=0.9)

                if self.no_label:
                    alpha = 0.99
                    loss = loss_ori * 100 * alpha + loss_adv * (1 - alpha)
                else:
                    alpha = 0.3  # 0.6  # 0.95(best) # 0.3 is best?
                    loss = loss_ori * alpha + loss_adv * (1 - alpha)  # 10 * alpha and (1-alpha) | best

                if torch.isnan(ori_feature).any():
                    print('*' * 50, "Tensor contains NaN values.")
                if not loss:
                    continue

                loss.backward(retain_graph=True)
                optimizer.step()
                self.image_model.target_layer = student_layer
                student_layer.zero_grad()

                mse_ori_ += mse_ori.item()
                ce_ori_ += ce_ori.item()
                mse_adv_ += mse_adv.item()
                ce_adv_ += ce_adv.item()

            # Check that unlearning is done.
            if not epo % 3:
                pre_acc, our_acc, pre_acc_de, our_acc_de, att_suc, att_suc_de = 0, 0, 0, 0, 0, 0
                for _, (batch_x, batch_y, _) in enumerate(self.val_loader):
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
                    break
                print(f'val - att_suc = {att_suc} | pre_acc = {pre_acc} | our_acc = {our_acc} | '
                      f'att_suc_de = {att_suc_de} | pre_acc_de = {pre_acc_de} | our_acc_de = {our_acc_de} | ')
            if not epo % 5 or (17 < epo < 25):
                if model_save and not epo % 5:
                    import os
                    name = f'isis_{self.data}'
                    os.makedirs(name, exist_ok=True)
                    torch.save(student_layer.state_dict(),
                               f'{name}/student_layer{epo}_{alpha}_{beta}_{att_mode}_{bound}_{step}_{iters}_{learn_mode}.pth')
                if epo >= 4:
                    self.test(False,
                              learn_mode='bafa') if learn_mode is not None and learn_mode == 'bafa' else self.test(
                        False)
            print(
                f'epoch: {epo} | mse_ori : {round(mse_ori_, 2)} | ce_ori: {round(ce_ori_, 2)}'
                f' | mse_adv : {round(mse_adv_, 2)} | | ce_adv : {round(ce_adv_, 2)}')
        self.image_model.target_layer = student_layer

    def get_feat(self, X_I, X_D=None):
        Z_D = None if X_D is None else self.text_model.encod(X_D)
        Z_I = self.image_model.encod(X_I.to(self.cfg.device))

        return Z_I, Z_D

    def get_can_embedding(self, loader_base, with_label=False, get_feature=False):
        all_embeddings, batch_y_set = [], []
        for batch_x, batch_y, _ in loader_base:
            all_embeddings.append(self.image_model.encod(batch_x.to(self.cfg.device), get_feature).float())
            batch_y_set.append(batch_y)
        return (torch.cat(all_embeddings), torch.cat(batch_y_set)) if with_label else torch.cat(all_embeddings)

    def get_textfeat(self, X_D):
        Z_D = self.text_model.encod(X_D)
        return Z_D

    def load_student(self, epoch=10, alpha=None, beta=None, mode=None, name=None, bound=None, step=None, iter=None):
        if alpha is None and beta is None:
            student_layer_state_dict = torch.load(f'{name}/student_layer{epoch}.pth')
        elif mode is None:
            student_layer_state_dict = torch.load(f'{name}/student_layer{epoch}_{alpha}_{beta}.pth')
        elif bound is None:
            student_layer_state_dict = torch.load(f'{name}/student_layer{epoch}_{alpha}_{beta}_{mode}.pth')
        else:
            student_layer_state_dict = torch.load(
                f'{name}/student_layer{epoch}_{alpha}_{beta}_{mode}_{bound}_{step}_{iter}.pth')
        self.student_layer.load_state_dict(student_layer_state_dict)

    def get_feature_with_attack(self, input, gt, load_epoch=None, alpha=None, beta=None, mode=None, name=None,
                                bound=None, step=None, iter=None):
        if load_epoch is not None:
            if name is None:
                name = 'save_for_test'

            self.load_student(load_epoch, alpha, beta, mode, name, bound, step, iter)
        if bound is None:
            bound, step, iter = 0.2, 5e-3, 5
        x = self.image_model.get_img_feature(input.to(self.cfg.device))
        adv = PGD(self.student_layer, bound, step, iter, False, True, False,
                  no_label=self.no_label, device=self.cfg.device, mode=mode)
        x_adv = adv.perturb_bafa(x, self.text_embeddings, self.debias_text_embedding, target_y=gt.to(self.cfg.device),
                                 model=self.student_layer, device=self.cfg.device)
        ori_feature = self.student_layer(x)
        adv_feature = self.student_layer(x_adv)
        return ori_feature, adv_feature

    def get_feature_with_orginal_attack(self, input, gt):
        self.load_student(0, 0.5, 0.9, 'bafa_kl', 'save_for_new_group')
        x = self.image_model.get_img_feature(input.to(self.cfg.device))
        adv = PGD(self.student_layer, 0.3, 0.01, 10, False, True, True, no_label=self.no_label, device=self.cfg.device)
        x_adv = adv.perturb_bafa(x, self.text_embeddings, self.debias_text_embedding, target_y=gt.to(self.cfg.device),
                                 model=self.student_layer, device=self.cfg.device, mode='ori')
        ori_feature = self.student_layer(x)
        adv_feature = self.student_layer(x_adv)
        return ori_feature, adv_feature

    def test(self, all_acc=True, learn_mode=None):
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
            if learn_mode is not None and learn_mode == 'bafa':
                dataset_predictions_origin_text = (
                    self.text_model.get_zeroshot_predictions(imfeat_test, self.text_embeddings, self.cfg,
                                                             temperature=100.))
                print('original text acc')
                evaluate_clip(dataset_predictions_origin_text, labels_test_y_gt, labels_test_s, verbose=True)
            elif learn_mode is not None and (learn_mode == 'CAN' or learn_mode == 'BCAU'):
                dataset_predictions_origin_text = (
                    self.uo.get_zeroshot_predictions(imfeat_test, self.text_embeddings, self.cfg,
                                                     temperature=100.))
                print('original text acc')
                evaluate_clip(dataset_predictions_origin_text, labels_test_y_gt, labels_test_s, verbose=True)

                if learn_mode == 'BCAU':
                    pre_image = self.get_can_embedding(self.test_loader_base, get_feature=True)
                    debias_image = self.image_model.target_layer(pre_image)
                dataset_predictions_my_text = (
                    self.uo.get_zeroshot_predictions(debias_image, self.text_model.target_layer(self.pre_embedding),
                                                     self.cfg, temperature=100.))
                print('this model text acc')
                evaluate_clip(dataset_predictions_my_text, labels_test_y_gt, labels_test_s, verbose=True)

            else:
                dataset_predictions = self.text_model.get_zeroshot_predictions(debias_image, self.debias_text_embedding,
                                                                               self.cfg, temperature=100.)
                print('result for debias text & vision set:')
                evaluate_clip(dataset_predictions, labels_test_y_gt, labels_test_s, verbose=True)

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
        from torch.utils.data import DataLoader
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

    def cifar_test(self):
        import os
        from torchvision.datasets import CIFAR100, CIFAR10
        # Load the datasets
        root = os.path.expanduser("~/.cache")
        test_cifar100 = CIFAR100(root, download=True, train=False, transform=self.text_model.base_transform)
        test_cifar10 = CIFAR10(root, download=True, train=False, transform=self.text_model.base_transform)

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

    def cifar_test2(self):
        cifar_test_ = Cifar_test(dc(self.text_model), dc(self.image_model), self.cfg, self.uo)
        cifar_test_.cifar_test()


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

    def get_embedding(self, x, f1=None, f2=None):
        if f1 is None:
            x = self.get_embeddings(x, self.base_model, self.cfg, normalize=True, verbose=False)
        else:
            x = self.get_embeddings(x, f1.to(self.cfg.device), self.cfg, normalize=True, verbose=False)
        return x if f2 is None else f2(x.to(self.cfg.device))

    def encod(self, X, get_feature=False):
        if self.P is not None:
            X = torch.matmul(X, self.P.T)
            out = F.normalize(X, dim=-1)
            if get_feature:
                return X
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
    def __init__(self, cfg=None, model=None, mode='img'):
        super(AttackTarget, self).__init__()
        self.device = cfg.device
        self.vis_mode = cfg.load_base_model.lower()
        self.image_encoder = model.visual
        self.model = model
        self.dtype = model.dtype
        self.target_layer = None
        self.mode = mode
        self.set_target()

    def set_target(self):
        if self.mode == 'txt':
            self.target_layer = Target(self.model.text_projection, 'text')
            self.model.text_projection = nn.Parameter(torch.eye(768, 768).to(self.device))
        elif self.mode == 'img':
            if 'vit' in self.vis_mode:
                self.target_layer = Target(self.image_encoder.proj, 'vit')
                self.image_encoder.proj = None
            else:
                self.target_layer = Target(dc(self.image_encoder.attnpool.c_proj))
                self.image_encoder.attnpool.c_proj.weight = nn.Parameter(torch.eye(2048, 2048).to(self.device))
                self.image_encoder.attnpool.c_proj.bias = nn.Parameter(torch.zeros(2048).to(self.device))

    def get_img_feature(self, img):
        with torch.no_grad():
            image_features = self.image_encoder(img)
        return image_features

    def get_txt_feature(self, txt):
        with torch.no_grad():
            txt_features = self.model(txt)
        return txt_features

    def encod(self, X, get_feature=False):
        if self.target_layer is not None:
            X = self.get_img_feature(X)
            out = self.target_layer(X)
            if get_feature:
                return X
        else:
            out = X
        return out


class Base_clip(nn.Module):
    def __init__(self, model, mode='img'):
        super(Base_clip, self).__init__()
        self.model = model
        self.model.eval()
        self.mode = mode

    def encod(self, x, get_feature=False):
        with torch.no_grad():
            if self.mode == 'img':
                return self.model.encode_image(x)
            else:
                return self.model.encode_text(x)


class Target(nn.Module):
    def __init__(self, fc, name='RN'):
        super(Target, self).__init__()
        self.fc = fc
        self.name = name

    def forward(self, x):
        if 'vit' in self.name:
            output = x @ self.fc.to(x.dtype)
        elif 'text' in self.name:
            output = x @ self.fc.to(x.dtype)
        else:
            output = self.fc(x)  # F.linear(x, self.fc.weight, self.fc.bias)
        output = output / output.norm(dim=-1, keepdim=True)
        return output


import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100, CIFAR10
from tqdm import tqdm


class Cifar_test:
    def __init__(self, text_model, image_model, cfg, uo=None):
        self.text_model = text_model
        self.text_model.to(cfg.device)
        self.image_model = image_model
        self.image_model.to(cfg.device)
        self.cfg = cfg
        self.uo = uo
        self.mode = cfg.mode

    def zeroshot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in tqdm(classnames):
                texts = [template.format(classname) for template in templates]  # Format with class
                class_embeddings = self.uo.get_embedding(texts) if not self.mode == 'CAN' else self.uo.get_embedding(
                    texts, self.text_model.model.to(self.cfg.device), self.text_model.target_layer.to(self.cfg.device))
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

    def cifar_test(self):
        # Load the datasets
        root = os.path.expanduser("~/.cache")
        try:
            transform = self.text_model.base_transform
        except AttributeError:
            transform = self.uo.base_transform
        test_cifar100 = CIFAR100(root, download=True, train=False, transform=transform)
        test_cifar10 = CIFAR10(root, download=True, train=False, transform=transform)

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
