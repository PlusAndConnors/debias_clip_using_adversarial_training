from copy import deepcopy as dc

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from torch.optim import AdamW
from itertools import combinations as combi

import os
import clip
from can.attack.simple_pgd import PGD, perturb_bafa_txt, perturb_bafa_img2txt
from can.utils.tools import get_parameter_count
from clip import clip
from network.clip import evaluate_clip
from utils.txt_input import mk_prompt_mapping


class Collaboration:  # Collaborative
    def __init__(self, cfg, uo, train_loader_base, val_loader_base, test_loader_base, test_info):
        # base setting
        self.cfg, self.mode = cfg, cfg.mode  # cfg
        self.uo = dc(uo)
        self.device = self.cfg.device
        self.txt_learn_mode = cfg.txt_learn_mode

        # model setting
        txt_model = SetTarget(cfg, uo.base_model, 1, 'txt', cfg.txt_learn_mode)  # cfg.target_layer
        img_model = SetTarget(cfg, uo.base_model.visual, cfg.target_layer, 'img', cfg.learn_mode)
        self.txt_model = txt_model.to(self.device)
        self.bias_txt_model = dc(self.uo)  # .to(self.device)
        self.img_model = img_model.to(self.device)
        self.bias_img_model = dc(img_model).to(self.device)  # 16 bit

        # dataset setting
        self.train_loader = train_loader_base
        self.val_loader = val_loader_base
        self.test_loader = test_loader_base
        self.len_t = len(val_loader_base.dataset)
        self.test_info = test_info

        # train setting
        self.init_set()
        self.use_debias_vl()

    def init_set(self):
        # self.img_model.eval()
        # self.txt_model.eval()
        # self.img_target_layer = self.img_model.target_layer  # Target layer
        # self.txt_target_layer = self.txt_model.target_layer  # Target layer

        self.bias_sample_check = True
        self.cls_num_list = [0, 1]
        self.data = self.cfg.embeddings_dir.split('/')[-1]
        print(f'txt target parameter : {get_parameter_count(self.txt_model)}')
        print(f'img target parameter : {get_parameter_count(self.img_model)}')

    def exper_set(self, model_save=False, att_mode=None, learn_mode=None, txt_input=None, ):
        self.att_mode, self.learn_mode, self.model_save = att_mode, learn_mode, model_save
        self.l_scale, self.batch, self.txt_input = 100, 128, txt_input
        self.pgd_set()
        self.biasclip = True
        self.sub = torch.tensor(0, device=self.cfg.device)

    def exper_set_txt(self, img_10k):
        self.img_10k_load = img_10k

    def pgd_set(self):
        self.att_bnd, self.att_stp, self.att_itr = 0.4, 1e-3, 10

    def use_debias_vl(self):
        from can.attack.debias_vl import debais_vl_s_c, debias_vl
        spurious_prompt, candidate_prompt, self.S, B, text = debais_vl_s_c(self.cfg)
        with torch.no_grad():
            candidate_embeddings = self.uo.get_embedding(candidate_prompt)
            spurious_embeddings = self.uo.get_embedding(spurious_prompt)
            self.P = debias_vl(spurious_embeddings, candidate_embeddings, self.S)

    def output(self, x, modal, mode):
        with torch.no_grad():
            if modal == 'txt':
                if mode == 'bias':
                    out = self.bias_txt_model.get_embedding(x)  # eval
                else:
                    self.txt_model.eval()
                    z = self.txt_model.get_feature(x)  # float16
                    out = self.txt_model(z)
            else:
                if mode == 'bias':
                    self.bias_img_model.eval()
                    out = self.bias_img_model.img_inference(x)
                else:
                    self.img_model.eval()
                    z = self.img_model.get_feature(x)
                    out = self.img_model(z)
        return out.half()

    def img_tuning(self, img_iters):
        self.txt_model.eval(), self.img_model.train()
        # target_model, d_aux : debias_auxiliary_model, b_aux : bias_auxiliary_model
        target_model, device = self.img_model, self.device
        # optimizer = AdamW(filter(lambda p: p.requires_grad, target_model.parameters()), lr=self.cfg.lr)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, target_model.parameters()), lr=self.cfg.lr,
                                    momentum=0.9)
        with torch.no_grad():
            bias_txt = self.output(self.txt_input, 'txt', 'bias').to(device)
            debias_txt = self.output(self.txt_input, 'txt', 'debias')

        for epo in range(img_iters):
            # 1 epoch setting
            # using clip or t-1 model as bias model
            # teacher_lay = self.bias_img_model if self.biasclip else dc(target_model)  # biased img model
            alpha, beta = 0.5, 0
            sim, ce, mse_adv_, ce_adv_ = 0, 0, 0, 0
            # thinking 1 : pgd target model : student or teacher [base student]
            pgd = PGD(self.cfg, self.att_bnd, self.att_stp, self.att_itr, device, self.att_mode, bias_txt, debias_txt,
                      self.l_scale, self.P)
            optimizer.zero_grad()

            for _, (batch_x, batch_y, _) in enumerate(self.train_loader):
                # 1 epoch setting
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                z = target_model.get_feature(batch_x)
                out_embed = target_model(z)

                # out_embed = out_embed / out_embed.norm(dim=-1, keepdim=True) <- check
                bias_logit = self.l_scale * out_embed @ bias_txt.t()
                debias_logit = self.l_scale * out_embed @ debias_txt.t()

                bias_predict, debias_predict = torch.argmax(bias_logit, dim=-1), torch.argmax(debias_logit, dim=-1)

                # thinking 2 : group update(SUD) | thinking 3 : KD ( 3.2: EMA)
                # well_sample = batch_y == debias_predict

                z_adv = pgd.perturb_bafa(z, target_model, batch_y)
                adv_embed = target_model(z_adv)
                # adv_logit = F.softmax(self.l_scale * adv_embed.float() @ debias_txt.t(), dim=-1)
                adv_logit = self.l_scale * adv_embed @ debias_txt.t()
                # adv_debias_predict = torch.argmax(adv_logit, dim=-1)

                loss_ce = F.cross_entropy(debias_logit, batch_y) + F.cross_entropy(adv_logit, batch_y)
                feat_num = debias_predict.size(0)  # debias_predict[best_group].size(0)
                loss_adv_sim = dc(self.sub)
                for i in range(feat_num):
                    loss_adv_sim += F.mse_loss(debias_logit[i], adv_logit[i], reduction="none").mean()

                # thinking 4. loss [base : kl + ce] --> b_main, deb_main ,b_aux, deb_aux
                loss = loss_ce * alpha + loss_adv_sim * (1 - alpha)  # 10 * alpha and (1-alpha) | best

                if torch.isnan(out_embed).any():
                    print('*' * 50, "Tensor contains NaN values.")
                if not loss:
                    continue

                ce += loss_ce.item()
                sim += loss_adv_sim.item()

                loss.backward(retain_graph=True)
                optimizer.step()

                target_model.zero_grad()

            print(f'epoch: {epo} | ce: {round(ce, 2)} | sim : {round(sim, 2)}')
            # Check that unlearning is done.
            if not epo % 3:
                pre_acc, our_acc, pre_acc_de, our_acc_de, att_suc, att_suc_de, len_v = 0, 0, 0, 0, 0, 0, 0
                for _, (batch_x, batch_y, _) in enumerate(self.val_loader):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    len_v += len(batch_x)
                    z = target_model.get_feature(batch_x)
                    out_embed = target_model(z)

                    debias_predict = torch.argmax(out_embed @ debias_txt.t(), dim=-1)

                    pre_out = self.output(batch_x, 'img', 'bias')
                    pre_predict = torch.argmax(pre_out @ debias_txt.t(), dim=-1)

                    pre_acc += (batch_y == pre_predict).sum()
                    our_acc += (batch_y == debias_predict).sum()
                    if torch.isnan(out_embed).any():
                        print('*' * 50, "Tensor contains NaN values.")
                    if not loss:
                        continue
                    target_model.zero_grad()
                print(f'epoch: {epo} val | pre: {int(pre_acc.item())} | our : {int(our_acc.item())}')

                # if not att_suc and not att_suc_de:
                #     break

            if not epo % 5:
                if self.model_save and not epo % 5:
                    name = f'grad_{self.data}'
                    os.makedirs(name, exist_ok=True)
                    torch.save(target_model.state_dict(), f'{name}/bafa{epo}.pth')
                if epo >= 4:
                    self.test(False, learn_mode='bafa')

    def img_tuning_sud(self, img_iters):
        # target_model, d_aux : debias_auxiliary_model, b_aux : bias_auxiliary_model
        self.txt_model.eval(), self.img_model.train()
        target_model, device = self.img_model, self.device
        # optimizer = AdamW(filter(lambda p: p.requires_grad, target_model.parameters()), lr=self.cfg.lr * 0.1)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, target_model.parameters()), lr=self.cfg.lr * 2,
                                    momentum=0.9)
        test_type = ''  # 'isis'
        with torch.no_grad():
            bias_txt = self.output(self.txt_input, 'txt', 'bias').to(device)
            if test_type == 'isis':
                debias_txt = F.normalize(torch.matmul(bias_txt.cpu(), self.P.T), dim=-1).to(device)
            else:
                debias_txt = bias_txt  # self.output(self.txt_input, 'txt', 'debias').float()

        pgd = PGD(self.cfg, self.att_bnd, self.att_stp, self.att_itr, device, self.att_mode, bias_txt, debias_txt,
                  self.l_scale, self.P)
        for epo in range(img_iters):
            # 1 epoch setting
            teacher_layer = dc(target_model)
            teacher_layer.eval()
            alpha, beta = 0.95, 0
            mse_ori_, ce_ori_, mse_adv_, ce_adv_ = 0, 0, 0, 0
            # thinking 1 : pgd target model : student or teacher [base student]
            # same : batch x, bias_txt, self.p

            optimizer.zero_grad()

            for _, (batch_x, batch_y, _) in enumerate(self.train_loader):
                # 1 epoch setting
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                # target_model.encoder(batch_x.half())
                with torch.no_grad():
                    z = target_model.get_feature(batch_x)
                out_embed = target_model(z)

                bias_logit = self.l_scale * out_embed @ bias_txt.t()
                debias_logit = self.l_scale * out_embed @ debias_txt.t()

                bias_predict, debias_predict = torch.argmax(bias_logit, dim=-1), torch.argmax(debias_logit, dim=-1)
                with torch.no_grad():
                    ori_feat_t = teacher_layer(z)

                z_adv = pgd.perturb_bafa(z, target_model, batch_y)
                adv_embed = target_model(z_adv)
                # adv_logit = F.softmax(self.l_scale * adv_embed.float() @ debias_txt.t(), dim=-1)
                adv_logit = self.l_scale * adv_embed @ debias_txt.t()
                adv_predict = torch.argmax(adv_logit, dim=-1)
                # check group # thinking 2 : group update(SUD) | thinking 3 : KD ( 3.2: EMA)
                y_best_group = debias_predict == batch_y
                adv_worst_group = y_best_group & (adv_predict != batch_y)

                loss_ori, mse_ori, ce_ori = self.sud_loss(out_embed, ori_feat_t, debias_logit, batch_y, y_best_group,
                                                          ~y_best_group, beta=0.9)
                loss_adv, mse_adv, ce_adv = self.sud_loss(adv_embed, ori_feat_t, adv_logit, batch_y, y_best_group,
                                                          ~y_best_group, beta=0)
                loss = loss_ori * alpha + loss_adv * (1 - alpha)

                if torch.isnan(out_embed).any():
                    print('*' * 50, "Tensor contains NaN values.")
                if not loss:
                    continue

                mse_ori_ += mse_ori.item()
                ce_ori_ += ce_ori.item()
                mse_adv_ += mse_adv.item()
                ce_adv_ += ce_adv.item()

                loss.backward(retain_graph=True)
                optimizer.step()
                target_model.zero_grad()
            print(f'epoch: {epo} | mse_ori : {round(mse_ori_, 2)} | ce_ori: {round(ce_ori_, 2)}'
                  f' | mse_adv : {round(mse_adv_, 2)} | | ce_adv : {round(ce_adv_, 2)}')
            # Check that unlearning is done.
            if not epo % 3:
                pre_acc, our_acc, pre_acc_de, our_acc_de, att_suc, att_suc_de, len_v = 0, 0, 0, 0, 0, 0, 0
                with torch.no_grad():
                    for _, (batch_x, batch_y, _) in enumerate(self.val_loader):
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        len_v += len(batch_x)
                        z = target_model.get_feature(batch_x)
                        out_embed = target_model(z)

                        debias_predict = torch.argmax(out_embed @ debias_txt.t(), dim=-1)

                        pre_out = self.output(batch_x, 'img', 'bias')
                        pre_predict = torch.argmax(pre_out @ debias_txt.t(), dim=-1)

                        pre_acc += (batch_y == pre_predict).sum()
                        our_acc += (batch_y == debias_predict).sum()
                        if torch.isnan(out_embed).any():
                            print('*' * 50, "Tensor contains NaN values.")
                        if not loss:
                            continue
                        target_model.zero_grad()
                    print(f'epoch: {epo} val | pre: {int(pre_acc.item())} | our : {int(our_acc.item())}')

                # if not att_suc and not att_suc_de:
                #     break

            if not epo % 5:
                if self.model_save and not epo % 5:
                    name = f'grad_{self.data}'
                    os.makedirs(name, exist_ok=True)
                    torch.save(target_model.state_dict(), f'{name}/bafa{epo}.pth')
                if epo >= 4:
                    self.test()

    def img_tuning_nolabel(self, img_iters, iter, just_tun=False):
        self.txt_model.eval(), self.img_model.train()
        target_model, device, bias_txts = self.img_model, self.device, self.ori_bias_txts.to(self.device).half()
        # teacher_layer = dc(target_model)
        # teacher_layer.eval()
        # optimizer = AdamW(filter(lambda p: p.requires_grad, target_model.parameters()), lr=self.cfg.lr * 0.1)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, target_model.parameters()), lr=self.cfg.lr,
                                    momentum=0.9)
        # --config-file configs/debias_celeba.yaml
        sub = torch.tensor(0, device=self.device).half()
        test_type = 'ours'  # 'ours', # img_first # debias_vl
        with torch.no_grad():
            if test_type == 'debias_vl':
                debias_txt = F.normalize(torch.matmul(self.ori_bias_txts, self.P.T), dim=-1).to(device).half()
            elif test_type == 'ours':
                debias_txt = self.debias_txt
            else:
                debias_txt = self.ori_bias_txts
        with torch.no_grad():
            z, t_ = self.txt_model.get_feature(self.bias_p, only_vec=False)
        bias_prompt_txt = self.txt_model(z, t_)

        pgd = PGD(self.cfg, self.att_bnd, self.att_stp, self.att_itr, device, self.att_mode, bias_txts,
                  debias_txt, self.l_scale, self.P)
        # equa_loss = dc(sub)
        batch = self.train_loader.batch_size
        filenames = f'save_temp/nolabel_{self.learn_mode}_{self.txt_learn_mode}_{img_iters}_img_{iter}_{img_iters - 1}.pth'
        if not os.path.exists(filenames):
            for epo in range(img_iters):
                teacher_layer = dc(target_model)
                teacher_layer.eval()
                optimizer.zero_grad()
                eq_l, dist_l, total_right, att_suc_num = 0, 0, 0, 0
                for it_, (batch_x, y, _) in enumerate(self.train_loader):

                    batch_x = batch_x.to(device)
                    equ_loss, equ_loss1, equ_loss2, distil_loss = dc(sub), dc(sub), dc(sub), dc(sub)
                    y_ = y.to(device)
                    with torch.no_grad():
                        z = target_model.get_feature(batch_x)
                        # out_embed_t = teacher_layer(z)
                        # out_embed_t = self.ori_bias_imgs[it_ * batch:(it_ + 1) * batch].half()
                        out_embed_t2 = teacher_layer(z)
                        debias_logit_t = self.l_scale * out_embed_t2 @ debias_txt.t()  # right 3
                        debias_logit_test = out_embed_t2 @ debias_txt.t()  # right 3

                    out_embed = target_model(z)
                    bias_logit = self.l_scale * out_embed @ bias_txts.t()
                    debias_logit = self.l_scale * out_embed @ debias_txt.t()

                    feat_num = debias_logit.size(0)
                    for i in range(feat_num):
                        distil_loss += F.mse_loss(out_embed_t2[i], out_embed[i], reduction="none").mean()
                    distil_loss = distil_loss / feat_num  # * 10

                    bias_predict, debias_predict = torch.argmax(bias_logit, dim=-1), torch.argmax(debias_logit, dim=-1)
                    both_right_group = bias_predict == debias_predict

                    # base setting
                    # debias_logit_t = debias_logit_t.softmax(dim=-1)
                    # z_adv1, z_adv2 = pgd.perturb_bafa_nolabel_mix(z, target_model, bias_txt)
                    # adv_embed2 = target_model(z_adv2)
                    # adv_logit2 = (self.l_scale * adv_embed2 @ debias_txt.t()).softmax(dim=-1)
                    # for i in range(feat_num):
                    #     equ_loss2 += F.mse_loss(adv_logit2[i], debias_logit_t[i], reduction="none").mean()
                    # equ_loss += equ_loss2 / feat_num / 2
                    #
                    # adv_embed1 = target_model(z_adv1)
                    # adv_logit1 = (self.l_scale * adv_embed1 @ debias_txt.t()).softmax(dim=-1)
                    #
                    # feat_num = debias_logit_t.size(0)
                    # for i in range(feat_num):
                    #     equ_loss1 += F.mse_loss(adv_logit1[i], debias_logit_t[i], reduction="none").mean()
                    # equ_loss += equ_loss1 / feat_num / 2
                    #
                    # if any(both_right_group):
                    #     z_adv = pgd.perturb_bafa_nolabel_label(z[both_right_group], target_model, y_predict)
                    #     # if iter:
                    #     #     z_adv = pgd.perturb_bafa_nolabel_label(z[both_right_group], target_model, y_predict)
                    #     # else:
                    #     # if len(both_right_group) == z.size(0):
                    #     #     z_adv = pgd.perturb_bafa_nolabel_use_predict(z[both_right_group], target_model,
                    #     #                                                  bias_txt, y_predict)
                    #     # else:
                    #     #     z_adv = pgd.perturb_bafa_nolabel_use_predict(z[:, both_right_group], target_model,
                    #     #                                                  bias_txt, y_predict)  # , bias_txt
                    #     adv_embed = target_model(z_adv)
                    #     adv_logit = (self.l_scale * adv_embed @ debias_txt.t()).softmax(dim=-1)
                    #
                    #     debias_logit_t = debias_logit_t.softmax(dim=-1)[both_right_group]
                    #     feat_num = debias_logit_t.size(0)
                    #     for i in range(feat_num):
                    #         equ_loss1 += F.mse_loss(adv_logit[i], debias_logit_t[i], reduction="none").mean()
                    #     equ_loss += equ_loss1 / feat_num / 2
                    if iter == 7:
                        pass
                    if any(both_right_group):
                        y_predict = debias_predict[both_right_group]
                        z_adv = pgd.perturb_bafa_nolabel_label(z[both_right_group], target_model, y_predict)  # right 1
                        # z_adv = pgd.perturb_bafa_nolabel_use_predict(z[both_right_group], target_model, bias_prompt_txt,
                        #                                              y_predict)  # right 2
                        adv_embed = target_model(z_adv)
                        adv_logit = (self.l_scale * adv_embed @ debias_txt.t()).softmax(dim=-1)
                        # adv_logit_test = adv_embed @ debias_txt.t()
                        # adv_logit_o = (self.l_scale * adv_embed @ bias_txts.t()).softmax(dim=-1) # if use left, not use right
                        base_logit = debias_logit[both_right_group].softmax(dim=-1)
                        debias_logit_t = debias_logit_t.softmax(dim=-1)[both_right_group]

                        att_suc_0 = (y_predict == 0) & (adv_logit[:, 0] < base_logit[:, 0])
                        att_suc_1 = (y_predict == 1) & (adv_logit[:, 1] < base_logit[:, 1])
                        att_suc = att_suc_0 | att_suc_1
                        try_ = 1
                        if any(att_suc):
                            if try_:
                                for j in range(sum(att_suc)):
                                    for i, cls_emb in enumerate(debias_txt):
                                        equ_loss1 += self.l_scale * (out_embed_t2[j] @ cls_emb - adv_embed[j] @ cls_emb) ** 2
                                equ_loss += equ_loss1 / sum(att_suc) / 2

                                # for i in range(sum(att_suc)):
                                #     equ_loss1 += F.l1_loss(adv_logit[att_suc][i], debias_logit_t[att_suc][i],
                                #                             reduction="none").mean()
                                # equ_loss += equ_loss1 / sum(att_suc)
                            else:
                                for i in range(sum(att_suc)):
                                    equ_loss1 += F.mse_loss(adv_logit[att_suc][i], debias_logit_t[att_suc][i],
                                                            reduction="none").mean()
                                equ_loss += equ_loss1 / sum(att_suc)

                        # z_adv1, z_adv2 = pgd.perturb_bafa_nolabel(z[both_right_group], target_model)
                        # adv_embed1, adv_embed2 = target_model(z_adv1), target_model(z_adv2)
                        # feat_num = adv_embed1.size(0)
                        # for j in range(feat_num):
                        #     for i, cls_emb in enumerate(debias_txt):
                        #         equ_loss1 += (self.l_scale * (adv_embed1[j] @ cls_emb - adv_embed2[j] @ cls_emb)) ** 2
                        # equ_loss += equ_loss1 / sum(both_right_group)

                    # if iter:
                    #     z_adv = pgd.perturb_bafa_nolabel_label(z[both_right_group], target_model, y_predict)
                    # else:
                    # if len(both_right_group) == z.size(0):
                    #     z_adv = pgd.perturb_bafa_nolabel_use_predict(z[both_right_group], target_model,
                    #                                                  bias_prompt_txt, y_predict)
                    # else:
                    #     z_adv = pgd.perturb_bafa_nolabel_use_predict(z[:, both_right_group], target_model,
                    #                                                  bias_prompt_txt, y_predict)

                    loss = equ_loss + distil_loss
                    if torch.isnan(out_embed).any():
                        print('*' * 50, "Tensor contains NaN values.")
                    if not loss:
                        continue
                    eq_l += equ_loss.item()
                    dist_l += distil_loss.item()
                    total_right += sum(both_right_group).item()
                    # att_suc_num += sum(att_suc).item()

                    loss.backward(retain_graph=True)
                    optimizer.step()
                    target_model.zero_grad()
                print(f'img_epoch: {epo} | eq_l : {round(eq_l, 2)} | distil: {round(dist_l, 2)}', 'total_right : ',
                      total_right, 'att_suc_num : ', att_suc_num)
                # print(f'epoch: {epo} | mse_ori : {round(mse_ori_, 2)} | ce_ori: {round(ce_ori_, 2)}'
                #       f' | mse_adv : {round(mse_adv_, 2)} | | ce_adv : {round(ce_adv_, 2)}')
                # Check that unlearning is done.
                if not epo % 3:
                    pre_acc, our_acc, pre_acc_de, our_acc_de, att_suc, att_suc_de, len_v = 0, 0, 0, 0, 0, 0, 0
                    with torch.no_grad():
                        for _, (batch_x, batch_y, _) in enumerate(self.val_loader):
                            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                            len_v += len(batch_x)
                            with torch.no_grad():
                                z = target_model.get_feature(batch_x)
                            out_embed = target_model(z)

                            debias_predict = torch.argmax(out_embed @ debias_txt.t(), dim=-1)
                            pre_out = self.output(batch_x, 'img', 'bias')
                            pre_predict = torch.argmax(pre_out @ debias_txt.t(), dim=-1)

                            pre_acc += (batch_y == pre_predict).sum()
                            our_acc += (batch_y == debias_predict).sum()
                            if torch.isnan(out_embed).any():
                                print('*' * 50, "Tensor contains NaN values.")
                            if not loss:
                                continue
                            target_model.zero_grad()
                        print(f'epoch: {epo} val | pre: {int(pre_acc.item())} | our : {int(our_acc.item())}')

                    # if not att_suc and not att_suc_de:
                    #     break

                if not epo % 5:
                    if self.model_save and not epo % 5:
                        name = f'grad_{self.data}'
                        os.makedirs(name, exist_ok=True)
                        torch.save(target_model.state_dict(), f'{name}/bafa{epo}.pth')
                    if epo >= 4:
                        self.test()

            self.img_model.eval()
            # torch.save(self.img_model.state_dict(),
            #            f'save_temp/nolabel_{self.learn_mode}_{self.txt_learn_mode}_{img_iters}_img_{iter}_{epo}.pth')
        else:
            print('img', iter, '<-', img_iters, 'way')
            self.img_model.load_state_dict(torch.load(filenames))
            self.test()
        debias_imgs, bias_imgs, batch_ys = [], [], []
        with torch.no_grad():
            for _, (batch_x, batch_y, _) in enumerate(self.train_loader):  # freeze img
                # 1 epoch setting
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                z = self.img_model.get_feature(batch_x)
                debias_img = self.img_model(z)
                debias_imgs.append(debias_img)
        self.debias_img_set = torch.cat(debias_imgs)

    # def img_tuning_label(self, img_iters, iter, just_tun=False):
    #     self.txt_model.eval(), self.img_model.train()
    #     target_model, device, bias_txts = self.img_model, self.device, self.ori_bias_txts.to(self.device).half()
    #     # optimizer = AdamW(filter(lambda p: p.requires_grad, target_model.parameters()), lr=self.cfg.lr * 0.1)
    #     optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, target_model.parameters()), lr=self.cfg.lr,
    #                                 momentum=0.9)
    #     sub = torch.tensor(0, device=self.device).half()
    #     test_type = 'ours'  # 'ours', # img_first # debias_vl
    #     test = None  # 'att_0.1' # None
    #     with torch.no_grad():
    #         if test_type == 'debias_vl':
    #             debias_txt = F.normalize(torch.matmul(self.ori_bias_txts, self.P.T), dim=-1).to(device).half()
    #         elif test_type == 'ours':
    #             # z, t_ = self.txt_model.get_feature(self.main_p, only_vec=False)
    #             # debias_txt = self.txt_model(z, t_)
    #             debias_txt = self.debias_txt
    #         else:
    #             debias_txt = self.ori_bias_txts
    #         z, t_ = self.txt_model.get_feature(self.bias_p, only_vec=False)
    #         bias_txt = self.txt_model(z, t_)
    #     # z, t_ = self.txt_model.get_feature(self.mix_p, only_vec=False)
    #     # mix_txt = self.txt_model(z, t_)
    #
    #     pgd = PGD(self.cfg, self.att_bnd, self.att_stp, self.att_itr, device, self.att_mode, bias_txts,
    #               debias_txt, self.l_scale, self.P)
    #     ce_loss, att_loss = dc(sub), dc(sub)
    #     batch = self.train_loader.batch_size
    #
    #     epo = f'just_tun_{img_iters - 1}' if just_tun else img_iters - 1
    #     filenames = f'save_temp/{self.learn_mode}_{self.txt_learn_mode}_{img_iters}_img_{iter}_{epo}.pth'
    #     if not os.path.exists(filenames):
    #         for epoc in range(img_iters):
    #             teacher_layer = dc(target_model)
    #             teacher_layer.eval()
    #             optimizer.zero_grad()
    #             eq_l, dist_l, ce_l, at_l = 0, 0, 0, 0
    #             for it_, (batch_x, batch_y, _) in enumerate(self.train_loader):
    #                 batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    #                 with torch.no_grad():
    #                     z = target_model.get_feature(batch_x)
    #                     out_embed_old = self.ori_bias_imgs[it_ * batch:(it_ + 1) * batch].half()
    #                     debias_logit_old = self.l_scale * out_embed_old @ debias_txt.t()
    #                     debias_old_predict = torch.argmax(debias_logit_old, dim=-1)
    #                 out_embed = target_model(z)
    #
    #                 bias_logit = self.l_scale * out_embed @ bias_txts.t()
    #                 debias_logit = self.l_scale * out_embed @ debias_txt.t()
    #
    #                 bias_predict, debias_predict = torch.argmax(bias_logit, dim=-1), torch.argmax(debias_logit, dim=-1)
    #
    #                 with torch.no_grad():
    #                     out_dim_t = teacher_layer(z)
    #                     debias_logit_t = self.l_scale * teacher_layer(z) @ debias_txt.t()
    #
    #                 wrong_group = (batch_y != debias_predict)
    #                 old_right = debias_old_predict == batch_y
    #
    #                 equ_loss, equ_loss1, equ_loss2, distil_loss = dc(sub), dc(sub), dc(sub), dc(sub)
    #                 if not just_tun:
    #                     z_adv1 = pgd.perturb_bafa(z, target_model, batch_y, bias_txt)  # , bias_txt
    #                     with torch.no_grad():
    #                         adv_logit1_t = self.l_scale * teacher_layer(z_adv1) @ debias_txt.t()
    #                     adv_embed1 = target_model(z_adv1)
    #
    #                     adv_logit1 = self.l_scale * adv_embed1 @ debias_txt.t()
    #                     adv_predict1_bias = torch.argmax(self.l_scale * adv_embed1 @ bias_txts.t(), dim=-1)
    #                     adv_predict1 = torch.argmax(adv_logit1, dim=-1)
    #
    #                     at_suc1 = (batch_y != adv_predict1_bias) & ~wrong_group
    #                     att_loss = F.cross_entropy(adv_logit1, batch_y) * 0.2
    #                     # if torch.any(at_suc1):
    #                     #     att_loss = F.cross_entropy(adv_logit1[at_suc1], batch_y[at_suc1])
    #                 if torch.any(wrong_group):
    #                     ce_loss = F.cross_entropy(debias_logit[wrong_group], batch_y[wrong_group])
    #                 if torch.any(old_right):
    #                     feat_num = debias_logit[old_right].size(0)
    #                     for i in range(feat_num):
    #                         distil_loss += F.mse_loss(out_embed_old[i], out_embed[i], reduction="none").mean()
    #                     distil_loss = distil_loss / feat_num * 100
    #
    #                 loss = att_loss + ce_loss + distil_loss if not just_tun else ce_loss + distil_loss
    #                 if torch.isnan(out_embed).any():
    #                     print('*' * 50, "Tensor contains NaN values.")
    #                 if not loss:
    #                     continue
    #                 eq_l += equ_loss.item()
    #                 dist_l += distil_loss.item()
    #                 ce_l += ce_loss.item()
    #                 at_l += att_loss.item()
    #                 loss.backward(retain_graph=True)
    #                 optimizer.step()
    #                 target_model.zero_grad()
    #             print(
    #                 f'img_epoch: {epoc} | ce_l : {round(ce_l, 2)} | at_l : {round(at_l, 2)} | distil: {round(dist_l, 2)}')
    #             # print(f'epoch: {epo} | mse_ori : {round(mse_ori_, 2)} | ce_ori: {round(ce_ori_, 2)}'
    #             #       f' | mse_adv : {round(mse_adv_, 2)} | | ce_adv : {round(ce_adv_, 2)}')
    #             # Check that unlearning is done.
    #             if not epoc % 3:
    #                 pre_acc, our_acc, pre_acc_de, our_acc_de, att_suc, att_suc_de, len_v = 0, 0, 0, 0, 0, 0, 0
    #                 with torch.no_grad():
    #                     for _, (batch_x, batch_y, _) in enumerate(self.val_loader):
    #                         batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    #                         len_v += len(batch_x)
    #                         with torch.no_grad():
    #                             z = target_model.get_feature(batch_x)
    #                         out_embed = target_model(z)
    #
    #                         debias_predict = torch.argmax(out_embed @ debias_txt.t(), dim=-1)
    #                         pre_out = self.output(batch_x, 'img', 'bias')
    #                         pre_predict = torch.argmax(pre_out @ debias_txt.t(), dim=-1)
    #
    #                         pre_acc += (batch_y == pre_predict).sum()
    #                         our_acc += (batch_y == debias_predict).sum()
    #                         if torch.isnan(out_embed).any():
    #                             print('*' * 50, "Tensor contains NaN values.")
    #                         if not loss:
    #                             continue
    #                         target_model.zero_grad()
    #                     print(f'epoch: {epoc} val | pre: {int(pre_acc.item())} | our : {int(our_acc.item())}')
    #                 # if not att_suc and not att_suc_de:
    #                 #     break
    #
    #             if not epoc % 5:
    #                 if self.model_save and not epoc % 5:
    #                     name = f'grad_{self.data}'
    #                     os.makedirs(name, exist_ok=True)
    #                     torch.save(target_model.state_dict(), f'{name}/bafa{epoc}.pth')
    #                 if epoc >= 4:
    #                     self.test()
    #
    #         self.img_model.eval()
    #         epo = f'just_tun_{epoc}' if just_tun else epoc
    #         epo = epo if test is None else test  #
    #         # torch.save(self.img_model.state_dict(),
    #         #            f'save_temp/{self.learn_mode}_{self.txt_learn_mode}_{img_iters}_img_{iter}_{epo}.pth')
    #         self.test()
    #     else:
    #         print('img', iter, '<-', img_iters, 'way')
    #         self.img_model.load_state_dict(torch.load(filenames))
    #         self.test()
    #     debias_imgs, bias_imgs, batch_ys = [], [], []
    #     with torch.no_grad():
    #         for _, (batch_x, batch_y, _) in enumerate(self.train_loader):  # freeze img
    #             # 1 epoch setting
    #             batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
    #             z = self.img_model.get_feature(batch_x)
    #             debias_img = self.img_model(z)
    #             debias_imgs.append(debias_img)
    #     self.debias_img_set = torch.cat(debias_imgs)

    def plan(self):
        # 16 am txt tuning biased img [img ce loss + txt sim loss],,,
        # 16 pm txt tuning based debiased img [img ce loss model - load('save') iter save learn
        # 17 am ck result (pgd inference visualize, acc, param, )
        # 16 =>
        # No label first 1. txt tun[] -> bafa(both aug) img sim -> bafa txt tun -> bafa(both aug) img
        # 2. with debais vl bafa 3. one target aug(predict based)
        # 17 => No BAFA : 1. txt tun -> img tun(ce) -> txt tun(ce) -> img tun, 3. all ce tun

        # txt-img pair - ce loss, [ce_loss[label is wrong], simi loss[adv is wrong], kd loss[adv is not wrong]]
        pass

    def clip_img_feat(self, exp=True, data_t='waterbirds'):
        debias_imgs, bias_imgs = [], []
        self.main_p, self.bias_p, self.mix_p, self.mapping = mk_prompt_mapping(data_t)
        self.mix_len, self.bias_len = len(self.mix_p), len(self.bias_p)
        if exp:
            with torch.no_grad():
                for _, (batch_x, batch_y, _) in enumerate(self.train_loader):  # freeze img
                    # 1 epoch setting
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    bias_img = self.bias_img_model.img_inference(batch_x)
                    bias_imgs.append(bias_img)
            self.ori_bias_imgs = torch.cat(bias_imgs)  # first only use bias
            self.ori_bias_txts = self.bias_txt_model.get_embedding(self.main_p)

    def txt_tuning_nolabel(self, txt_iters, iter, just_tun=False):
        # target_model, d_aux : debias_auxiliary_model, b_aux : bias_auxiliary_model
        self.img_model.eval(), self.txt_model.train(), self.bias_img_model.eval()
        target_model, device = self.txt_model, self.device
        sub = torch.tensor(0, device=self.device).half()
        # optimizer = AdamW(filter(lambda p: p.requires_grad, target_model.parameters()), lr=self.cfg.lr)
        # torch.save(torch.cat(all_embeddings), 'img_save')

        epo = f'just_tun_{txt_iters - 1}' if just_tun else txt_iters - 1
        filenames = f'save_temp/nolabel_{self.learn_mode}_{self.txt_learn_mode}_{txt_iters}_txt_{iter}_{epo}.pth'
        if not os.path.exists(filenames):
            if iter:
                lr = self.cfg.lr
            else:
                lr = self.cfg.lr * 3
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, target_model.parameters()), lr=lr,
                                        momentum=0.9)
            with torch.no_grad():
                z, t_ = target_model.get_feature(self.main_p, only_vec=False)
                z_bias, t_b = target_model.get_feature(self.bias_p, only_vec=False)
                z_mix, t_m = target_model.get_feature(self.mix_p, only_vec=False)
                em_tar0 = target_model(z, t_)
                em_bias0 = target_model(z_bias, t_b)
                em_mix0 = target_model(z_mix, t_m)
                if iter:
                    logit_t = (100 * self.debias_img_set @ em_tar0.t()).softmax(dim=-1)

            mix_len, bias_len = em_mix0.size(0), em_bias0.size(0)
            for epo in range(txt_iters):
                equ_loss, distil_loss, target_loss, equa_loss, pgd_loss = dc(sub), dc(sub), dc(sub), dc(sub), dc(sub)
                optimizer.zero_grad()

                em_tar = target_model(z, t_)
                # em_bias = target_model(z_bias, t_b)
                # em_mix = target_model(z_mix, t_m)
                for wz_, z0_ in zip(em_tar, em_tar0):
                    distil_loss += F.mse_loss(wz_, z0_).to(device)  # * 10

                for i, cls_emb in enumerate(em_tar):  # binary
                    uq_pair = [v[2] for v in self.mapping.values() if v[1] == i]
                    for j, (f, s) in enumerate(combi(uq_pair, 2)):
                        equa_loss += (cls_emb @ em_mix0[i * bias_len + f] - cls_emb @ em_mix0[i * bias_len + s]) ** 2
                eq_loss = equa_loss / (i + 1) / (j + 1) * 500

                if iter:
                    # z_adv1, z_adv2 = perturb_bafa_img2txt(z, target_model, self.debias_img_set,
                    #                                       self.ori_bias_imgs.half(), em_bias0, t_)
                    z_adv = perturb_bafa_txt(z, target_model, em_bias0, t_)
                    em_tar1_adv = target_model(z_adv, t_)
                    adv_logit = (self.l_scale * self.debias_img_set @ em_tar1_adv.t()).softmax(dim=-1)
                    feat_num = adv_logit.size(0)
                    for i in range(feat_num):
                        pgd_loss += F.mse_loss(adv_logit[i], logit_t[i], reduction="none").mean()  # pairwise sim
                    pgd_loss /= feat_num
                    loss = eq_loss + distil_loss + pgd_loss

                    # em_tar1_adv, em_tar2_adv = target_model(z_adv1, t_), target_model(z_adv2, t_)
                    # adv_logit1 = (self.l_scale * self.debias_img_set @ em_tar1_adv.t()).softmax(dim=-1)
                    # adv_logit2 = (self.l_scale * self.debias_img_set @ em_tar2_adv.t()).softmax(dim=-1)
                    # feat_num = adv_logit1.size(0)
                    # for i in range(feat_num):
                    #     pgd_loss += F.mse_loss(adv_logit1[i], logit_t[i], reduction="none").mean()  # pairwise sim
                    #     pgd_loss += F.mse_loss(adv_logit2[i], logit_t[i], reduction="none").mean()
                    # pgd_loss /= feat_num * 2
                    # loss = eq_loss + distil_loss + pgd_loss
                else:
                    loss = eq_loss + distil_loss  # + match_loss
                loss.backward()
                optimizer.step()
                # target_model.zero_grad()
                print(f'txt {epo} epoch |' 'eq_loss : ', round(eq_loss.item(), 4), 'pgd_loss:',
                      round(pgd_loss.item(), 4), 'dist_loss:', round(distil_loss.item(), 4))
                if not epo % 5:
                    if self.model_save and not epo % 5:
                        name = f'grad_{self.data}'
                        os.makedirs(name, exist_ok=True)
                        torch.save(target_model.state_dict(), f'{name}/txt_{epo}.pth')
                    if epo >= 4:
                        avg_acc, robust_acc, groups_acc = self.test()
            if epo >= 5:
                self.txt_model.eval()
                z, t_ = self.txt_model.get_feature(self.main_p, only_vec=False)
                self.debias_txt = self.txt_model(z, t_)
                if just_tun:
                    epo = f'just_tun_{epo}'
                # torch.save(self.txt_model.state_dict(),
                #            f'save_temp/nolabel_{self.learn_mode}_{self.txt_learn_mode}_{txt_iters}_txt_{iter}_{epo}.pth')
        else:
            self.txt_model.load_state_dict(torch.load(filenames))  #
            z, t_ = self.txt_model.get_feature(self.main_p, only_vec=False)
            self.debias_txt = self.txt_model(z, t_)
            print('txt', iter, '<-', txt_iters, 'way')
            self.test()

    def just_txt_tuning(self, txt_iters, iter, just_tun=False):
        # target_model, d_aux : debias_auxiliary_model, b_aux : bias_auxiliary_model
        self.img_model.eval(), self.txt_model.train(), self.bias_img_model.eval()
        target_model, device = self.txt_model, self.device
        sub = torch.tensor(0, device=self.device).half()
        # optimizer = AdamW(filter(lambda p: p.requires_grad, target_model.parameters()), lr=self.cfg.lr)
        # torch.save(torch.cat(all_embeddings), 'img_save')
        if iter:
            debias_imgs, bias_imgs, batch_ys = [], [], []
            with torch.no_grad():
                for _, (batch_x, batch_y, _) in enumerate(self.train_loader):  # freeze img
                    # 1 epoch setting
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    z = self.img_model.get_feature(batch_x)
                    debias_img = self.img_model(z)
                    debias_imgs.append(debias_img)
                    batch_ys.append(batch_y)
            debias_img_set = torch.cat(debias_imgs)
            batch_y_set = torch.cat(batch_ys)

        epo = f'just_tun_{txt_iters - 1}' if just_tun else txt_iters - 1
        filenames = f'save_temp/{self.learn_mode}_{self.txt_learn_mode}_{txt_iters}_txt_{iter}_{epo}.pth'
        if not os.path.exists(filenames):
            if iter:
                lr = self.cfg.lr
            else:
                lr = self.cfg.lr * 3
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, target_model.parameters()), lr=lr,
                                        momentum=0.9)
            with torch.no_grad():
                z, t_ = target_model.get_feature(self.main_p, only_vec=False)
                z_bias, t_b = target_model.get_feature(self.bias_p, only_vec=False)
                z_mix, t_m = target_model.get_feature(self.mix_p, only_vec=False)
                em_tar0 = target_model(z, t_)
                em_bias0 = target_model(z_bias, t_b)
                em_mix0 = target_model(z_mix, t_m)

            mix_len, bias_len = em_mix0.size(0), em_bias0.size(0)
            for epo in range(txt_iters):
                equ_loss, distil_loss, target_loss, equa_loss, pgd_loss = dc(sub), dc(sub), dc(sub), dc(sub), dc(sub)
                optimizer.zero_grad()

                em_tar = target_model(z, t_)
                # em_bias = target_model(z_bias, t_b)
                # em_mix = target_model(z_mix, t_m)
                for wz_, z0_ in zip(em_tar, em_tar0):
                    distil_loss += F.mse_loss(wz_, z0_).to(device)  # * 10

                for i, cls_emb in enumerate(em_tar):  # binary
                    uq_pair = [v[2] for v in self.mapping.values() if v[1] == i]
                    for j, (f, s) in enumerate(combi(uq_pair, 2)):
                        equa_loss += (cls_emb @ em_mix0[i * bias_len + f] - cls_emb @ em_mix0[i * bias_len + s]) ** 2
                eq_loss = equa_loss / (i + 1) / (j + 1) * 500

                if iter:
                    logit = 100 * debias_img_set @ em_tar.t()
                    pgd_loss = F.cross_entropy(logit, batch_y_set)
                    if not just_tun:
                        z_adv = perturb_bafa_txt(z, target_model, em_bias0, t_)
                        # z_adv = perturb_bafa_img2txt(z, target_model, debias_img_set, self.ori_bias_imgs.half(), t_,
                        #                              batch_y_set, self.att_mode)
                        em_tar_adv = target_model(z_adv, t_)
                        logit_adv = 100 * debias_img_set @ em_tar_adv.t()
                        pgd_loss += F.cross_entropy(logit_adv, batch_y_set)
                    loss = eq_loss + pgd_loss + distil_loss
                else:
                    # ind_ = torch.eye(bias_len, dtype=torch.bool)
                    # mix_wrong1 = (em_mix[:bias_len] @ em_mix0[bias_len:].T)[~ind_]
                    # mix_wrong2 = (em_mix0[:bias_len] @ em_mix[bias_len:].T)[~ind_]  # bias is not same
                    # mix_right = (em_mix[:bias_len] @ em_mix[bias_len:].T)[ind_]  # bias is same

                    # indices_3 = torch.tensor([[0] * bias_len + [1] * bias_len, [1] * bias_len + [0] * bias_len],
                    #                          device=device, dtype=torch.bool)
                    # tm_wrong = (em_tar @ em_mix.T)[indices_3]
                    # tm_right = (em_tar @ em_mix0.T)[~indices_3]

                    # match_loss = max(max(mix_wrong1 + mix_wrong2) - tm_right.min(), 0)

                    # z_adv = perturb_bafa_txt(z, target_model, em_bias0, t_)
                    # em_tar_adv = target_model(z_adv, t_)
                    # s = em_tar @ em_tar_adv.t()
                    # mask = torch.eye(s.size(0), device=device, dtype=s.dtype)
                    # pgd_loss = F.mse_loss(s * mask, torch.ones_like(s) * mask) * 100

                    # z_adv = perturb_bafa_img2txt(z, target_model, debias_img_set, self.ori_bias_imgs.half(), t_,  batch_y_set, self.att_mode)
                    # em_tar_adv = target_model(z_adv, t_)
                    # logit = 100 * debias_img_set @ em_tar_adv.t()
                    # pgd_loss = F.cross_entropy(logit, batch_y_set)

                    loss = eq_loss + distil_loss  # + match_loss
                loss.backward()
                optimizer.step()
                # target_model.zero_grad()
                print(f'txt {epo} epoch |' 'eq_loss : ', round(eq_loss.item(), 4), 'pgd_loss:',
                      round(pgd_loss.item(), 4), 'dist_loss:', round(distil_loss.item(), 4))
                if not epo % 5:
                    if self.model_save and not epo % 5:
                        name = f'grad_{self.data}'
                        os.makedirs(name, exist_ok=True)
                        torch.save(target_model.state_dict(), f'{name}/txt_{epo}.pth')
                    if epo >= 4:
                        avg_acc, robust_acc, groups_acc = self.test()
            if epo >= 5:
                self.txt_model.eval()
                z, t_ = self.txt_model.get_feature(self.main_p, only_vec=False)
                self.debias_txt = self.txt_model(z, t_)
                if just_tun:
                    epo = f'just_tun_{epo}'
                torch.save(self.txt_model.state_dict(),
                           f'save_temp/{self.learn_mode}_{self.txt_learn_mode}_{txt_iters}_txt_{iter}_{epo}.pth')
        else:
            self.txt_model.load_state_dict(torch.load(filenames))  #
            z, t_ = self.txt_model.get_feature(self.main_p, only_vec=False)
            self.debias_txt = self.txt_model(z, t_)
            print('txt', iter, '<-', txt_iters, 'way')
            self.test()

    def txt_tuning(self, img_iters):
        pass

    def test(self):
        self.img_model.eval(), self.txt_model.eval()
        img_embeds, gt = [], []
        for _, _, _, labels_test_s, labels_test_y_gt in self.test_info:
            labels_test_s = labels_test_s
        with torch.no_grad():
            # txt
            z, t_ = self.txt_model.get_feature(self.main_p, only_vec=False)
            debias_txt_em = self.txt_model(z, t_)
            txt_em = self.bias_txt_model.get_embedding(self.main_p)  # eval
            debias_vl = F.normalize(torch.matmul(txt_em.cpu(), self.P.T), dim=-1).to(self.device)
            # img
            for batch_x, batch_y, _ in self.test_loader:
                img_em = self.img_model(self.img_model.get_feature(batch_x.to(self.device)))
                img_embeds.append(img_em)
                gt += batch_y.tolist()
            # predict = self.uo.get_zeroshot_predictions(torch.cat(img_embeds).float(), txt_em.to(self.device), self.cfg,
            #                                            temperature=100.)
            # predict_isis = self.uo.get_zeroshot_predictions(torch.cat(img_embeds).float(), debias_vl, self.cfg,
            #                                                 temperature=100.)
            predict_our = self.uo.get_zeroshot_predictions(torch.cat(img_embeds).float(), debias_txt_em.float(),
                                                           self.cfg, temperature=100.)
            # evaluate_clip(predict, labels_test_y_gt, labels_test_s, verbose=True)
            # evaluate_clip(predict_isis, labels_test_y_gt, labels_test_s, verbose=True)
            avg_acc, robust_acc, groups_acc = evaluate_clip(predict_our, labels_test_y_gt, labels_test_s, verbose=True)
            return avg_acc, robust_acc, groups_acc

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
        return [float(correct[:k].reshape(-1).sum(0, keepdim=True).cpu().numpy()) for k in topk]

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
        from test_utils.zs import Cifar_test
        self.txt_model.eval(), self.img_model.eval()
        cifar_test_ = Cifar_test(dc(self.txt_model), dc(self.img_model), self.cfg, self.uo.base_transform)
        cifar_test_.cifar_test()

    def imagenet_test(self):
        from test_utils.zs import Imagenet_test
        # Define paths to validation images and annotations
        img_dir = "../../ILSVRC/Data/CLS-LOC/val/"
        anno_dir = "../../ILSVRC/Annotations/CLS-LOC/val/"

        from utils.imagenet_info import templates, class_names
        imagenet_test_ = Imagenet_test(dc(self.text_model), dc(self.image_model), self.cfg)
        imagenet_test_.imagenet_test(img_dir, anno_dir, class_names, templates)

    def sud_loss(self, outputs, teacher_outputs, ori_logit, y_anchor, best_group, bias_group, beta=0.8, mask=None):
        if mask is not None:
            outputs, teacher_outputs = outputs[mask], teacher_outputs[mask]
            ori_logit, y_anchor = ori_logit[mask], y_anchor[mask]
        sub = torch.tensor(0, device=self.cfg.device)
        ce_loss, mse_loss = dc(sub).half(), dc(sub).half()
        if torch.any(best_group):
            feat_num = outputs[best_group].size(0)
            for i in range(feat_num):
                mse_loss += F.mse_loss(outputs[best_group][i], teacher_outputs[best_group][i], reduction="none").mean()

        if torch.any(bias_group):
            ce_loss = F.cross_entropy(ori_logit[bias_group], y_anchor[bias_group])
        sud_loss = mse_loss * (beta) + ce_loss * (1. - beta)
        if torch.isnan(sud_loss).any():
            print('*' * 50, "Tensor contains NaN values.")
        return sud_loss, mse_loss * (beta), ce_loss * (1. - beta)


class Use_original:
    def __init__(self, base_model, base_transform, get_embeddings, get_dataset_embeddings, get_zeroshot_predictions,
                 cfg):
        self.base_model = base_model.to(cfg.device)
        self.tokenizer = clip.tokenize
        self.base_transform = base_transform  # img
        self.get_embeddings = get_embeddings
        self.get_dataset_embeddings = get_dataset_embeddings
        self.get_zeroshot_predictions = get_zeroshot_predictions
        self.cfg = cfg

    def get_embedding(self, x):
        x = self.get_embeddings(x, self.base_model, self.cfg, normalize=True, verbose=False)
        return x


class SetTarget(nn.Module):
    # img_encoder = uo.base_model.visual(.transformer.resblocks) - out : SetTarget(img)
    # txt_encoder = uo.base_model(.transformer.resblocks) - out : SetTarget.encode_text(txt)
    def __init__(self, cfg, encoder, target_layer_num=1, modal='img', learn_mode='linear'):
        super(SetTarget, self).__init__()
        self.encoder = encoder
        self.dtype = self.encoder.dtype if modal == 'txt' else self.encoder.conv1.weight.dtype
        self.learn_mode = learn_mode
        for param in self.encoder.parameters():  # freeze model
            param.requires_grad = False
        if learn_mode == 'proj':
            if modal == 'img':
                self.encoder.proj.requires_grad = True
            elif modal == 'txt':
                self.encoder.text_projection.requires_grad = True
        elif learn_mode == 'linear':
            for param in self.encoder.transformer.resblocks[-target_layer_num:].parameters():
                param.requires_grad = True
        elif learn_mode == 'lora':
            from trainer.peft import TargetLoRA
            for i in range(-target_layer_num, 0):
                block = self.encoder.transformer.resblocks[i]
                block.attn = TargetLoRA(block.attn, r=4)
                for name, param in block.named_parameters():
                    if 'lora_A' not in name and 'lora_B' not in name:
                        param.requires_grad = False
                    if 'ln' not in name and param.dtype == torch.float32:
                        param.data = param.data.to(torch.float16)

            # from trainer.cliplora import TargetLoRA
            # for i in range(-target_layer_num, 0):
            #     block = self.encoder.transformer.resblocks[i]
            #     block.attn = TargetLoRA(dc(block.attn), enable_lora=['q', 'k', 'v', 'o'], r=4, lora_alpha=2)
            #     for name, param in block.named_parameters():
            #         if 'lora_A' not in name and 'lora_B' not in name:
            #             param.requires_grad = False
            #         if 'ln' not in name and param.dtype == torch.float32:
            #             param.data = param.data.to(torch.float16)


        elif learn_mode == 'vpt':
            from trainer.peft import TargetVPT
            for i in range(-target_layer_num, 0):
                self.encoder.transformer.resblocks[i] = TargetVPT(self.encoder.transformer.resblocks[i], prompt_size=10)
                # for name, param in block.named_parameters():
                #     if 'prompt_embed' not in name:
                #         param.requires_grad = False
        # for name, param in block.named_parameters():
        #     print(name, ':', param.requires_grad)
        self.target_layer_num = target_layer_num
        self.device = cfg.device
        self.modal = modal

    def tokenizer(self, text):
        text_tokens = clip.tokenize(text)
        return text_tokens

    def txt_process(self, txt):
        with torch.no_grad():
            text_tokens = self.tokenizer(txt).to(self.device)
            self.token_max = text_tokens.argmax(dim=-1)
            x = self.encoder.token_embedding(text_tokens.to(self.device)).type(self.dtype)
            x = x + self.encoder.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)
        return x, self.token_max

    def img_process(self, img):
        with torch.no_grad():
            x = self.encoder.conv1(img)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
            x = torch.cat(
                [self.encoder.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                        dtype=x.dtype, device=x.device), x], dim=1)
            # shape = [*, grid ** 2 + 1, width]
            x = x + self.encoder.positional_embedding.to(x.dtype)
            x = self.encoder.ln_pre(x)
            x = x.permute(1, 0, 2)  # NLD -> LND
        return x

    def get_feature(self, input, only_vec=True):
        token_max = None
        with torch.no_grad():
            if self.modal == 'txt':
                x, token_max = self.txt_process(input)
            else:
                x = self.img_process(input.type(self.dtype))
            if self.learn_mode == 'proj':  # only proj [ISIS]
                for layer in self.encoder.transformer.resblocks:
                    x = layer(x)
                x = x.permute(1, 0, 2)
                x = self.encoder.ln_post(x[:, 0, :]) if self.modal == 'img' else self.encoder.ln_final(x).type(
                    self.dtype)
            else:  # grad
                for layer in self.encoder.transformer.resblocks[:-self.target_layer_num]:
                    x = layer(x)
        return x.detach() if only_vec else (x.detach(), token_max)

    def forward(self, z, token_max=None, got_feature=False):
        if not self.learn_mode == 'proj':
            for layer in self.encoder.transformer.resblocks[-self.target_layer_num:]:
                z = layer(z)
            z = z.permute(1, 0, 2)  # LND -> NLD

            z = self.encoder.ln_post(z[:, 0, :]) if self.modal == 'img' else self.encoder.ln_final(z).type(self.dtype)
            if got_feature:
                return z

        if self.modal == 'img':
            out = z @ self.encoder.proj
        else:
            if token_max is not None:
                out = z[torch.arange(z.shape[0]), token_max] @ self.encoder.text_projection
            else:
                out = z[torch.arange(z.shape[0]), self.token_max] @ self.encoder.text_projection
        out = out / out.norm(dim=-1, keepdim=True)
        return out

    def txt_inference(self, txt):
        self.encoder.eval()
        text_tokens = self.tokenizer(txt)
        with torch.no_grad():
            text_tokens = text_tokens.to(self.device)
            text_embeddings = self.encoder.encode_text(text_tokens).float()
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        return text_embeddings

    def img_inference(self, img):
        self.encoder.eval()
        with torch.no_grad():
            img_embeddings = self.encoder(img.type(self.dtype)).float()
            img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
        return img_embeddings
