import torch
from tqdm import tqdm
import adversarial_training as at
import torch.nn.functional as F
from copy import deepcopy as dc
from can.attack.debias_vl import debais_vl_s_c, debias_vl, bias_vl
from can.attack.adversarial_attack import PGD
from torch import nn
from can.utils.tools import get_parameter_count
from transformers import CLIPProcessor, CLIPModel
from peft import get_peft_model, LoraConfig

class Unlearn:
    def __init__(self, opts, uo, train_loader_base, val_loader_base, get_zeroshot_predictions, no_label=False):
        self.mode = opts.mode
        self.cfg = opts
        self.P = uo.debias()

        model = Rebuilding(opts)
        self.text_model = dc(model)
        self.image_model = dc(model)

        self.train_loader = train_loader_base
        self.val_loader = val_loader_base
        self.len_t = len(val_loader_base.dataset)

        self.get_zeroshot_predictions = get_zeroshot_predictions
        self.no_label = no_label
        self.bias_sample_check = True

    def info(self, text_embeddings, Y_I):
        self.text_embeddings = text_embeddings
        self.y_binary = ((Y_I + 1) / 2)[:, 1].int()
        self.X_T = text_embeddings[self.y_binary]
        self.debias_text_embedding = self.text_model.encod(self.text_embeddings)
        self.bias_text_embedding = self.text_model.get_bias_text(self.text_embeddings)
        self.rebuilding_img()

    def kd_loss(self, outputs, labels, teacher_outputs, mask, alpha=0.5, T=1):
        kl_loss = self.kl_loss_(outputs, teacher_outputs, T)
        ce_loss = F.cross_entropy(outputs[mask], labels[mask])
        KD_loss = kl_loss * (alpha * T * T) + ce_loss * (1. - alpha)
        return KD_loss, kl_loss, ce_loss

    def kl_loss_(self, outputs, teacher_outputs, T=1):
        return nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1))
    def train(self, num_iters):
        method = 'proj'  # 'info' # 'covar' # 'triplet'
        logit_scale, batch, device, check_unlearned = 100, 128, self.cfg.device, True
        student_layer = self.student_layer
        optimizer = torch.optim.SGD(student_layer.parameters(), lr=self.cfg.lr, momentum=0.9)
        bias_em, mix = self.text_model.spurious_embeddings.to(device), self.text_model.candidate_embeddings.to(device)
        for epo in range(num_iters):
            ce_loss_att_track, ce_loss_bias_track, kl_loss_att_track, kl_loss_bias_track = 0, 0, 0, 0
            teacher_layer = dc(student_layer)

            # adv = PGD(teacher_layer, 0.4, 0.5, 5, False, True, True, device, self.P)
            adv = PGD(teacher_layer, 0.6, 10, 10, False, True, True, device,
                      self.P, no_label=self.no_label, using=self.text_model)
            optimizer.zero_grad()
            for step, (batch_x, batch_y, _) in enumerate(self.train_loader):

                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                x = self.image_model.get_img_feature(batch_x)
                teacher_layer.eval()

                ori_feature_t = teacher_layer(x)
                ori_feature_s = student_layer(x)

                ori_logit_debias = logit_scale * ori_feature_t.float() @ self.debias_text_embedding.t()
                ori_logit_debias_s = logit_scale * ori_feature_s.float() @ self.debias_text_embedding.t()
                ori_logit_base = logit_scale * ori_feature_s.float() @ self.text_embeddings.t()
                ori_logit_base_t = logit_scale * ori_feature_t.float() @ self.text_embeddings.t()
                ori_logit_y_base = torch.argmax(ori_logit_base, dim=-1)
                ori_logit_y_debias = torch.argmax(ori_logit_debias, dim=-1) if self.no_label else batch_y
                # self.covariance(ori_logit_base, ori_logit_debias)

                # ori_logit_bias = logit_scale * ori_feature_t.float() @ self.bias_text_embedding.t()

                # ori_logit_base = logit_scale * ori_feature_s.float() @ self.text_embeddings.t()

                text_info = self.text_embeddings[batch_y]
                # self.HSIC(ori_feature_s, text_info)
                if self.no_label:
                    # 1. best_group to be bias_group using attack for robust training <- bias attack
                    best_group = ori_logit_y_debias == ori_logit_y_base  # matrix un-aware bias
                    # 2. logit of bias group to be debias logit using kl_d
                    bias_group = ori_logit_y_debias != ori_logit_y_base  # matrix aware bias - mixed

                    # attacked feature : bias gradient augmentation
                    x_adv = adv.perturb_bias_p(x, self.text_embeddings, self.debias_text_embedding, target_y=batch_y,
                                               model=teacher_layer, device=device, group=best_group)
                    adv_feature = student_layer(x_adv)

                    adv_logit = logit_scale * adv_feature.float() @ self.text_embeddings.t()
                    adv_logit_debias = logit_scale * adv_feature.float() @ self.debias_text_embedding.t()
                    adv_logit_y = torch.argmax(adv_logit, dim=-1)
                    adv_logit_debias_y = torch.argmax(adv_logit_debias, dim=-1)

                    # 3. logit of attacked group to make bias
                    att_bias_group = (ori_logit_y_base[best_group] != adv_logit_y[best_group]) == (
                            ori_logit_y_debias[best_group] == adv_logit_debias_y[best_group])

                    ce_loss_att, ce_loss_bias = torch.tensor(0, device=device), torch.tensor(0, device=device)
                    if torch.any(att_bias_group):
                        ce_loss_att = F.cross_entropy(adv_logit[best_group][att_bias_group],
                                                      batch_y[best_group][att_bias_group])
                    # ce_loss1 = F.cross_entropy(adv_logit[best_group], batch_y[best_group])
                    if torch.any(bias_group):
                        ce_loss_bias = F.cross_entropy(ori_logit_base[bias_group], batch_y[bias_group])
                    if torch.any(best_group):
                        ce_loss_base = self.kl_loss_(ori_logit_base[best_group], ori_logit_debias[best_group])

                    # 4. kd_loss
                    loss, kl_loss, ce_loss = self.kd_loss(adv_logit_debias, ori_logit_y_debias, ori_logit_debias,
                                                          best_group, 0.4)
                    kl_loss_att, kl_loss_bias = torch.tensor(0, device=device), torch.tensor(0, device=device)
                    if torch.any(att_bias_group):
                        kl_loss_att = self.kl_loss_(adv_logit[best_group][att_bias_group],
                                                    ori_logit_debias[best_group][att_bias_group])
                    if torch.any(bias_group):
                        kl_loss_bias = self.kl_loss_(ori_logit_base[bias_group], ori_logit_debias[bias_group])

                    kl_loss = kl_loss_att + kl_loss_bias
                    kl_loss_att_track += kl_loss_att.item()
                    kl_loss_bias_track += kl_loss_bias.item()
                    loss = ce_loss  # + kl_loss
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
                    ce = torch.tensor(0, device=device).float()
                    ce_loss_att, ce_loss_bias, ce_loss_base = ce.clone(), ce.clone(), ce.clone()
                    if torch.any(att_bias_group):
                        ce_loss_att = F.cross_entropy(adv_logit[best_group][att_bias_group],
                                                      batch_y[best_group][att_bias_group])
                        ce_loss_base1 = F.cross_entropy(ori_logit_base[best_group][att_bias_group],
                                                        batch_y[best_group][att_bias_group])
                        ce_loss_base += ce_loss_base1
                    if torch.any(att_bias_group_fail):
                        ce_loss_base2 = F.cross_entropy(ori_logit_base[best_group][att_bias_group_fail],
                                                        batch_y[best_group][att_bias_group_fail])
                        ce_loss_base += ce_loss_base2 * 0.5
                    # ce_loss1 = F.cross_entropy(adv_logit[best_group], batch_y[best_group])
                    if torch.any(bias_group):
                        ce_loss_bias = F.cross_entropy(ori_logit_base[bias_group], batch_y[bias_group])
                    # if torch.any(best_group):
                    #     ce_loss_base = F.cross_entropy(ori_logit_debias_s[best_group], batch_y[best_group])
                    ce_loss = (ce_loss_att + ce_loss_bias + ce_loss_base) * 0.1
                    if not ce_loss:
                        continue
                    # print('ce loss1', ce_loss1.item(), 'ce loss2', ce_loss2.item())
                    ce_loss_att_track += ce_loss_att.item()
                    ce_loss_bias_track += ce_loss_bias.item()
                    # KL loss
                    kl_loss_att, kl_loss_bias = torch.tensor(0, device=device), torch.tensor(0, device=device)
                    if torch.any(att_bias_group):
                        kl_loss_att = self.kl_loss_(adv_logit[best_group][att_bias_group],
                                                    ori_logit_debias[best_group][att_bias_group])
                    if torch.any(bias_group):
                        kl_loss_bias = self.kl_loss_(ori_logit_base[bias_group], ori_logit_debias[bias_group])

                    kl_loss = kl_loss_att + kl_loss_bias
                    kl_loss_att_track += kl_loss_att.item()
                    kl_loss_bias_track += kl_loss_bias.item()
                    loss = ce_loss  # + kl_loss
                loss.backward(retain_graph=True)
                optimizer.step()
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
                f' | kl_loss_att_track : {round(kl_loss_att_track, 2)} | | kl_loss_bias_track : {round(kl_loss_bias_track, 2)}')
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
        spurious_prompt, candidate_prompt, S, B, text = debais_vl_s_c(self.cfg)
        self.candidate_embeddings = self.get_embedding(candidate_prompt)
        self.spurious_embeddings = self.get_embedding(spurious_prompt)
        P = debias_vl(self.spurious_embeddings, self.candidate_embeddings, S)
        self.Pb = bias_vl(self.spurious_embeddings, self.candidate_embeddings, B).to(self.cfg.device)
        self.P = P.to(self.cfg.device)
        return P


class Rebuilding(nn.Module):
    def __init__(self, cfg=None):
        super(Rebuilding, self).__init__()
        self.vis_mode = cfg.load_base_model.lower()
        self.target_layer = None
        model_url = "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
        model_path = "ViT-L-14.pt"
        torch.hub.download_url_to_file(model_url, model_path)

        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        print(f'unlearn target parameter : {get_parameter_count(self.student_layer)}')
        embed_dim = model.text_model.embeddings.position_embeddings.embedding_dim
        prompt_tuning = PromptTuning(embed_dim)
        model.text_model.embeddings.forward = lambda inputs_embeds: prompt_tuning(inputs_embeds)
        self.image_encoder = model.visual
        self.dtype = model.dtype
        self.set_target()

    def set_target(self):
        if 'vit' in self.vis_mode:
            self.target_layer = Target(self.image_encoder.proj, 'vit')
            self.image_encoder.proj = None
        else:
            self.target_layer = Target(self.image_encoder.attnpool)
            self.image_encoder.attnpool = nn.Identity()

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


class PromptTuning(torch.nn.Module):
    def __init__(self, embed_dim, prompt_length=10):
        super().__init__()
        self.prompt_length = prompt_length
        self.prompt_embeddings = torch.nn.Parameter(torch.randn(prompt_length, embed_dim))

    def forward(self, inputs_embeds):
        batch_size = inputs_embeds.shape[0]
        prompt_embeds = self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([prompt_embeds, inputs_embeds], dim=1)


