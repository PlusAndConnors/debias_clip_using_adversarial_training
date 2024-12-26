import build_kernel as bk
import torch
from tqdm import tqdm
import adversarial_training as at
import torch.nn.functional as F
from copy import deepcopy as dc
from can.attack.debias_vl import debais_vl_s_c, debias_vl
from can.attack.adversarial_attack import PGD
from torch import nn
from can.utils.tools import get_parameter_count


class AlternatingOptimizer:
    def __init__(self, opts, uo=None):
        self.mode = opts.mode
        if self.mode == 'base':
            self.image_model = dc(uo)
            self.text_model = dc(uo)
        elif self.mode == 'debais_vl':
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

        if self.mode == 'base' or self.mode == 'debais_vl':
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

    def info(self, text_embeddings, Y_I):
        self.text_embeddings = text_embeddings
        self.y_binary = ((Y_I + 1) / 2)[:, 1].int()
        self.X_T = text_embeddings[self.y_binary]
        self.debias_text_embedding = self.text_model.encod(self.text_embeddings)

    def kd_loss(self, outputs, labels, teacher_outputs, mask, alpha=0.5, T=1):
        kl_loss = self.kl_loss_(outputs, teacher_outputs, T)
        ce_loss = F.cross_entropy(outputs[mask], labels[mask])
        KD_loss = kl_loss * (alpha * T * T) + ce_loss * (1. - alpha)
        return KD_loss, kl_loss, ce_loss

    def kl_loss_(self, outputs, teacher_outputs, T=1):
        return nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(teacher_outputs / T, dim=1))

    def init_set(self):
        self.image_model.eval()
        self.student_layer = self.image_model.target_layer  # 1024*768
        print(f'unlearn target parameter : {get_parameter_count(self.student_layer)}')

    def train(self, num_iters):
        logit_scale, batch, device, check_unlearned = 100, 128, self.cfg.device, True
        student_layer = self.student_layer
        optimizer = torch.optim.SGD(student_layer.parameters(), lr=self.cfg.lr, momentum=0.9)
        for epo in range(num_iters):
            ce_loss_track, kl_loss_track, kl_loss_track2 = 0, 0, 0
            teacher_layer = dc(student_layer)

            # adv = PGD(teacher_layer, 0.4, 0.5, 5, False, True, True, device, self.P)
            adv = PGD(teacher_layer, 0.4, 0.5, 5, False, True, True, device,
                      self.P, no_label=self.no_label)
            optimizer.zero_grad()

            for step, (batch_x, batch_y, _) in enumerate(self.train_loader):

                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                x = self.image_model.get_img_feature(batch_x)

                # original feature : target
                teacher_layer.eval()
                ori_feature_t = teacher_layer(x)
                ori_feature_s = student_layer(x)

                ori_logit_debias = logit_scale * ori_feature_t.float() @ self.debias_text_embedding.t()
                ori_logit_base = logit_scale * ori_feature_s.float() @ self.text_embeddings.t()
                ori_logit_y_base = torch.argmax(ori_logit_base, dim=-1)
                ori_logit_y_debias = torch.argmax(ori_logit_debias, dim=-1) if self.no_label else batch_y

                # 1. best_group to be bias_group using attack for robust training <- bias attack
                best_group = ori_logit_y_debias == ori_logit_y_base  # matrix un-aware bias
                # 2. logit of bias group to be debias logit using kl_d
                bias_group = ori_logit_y_debias != ori_logit_y_base  # matrix aware bias - mixed

                # attacked feature : bias gradient augmentation
                x_adv = adv.perturb_bias_p(x, self.text_embeddings, target_y=batch_y, model=teacher_layer,
                                           device=device, group=best_group)
                adv_feature = student_layer(x_adv)
                adv_logit = logit_scale * adv_feature.float() @ self.text_embeddings.t()
                adv_logit_debias = logit_scale * adv_feature.float() @ self.debias_text_embedding.t()
                adv_logit_y = torch.argmax(adv_logit, dim=-1)
                adv_logit_debias_y = torch.argmax(adv_logit_debias, dim=-1)
                # max_conf, pseudo_labels = torch.max(F.softmax(ori_logit, dim=1), dim=1)
                # batch_y = batch_y if self.label else pseudo_labels # if pseudo label

                # 3. logit of attacked group to make bias
                att_bias_group = (ori_logit_y_debias[best_group] != adv_logit_y[best_group]) == (
                            ori_logit_y_debias[best_group] == adv_logit_debias_y[best_group])

                if self.no_label:
                    # confident_mask = max_conf >= confidence_threshold
                    # simclr_loss(ori_feature, adv_feature)
                    loss, kl_loss, ce_loss = self.kd_loss(adv_logit_debias, ori_logit_y_debias, ori_logit_debias,
                                                          best_group, 0.4)
                    kl_loss1, kl_loss2 = torch.tensor(0, device=device), torch.tensor(0, device=device)
                    if torch.any(bias_group):
                        kl_loss1 = self.kl_loss_(ori_logit_base[bias_group], ori_logit_debias[bias_group])
                    if torch.any(att_bias_group):
                        kl_loss2 = self.kl_loss_(adv_logit[best_group][att_bias_group],
                                                 ori_logit_debias[best_group][att_bias_group])
                    kl_loss = kl_loss1 + kl_loss2
                    kl_loss_track += kl_loss1.item()
                    kl_loss_track2 += kl_loss2.item()
                    loss = ce_loss + kl_loss
                else:
                    # CE loss
                    ce_loss1, ce_loss2 = torch.tensor(0, device=device), torch.tensor(0, device=device)
                    if torch.any(att_bias_group):
                        ce_loss1 = F.cross_entropy(adv_logit[best_group][att_bias_group], batch_y[best_group][att_bias_group])
                    if torch.any(bias_group):
                        ce_loss2 = F.cross_entropy(ori_feature_s[bias_group], batch_y[bias_group])
                    ce_loss = ce_loss1 + 0.1 * ce_loss2
                    if not ce_loss:
                        continue
                    print('ce loss1', ce_loss1.item(), 'ce loss2', ce_loss2.item())
                    ce_loss_track += ce_loss.item()
                    # KL loss
                    kl_loss1, kl_loss2 = torch.tensor(0, device=device), torch.tensor(0, device=device)
                    if torch.any(bias_group):
                        kl_loss1 = self.kl_loss_(ori_logit_base[bias_group], ori_logit_debias[bias_group])
                    if torch.any(att_bias_group):
                        kl_loss2 = self.kl_loss_(adv_logit[best_group][att_bias_group],
                                                 ori_logit_debias[best_group][att_bias_group])
                    kl_loss = kl_loss1 + kl_loss2
                    kl_loss_track += kl_loss1.item()
                    kl_loss_track2 += kl_loss2.item()
                    loss = ce_loss  # + kl_loss
                loss.backward(retain_graph=True)
                optimizer.step()
                self.image_model.target_layer = student_layer
                student_layer.zero_grad()

            # Check that unlearning is done.
            if not epo % 3:
                for step, (batch_x, batch_y, _) in enumerate(self.val_loader):
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    x = self.image_model.get_img_feature(batch_x)
                    teacher_layer = dc(student_layer)
                    teacher_layer.eval()
                    ori_feature = teacher_layer(x)
                    if torch.isnan(ori_feature).any():
                        print('*' * 50, "Tensor contains NaN values.")
                    ori_logit = torch.argmax(logit_scale * ori_feature.float() @ self.debias_text_embedding.t(), dim=-1)
                    ori_logit_base = torch.argmax(logit_scale * ori_feature.float() @ self.text_embeddings.t(), dim=-1)
                    mask = ori_logit != ori_logit_base  # == bias group
                    student_layer.zero_grad()
                    check_unlearned += mask.sum()
                    del x
                if not check_unlearned:
                    break
                un_c = int(check_unlearned)
                print(f'biased = {un_c} -- {round(un_c * 100 / self.len_t, 2)}% is biased bias_group')
                check_unlearned = 0
            else:
                print(
                    f'epoch: {epo} | ce loss : {round(ce_loss_track, 2)} | kl_loss1 : {round(kl_loss_track, 2)} | | kl_loss2 : {round(kl_loss_track2, 2)}')
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

    def debias(self):
        spurious_prompt, candidate_prompt, S = debais_vl_s_c(self.cfg)
        candidate_embeddings = self.get_embedding(candidate_prompt)
        spurious_embeddings = self.get_embedding(spurious_prompt)
        P = debias_vl(spurious_embeddings, candidate_embeddings, S)
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
