import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy as dc


class PGD():  # (self.cfg, self.att_bnd, self.att_stp, self.att_itr, device, self.att_mode, bias_txts, debias_txt, self.l_scale, self.P)
    def __init__(self, cfg, bound, step, iters, device, att_mode, bias_txt, debias_txt, l_scale, P=None):
        self.bound, self.step, self.iter, self.cfg = bound, step, iters, cfg
        self.device, self.mode, self.l_scale = device, att_mode, l_scale
        self.bias_txt = bias_txt
        torch.manual_seed(cfg.seed)
        self.debias_txt = debias_txt
        self.my_pgd_set()

    def my_pgd_set(self):
        self.norm, self.rand, self.discrete = False, True, True

    def clamper(self, x_adv, x_nat, bound=None, metric="inf", inverse_normalized=False):
        if not inverse_normalized:
            x_adv = self.inverse_normalize(x_adv)
            x_nat = self.inverse_normalize(x_nat)
        if metric == "inf":
            clamp_delta = torch.clamp(x_adv - x_nat, -bound, bound)
        else:
            clamp_delta = x_adv - x_nat
            for batch_index in range(clamp_delta.size(0)):
                image_delta = clamp_delta[batch_index]
                image_norm = image_delta.norm(p=metric, keepdim=False)
                if image_norm > bound:
                    clamp_delta[batch_index] /= image_norm
                    clamp_delta[batch_index] *= bound
        x_adv = x_nat + clamp_delta
        return x_adv.clone().detach().requires_grad_(True)

    def perturb_bafa(self, z, target_model, target_y=None, mix=None):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        batch = z_adv.size(0)
        # if self.rand:
        #     rand_perturb_dist = torch.distributions.uniform.Uniform(-self.bound, self.bound)
        #     rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(self.device)
        #     z_adv = self.clamper(self.inverse_normalize(z_adv) + rand_perturb, self.inverse_normalize(z_nat),
        #                          bound=self.bound, inverse_normalized=True)  # .half()
        out_feat = target_model(z)
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        # # out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)  <- check
        # bias_logit = self.l_scale * out_feat.float() @ self.bias_txt.t()
        # debias_logit = self.l_scale * out_feat.float() @ self.debias_txt.t()
        # bias_predict, debias_predict = torch.argmax(bias_logit, dim=-1), torch.argmax(debias_logit, dim=-1)
        for i in range(self.iter):  # TODO argmax need to debias Y. So before argmax please use P0 Sigma-1
            adv_feat = target_model(z_adv.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)

            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T

            if self.cfg.att_mode == 'bafa':
                loss = F.cross_entropy(adv_bias, target_y) - F.cross_entropy(adv_debias, target_y)
            elif self.cfg.att_mode == 'txt_guidance':
                logits = torch.stack([
                    (adv_feat @ mix[:4].t()).mean(dim=1), (adv_feat @ mix[4:].t()).mean(dim=1),  # land | water
                ]).view(batch, -1)
                loss = F.cross_entropy(logits, target_y)  # landbird is landbackground, waterbird is water background
                loss -= F.mse_loss(ori_feat, adv_debias) * 10 / self.l_scale
            else:
                loss = F.cross_entropy(adv_bias, target_y)
            # loss = F.cross_entropy(adv_bias, target_y) - F.cross_entropy(adv_debias, target_y)
            loss.backward(retain_graph=True)

            grad_sign = z_adv.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv = self.clamper(z_adv_new, z_nat, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv.grad = None
        return z_adv.detach().to(self.device).half()

    def perturb_bafa_nolabel(self, z, target_model):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        # if self.rand:
        #     rand_perturb_dist = torch.distributions.uniform.Uniform(-self.bound, self.bound)
        #     rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(self.device)
        #     z_adv = self.clamper(self.inverse_normalize(z_adv) + rand_perturb, self.inverse_normalize(z_nat),
        #                          bound=self.bound, inverse_normalized=True)  # .half()

        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            batch = adv_feat.size(0)
            label = torch.zeros(batch).type(torch.LongTensor).to(self.device)
            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T

            loss = F.cross_entropy(adv_bias, label) - F.cross_entropy(adv_debias, label)
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        for i in range(self.iter):
            adv_feat = target_model(z_adv2.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)

            label = torch.ones(batch).type(torch.LongTensor).to(self.device)
            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T
            loss = F.cross_entropy(adv_bias, label) - F.cross_entropy(adv_debias, label)
            loss.backward(retain_graph=True)

            grad_sign = z_adv2.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv2) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv2 = self.clamper(z_adv_new, z_nat2, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv2.grad = None

        return z_adv1.detach().to(self.device).half(), z_adv2.detach().to(self.device).half()

    def perturb_bafa_nolabel_mix(self, z, target_model, mix=None):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)

        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        out_feat = target_model(z)
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        batch = ori_feat.size(0)
        label = torch.zeros(batch).type(torch.LongTensor).to(self.device)

        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)

            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T

            loss = F.cross_entropy(adv_bias, label) - F.cross_entropy(adv_debias, label)
            if mix is not None:
                logits = torch.stack([
                    (adv_feat @ mix[:4].t()).mean(dim=1), (adv_feat @ mix[4:].t()).mean(dim=1),  # land | water
                ]).view(batch, -1)
                entropy_loss = F.cross_entropy(logits, label)
                loss += entropy_loss
                loss -= F.mse_loss(ori_feat, adv_debias) / self.l_scale
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        label = torch.ones(batch).type(torch.LongTensor).to(self.device)
        for i in range(self.iter):
            adv_feat = target_model(z_adv2.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)

            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T
            loss = F.cross_entropy(adv_bias, label) - F.cross_entropy(adv_debias, label)
            if mix is not None:
                logits = torch.stack([
                    (adv_feat @ mix[:4].t()).mean(dim=1), (adv_feat @ mix[4:].t()).mean(dim=1),  # land | water
                ]).view(batch, -1)
                loss += F.cross_entropy(logits, label)
                loss -= F.mse_loss(ori_feat, adv_debias) / self.l_scale
            loss.backward(retain_graph=True)

            grad_sign = z_adv2.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv2) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv2 = self.clamper(z_adv_new, z_nat2, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv2.grad = None

        return z_adv1.detach().to(self.device).half(), z_adv2.detach().to(self.device).half()

    def perturb_bafa_nolabel_use_predict(self, z, target_model, mix=None, batch_y=None):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)

        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        out_feat = target_model(z)
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T
        batch = ori_feat.size(0)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)

            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T

            # loss = F.cross_entropy(adv_bias, batch_y) - F.cross_entropy(adv_debias, batch_y)
            if mix is not None:
                logits = torch.stack([(adv_feat @ mix[:4].t()).mean(dim=1), (adv_feat @ mix[4:].t()).mean(dim=1),
                                      ]).view(batch, -1) # land | water
                loss = F.cross_entropy(logits, batch_y)
            loss -= F.mse_loss(ori_feat, adv_debias) / self.l_scale
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        return z_adv1.detach().to(self.device).half()

    def perturb_bafa_nolabel_label(self, z, target_model, label):
        target_model.zero_grad()
        z_nat = self.inverse_normalize(z.detach().clone().to(self.device))  # .half()
        z_adv = z.detach().clone().requires_grad_(True).to(self.device)
        # if self.rand:
        #     rand_perturb_dist = torch.distributions.uniform.Uniform(-self.bound, self.bound)
        #     rand_perturb = rand_perturb_dist.sample(sample_shape=z_adv.shape).to(self.device)
        #     z_adv = self.clamper(self.inverse_normalize(z_adv) + rand_perturb, self.inverse_normalize(z_nat),
        #                          bound=self.bound, inverse_normalized=True)  # .half()
        out_feat = target_model(z)
        out_feat = out_feat / out_feat.norm(dim=-1, keepdim=True)
        ori_feat = self.l_scale * out_feat @ self.debias_txt.T

        z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
        for i in range(self.iter):
            adv_feat = target_model(z_adv1.half())
            adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
            label = label.type(torch.LongTensor).to(self.device)
            adv_bias = self.l_scale * adv_feat @ self.bias_txt.T
            adv_debias = self.l_scale * adv_feat @ self.debias_txt.T

            loss = F.cross_entropy(adv_bias, label) - F.cross_entropy(adv_debias, label)
            loss -= F.mse_loss(ori_feat, adv_debias) / self.l_scale
            loss.backward(retain_graph=True)

            grad_sign = z_adv1.grad.data.detach().sign()
            z_adv_new = self.inverse_normalize(z_adv1) + grad_sign * self.step  # a sign( d( L (x_adv)))
            z_adv1 = self.clamper(z_adv_new, z_nat1, bound=self.bound, inverse_normalized=True)  # , metric=1,2
            target_model.zero_grad()
            z_adv1.grad = None

        return z_adv1.detach().to(self.device).half()

    def base_perturb(self, x, y, target_y=None, model=None, bound=None, step=None, iters=None, x_nat=None, device=None,
                     **kwargs):
        criterion = self.CE
        model = model or self.model
        bound = bound or self.bound
        step = step or self.step
        iters = iters or self.iter
        device = device or self.device

        model.zero_grad()
        if x_nat is None:
            x_nat = self.inverse_normalize(x.detach().clone().to(device))
        else:
            x_nat = self.inverse_normalize(x_nat.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand:
            rand_perturb_dist = torch.distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, self.inverse_normalize(x_nat),
                                 bound=bound, inverse_normalized=True)
        for i in range(iters):
            adv_pred = model(x_adv)  # 64 256 -> 64 10
            loss = criterion(adv_pred, y)
            loss.backward(retain_graph=True)

            grad_sign = x_adv.grad.data.detach().sign()
            x_adv = self.inverse_normalize(x_adv) + grad_sign * step  # a sign( d( L (x_adv)))
            x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)
            model.zero_grad()

        return x_adv.detach().to(device)

    def normalize(self, x):
        return x

    def inverse_normalize(self, x):
        return x

    def discretize(self, x):
        return torch.round(x * 255) / 255


def perturb_bafa_txt(z, target_model, mix, t_):
    target_model.zero_grad()
    device = z.device
    z_nat = z.detach().clone().to(device)  # .half()
    z_adv = z.detach().clone().requires_grad_(True).to(device)

    z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
    att_bnd, att_stp, att_itr = 0.4, 5e-3, 5
    for i in range(att_itr):
        adv_feat = target_model(z_adv1.half(), t_)
        adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
        logits = torch.stack([
            (adv_feat[0] @ mix[:4].t()).mean(), (adv_feat[0] @ mix[4:].t()).mean(),
            (adv_feat[1] @ mix[:4].t()).mean(), (adv_feat[1] @ mix[4:].t()).mean()
        ]).view(2, 2)
        loss = F.cross_entropy(logits, torch.tensor([0, 1], device='cuda'))
        loss.backward(retain_graph=True)

        grad_sign = z_adv1.grad.data.detach().sign()
        z_adv_new = z_adv1 + grad_sign * att_stp  # a sign( d( L (x_adv)))
        z_adv1 = clamper(z_adv_new, z_nat1, bound=att_bnd)  # , metric=1,2
        target_model.zero_grad()
        z_adv1.grad = None

    return z_adv1.detach().to(device).half()


def perturb_bafa_img2txt(z, target_model, img_set, ori_bias_imgs, mix, t_):
    target_model.zero_grad()
    device = z.device
    z_nat = z.detach().clone().to(device)  # .half()
    z_adv = z.detach().clone().requires_grad_(True).to(device)

    z_adv1, z_adv2, z_nat1, z_nat2 = dc(z_adv), dc(z_adv), dc(z_nat), dc(z_nat)
    att_bnd, att_stp, att_itr, batch = 0.3, 1e-3, 5, img_set.size(0)

    label1 = torch.zeros(batch).type(torch.LongTensor).to(z.device)
    label2 = torch.ones(batch).type(torch.LongTensor).to(z.device)
    with torch.no_grad():
        ori_em = target_model(z, t_)
        ori_em = ori_em / ori_em.norm(dim=-1, keepdim=True)

    for i in range(att_itr):
        adv_feat = target_model(z_adv1.half(), t_)
        adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
        logits = 100 * img_set @ adv_feat.t()
        ori_logits = 100 * ori_bias_imgs @ adv_feat.t()
        # loss = F.cross_entropy(ori_logits, label1) - F.cross_entropy(logits, label1)  # all middle this right, nobias_p left
        logits = torch.stack([
            (adv_feat[0] @ mix[:4].t()).mean(), (adv_feat[0] @ mix[4:].t()).mean(),
            (adv_feat[1] @ mix[:4].t()).mean(), (adv_feat[1] @ mix[4:].t()).mean()
        ]).view(2, 2)
        loss = F.cross_entropy(logits, torch.tensor([0, 1], device='cuda'))
        loss -= F.mse_loss(adv_feat, ori_em)
        loss.backward(retain_graph=True)

        grad_sign = z_adv1.grad.data.detach().sign()
        z_adv_new = z_adv1 + grad_sign * att_stp  # a sign( d( L (x_adv)))
        z_adv1 = clamper(z_adv_new, z_nat1, bound=att_bnd)  # , metric=1,2
        target_model.zero_grad()
        z_adv1.grad = None

    for i in range(att_itr):
        adv_feat = target_model(z_adv2.half(), t_)
        adv_feat = adv_feat / adv_feat.norm(dim=-1, keepdim=True)
        logits = 100 * img_set @ adv_feat.t()
        ori_logits = 100 * ori_bias_imgs @ adv_feat.t()
        # loss = F.cross_entropy(ori_logits, label2) - F.cross_entropy(logits, label2)
        logits = torch.stack([
            (adv_feat[0] @ mix[:4].t()).mean(), (adv_feat[0] @ mix[4:].t()).mean(),
            (adv_feat[1] @ mix[:4].t()).mean(), (adv_feat[1] @ mix[4:].t()).mean()
        ]).view(2, 2)
        loss = F.cross_entropy(logits, torch.tensor([0, 1], device='cuda'))
        loss -= F.mse_loss(adv_feat, ori_em)
        loss.backward(retain_graph=True)

        grad_sign = z_adv2.grad.data.detach().sign()
        z_adv_new = z_adv2 + grad_sign * att_stp  # a sign( d( L (x_adv)))
        z_adv2 = clamper(z_adv_new, z_nat2, bound=att_bnd)  # , metric=1,2
        target_model.zero_grad()
        z_adv2.grad = None

    return z_adv1.detach().to(device).half(), z_adv2.detach().to(device).half()


def clamper(x_adv, x_nat, bound=None, metric="inf"):
    if metric == "inf":
        clamp_delta = torch.clamp(x_adv - x_nat, -bound, bound)
    else:
        clamp_delta = x_adv - x_nat
        for batch_index in range(clamp_delta.size(0)):
            image_delta = clamp_delta[batch_index]
            image_norm = image_delta.norm(p=metric, keepdim=False)
            if image_norm > bound:
                clamp_delta[batch_index] /= image_norm
                clamp_delta[batch_index] *= bound
    x_adv = x_nat + clamp_delta
    return x_adv.clone().detach().requires_grad_(True)


def simclr_loss(features1, features2, temperature=0.1):
    """
    Computes the SimCLR loss between two sets of features.

    Args:
    features1: torch.Tensor - Tensor of shape (batch_size, feature_dim)
    features2: torch.Tensor - Tensor of shape (batch_size, feature_dim)
    temperature: float - Temperature parameter for scaling the logits

    Returns:
    loss: torch.Tensor - Computed SimCLR loss
    """
    # Normalize the features
    features1 = F.normalize(features1, dim=-1)
    features2 = F.normalize(features2, dim=-1)

    # Concatenate features for the similarity matrix computation
    features = torch.cat([features1, features2], dim=0)

    # Compute the similarity matrix
    similarity_matrix = torch.matmul(features, features.T) / temperature

    # Create labels for the contrastive learning
    batch_size = features1.shape[0]
    labels = torch.arange(batch_size, device=features1.device)
    labels = torch.cat([labels, labels], dim=0)

    # Mask to remove self-similarity
    mask = torch.eye(labels.shape[0], dtype=torch.bool, device=features1.device)
    similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))

    # Compute the positive pair loss
    positives = torch.cat([torch.diag(similarity_matrix, batch_size), torch.diag(similarity_matrix, -batch_size)],
                          dim=0)
    positives = positives.view(2 * batch_size, 1)

    # Compute the cross-entropy loss
    loss = -torch.log(torch.exp(positives) / torch.exp(similarity_matrix).sum(dim=1, keepdim=True)).mean()
    return loss


def get_knn_avg_dist(features1: torch.Tensor, features2: torch.Tensor, knn: int = 10,
                     **_: torch.Tensor) -> torch.Tensor:
    # get the top-k nearest neighbors
    scores = features1 @ features2.T
    topk_distances = scores.topk(int(knn), dim=1, largest=True, sorted=True)[0]
    # get the average distance
    average_dist = topk_distances.mean(dim=1)
    return average_dist


def bias_ce_loss(adv_pred, device, bias_label=True):
    if len(adv_pred.shape) == 3:
        batch, bias_num, label = adv_pred.shape
        cross_entropy_results = torch.zeros(batch, bias_num, device=device, dtype=torch.float)
        # if attack for 1, else training for 0
        y_ = torch.ones(batch, device=device, dtype=torch.long) if bias_label else torch.zeros(batch, device=device,
                                                                                               dtype=torch.long)
        for bias_ in range(bias_num):
            # Select the logits for the i-th kind of output feature
            logits_i = adv_pred[:, bias_, :]

            # Compute cross-entropy for the selected logits
            cross_entropy_results[:, bias_] = F.cross_entropy(logits_i, y_, reduction='none')
    else:
        batch, label = adv_pred.shape
        max_indices = torch.argmax(adv_pred, dim=1).long()
        one_hot_vectors = torch.eye(6, device=device)[max_indices]

        cross_entropy_results = F.cross_entropy(adv_pred, y_, reduction='none')

    return cross_entropy_results


def orthogonal_projection(basis):
    proj = torch.inverse(torch.matmul(basis.T, basis))
    proj = torch.matmul(basis, proj)
    proj = torch.matmul(proj, basis.T)
    proj = torch.eye(basis.shape[0]).to(basis.device) - proj
    return proj

# # Each pair of logits is treated as mix binary classification problem
# logits_pairs = torch.cat((adv_mix_logit[group][:, :5].unsqueeze(2), adv_mix_logit[group][:, 5:].unsqueeze(2)),
#                          dim=2)  # Shape (128, 5, 2)
# logits_flat = logits_pairs.view(-1, 2)  # Shape (640, 2)
# labels_expanded = target_y[group].repeat_interleave(5)
# new_loss = F.cross_entropy(logits_flat, labels_expanded)
# loss += new_loss
