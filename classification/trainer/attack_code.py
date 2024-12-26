import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


class PGD():
    def __init__(self, model=None, bound=None, step=None, iters=None, norm=False, random_start=True, discrete=True,
                 device=None, P=None, P_info=None, no_label=False, using=None, **kwargs):
        self.model, self.norm, self.device, self.discrete = model, norm, device, discrete
        self.bound, self.step, self.iter = bound, step, iters
        self.rand = random_start
        self.P, self.no_label = P, no_label
        if P_info is not None:
            self.U, self.S, self.VT = P_info
        if using is not None:
            self.using = using
        self.bias = (torch.eye(self.P.size(0)) - self.P).to(device)
        self.bias_em = None

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
        x_adv = torch.clamp(x_adv, 0., 1.)
        return self.normalize(self.discretize(x_adv)).clone().detach().requires_grad_(True)

    def kl_loss_(self, outputs, target, T=1):
        kl_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(target / T, dim=1))
        return kl_loss

    def rbf_kernel(self, X, sigma=1.0):
        pairwise_dists = torch.cdist(X, X, p=2) ** 2
        return torch.exp(-pairwise_dists / (2 * sigma ** 2))

    def hsic_loss(self, X, Y, sigma=1.0):
        m = X.size(0)
        H = torch.eye(m, device=X.device) - (1 / m) * torch.ones(m, m, device=X.device)
        K = self.rbf_kernel(X, sigma)
        L = self.rbf_kernel(Y, sigma)
        HSIC = torch.trace(K @ H @ L @ H) / (m - 1) ** 2
        return HSIC

    def perturb_bias_p(self, x, query_embeddings, debais_well_q, target_y=None, model=None, group=None, beta=0.4,
                       method='proj', **kwargs):
        bound, step, iters, device = self.bound, self.step, self.iter, self.device
        model.zero_grad()
        bias_em, mix = self.using.spurious_embeddings.to(device), self.using.candidate_embeddings.to(device)

        self.bias_em = self.using.get_bias_text(query_embeddings) if self.bias_em is None else self.bias_em
        num_aug, num_cls, dim = self.bias_em.shape
        bias_em = self.bias_em.view(num_aug * num_cls, dim)
        x_nat = self.inverse_normalize(x.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand:
            rand_perturb_dist = torch.distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, self.inverse_normalize(x_nat),
                                 bound=bound, inverse_normalized=True)
        ori_pred = model(x).float()
        ori_pred = ori_pred / ori_pred.norm(dim=-1, keepdim=True)

        ori_debais_logit = F.softmax(100 * ori_pred @ debais_well_q.T, dim=-1)
        ori_logit = F.softmax(100 * ori_pred @ query_embeddings.T, dim=-1)

        ori_bias_em_logit = 100 * ori_pred @ bias_em.T
        ori_bias_em_logit = ori_bias_em_logit.view(ori_bias_em_logit.size(0), num_aug, num_cls)
        ori_bias_em_logit = ori_bias_em_logit.view(ori_pred.size(0) * num_aug, num_cls)
        expanded_GT_class = target_y.unsqueeze(1).expand(-1, num_aug).contiguous().view(-1)

        # bias_em_logit_y = torch.argmax(F.softmax(bias_em_logit, dim=-1), dim=-1)

        ori_mix_logit_debias = 100 * ori_pred @ torch.matmul(mix, self.P.to(device).T).T
        ori_mix_logit_bias = 100 * ori_pred @ torch.matmul(mix, self.bias.to(device).T).T
        ori_mix_logit_debias_y = torch.argmax(F.softmax(ori_mix_logit_debias, dim=-1), dim=-1)

        for i in range(iters):  # TODO argmax need to debias Y. So before argmax please use P0 Sigma-1
            adv_pred = model(x_adv)  # 64 256 -> 64 10
            adv_pred = adv_pred / adv_pred.norm(dim=-1, keepdim=True)

            # x_set
            adv_debais = 100 * adv_pred @ debais_well_q.T
            adv_debais_logit = F.log_softmax(adv_debais, dim=-1)
            adv_ori = 100 * adv_pred @ query_embeddings.T
            adv_ori_logit = F.log_softmax(adv_ori, dim=-1)

            bias_em_logit = 100 * ori_pred @ bias_em.T
            bias_em_logit = bias_em_logit.view(bias_em_logit.size(0), num_aug, num_cls)
            bias_em_logit = bias_em_logit.view(ori_pred.size(0) * num_aug, num_cls)
            # adv_bias_em_logit = 100 * adv_pred @ bias_em.T
            # adv_mix_logit_debias = 100 * adv_pred @ torch.matmul(mix, self.P.to(device).T).T
            # adv_mix_logit_bias = 100 * adv_pred @ torch.matmul(mix, self.bias.to(device).T).T

            # y_set
            if self.no_label:
                # loss = (self.kl_loss_(adv_bias_logit, ori_bias_logit) + self.kl_loss_(adv_ori_logit, ori_logit)
                #         - 2 * self.kl_loss_(adv_debais_logit, ori_debais_logit))
                # loss = (-self.kl_loss_(adv_bias_logit, ori_bias_logit) + -self.kl_loss_(adv_ori_logit, ori_logit)
                #         + 2 * self.kl_loss_(adv_debais_logit, ori_debais_logit))
                # loss = self.kl_loss_(adv_ori_logit, ori_logit) * beta - self.kl_loss_(adv_debais_logit, ori_debais_logit) * (1 - beta)
                loss = (- self.kl_loss_(adv_ori_logit, ori_logit) * beta
                        + self.kl_loss_(adv_debais_logit, ori_debais_logit) * (1 - beta))
            else:
                # torch.argmax(ori_pred @ query_embeddings.T, dim=1)
                if group is not None:
                    if method == 'proj':
                        loss = (F.cross_entropy(adv_ori[group], target_y[group])
                                - F.cross_entropy(adv_debais[group], target_y[group]))
                        # loss = (F.cross_entropy(bias_em_logit, expanded_GT_class)
                        #         - F.cross_entropy(adv_debais[group], target_y[group]))
                    elif method == 'covar':
                        loss = F.cross_entropy(adv_ori[group], target_y[group]) - F.cross_entropy(adv_debais[group],
                                                                                                  target_y[group])
                    elif method == 'info':
                        loss = F.cross_entropy(adv_ori[group], target_y[group]) - F.cross_entropy(adv_debais[group],
                                                                                                  target_y[group])
                    elif method == 'triplet':
                        loss = F.cross_entropy(adv_ori[group], target_y[group]) - F.cross_entropy(adv_debais[group],
                                                                                                  target_y[group])
                else:
                    loss = F.cross_entropy(adv_ori_logit, target_y) - F.cross_entropy(adv_debais_logit, target_y)
                    # loss = 2 * F.cross_entropy(adv_ori_logit, target_y) - F.cross_entropy(adv_debais_logit, target_y)
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