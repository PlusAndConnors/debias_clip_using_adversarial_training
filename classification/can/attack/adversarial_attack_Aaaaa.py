import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


class PGD():
    def __init__(self, model=None, bound=None, step=None, iters=None, norm=False, random_start=False, discrete=True,
                 device=None, P=None, P_info=None, no_label=False, **kwargs):
        self.model, self.norm, self.device, self.discrete = model, norm, device, discrete
        self.bound, self.step, self.iter = bound, step, iters
        self.rand = random_start
        self.P, self.no_label = P, no_label
        if P_info is not None:
            self.U, self.S, self.VT = P_info

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

    def perturb_bias_p(self, x, query_embeddings, target_y=None, model=None, group=None, beta=0.4, **kwargs):
        bound, step, iters, device = self.bound, self.step, self.iter, self.device
        model.zero_grad()
        debais_well = self.P.clone().detach().to(self.device)
        x_nat = self.inverse_normalize(x.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand:
            rand_perturb_dist = torch.distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, self.inverse_normalize(x_nat),
                                 bound=bound, inverse_normalized=True)
        ori_pred = model(x).float()
        ori_pred = ori_pred / ori_pred.norm(dim=-1, keepdim=True)

        debais_well_q = torch.mm(query_embeddings, debais_well.float())
        bias_well = torch.eye(debais_well.shape[0], device=device) - debais_well.float()
        bias_well_q = torch.mm(query_embeddings, bias_well)

        ori_debais_logit = F.softmax(100 * ori_pred @ debais_well_q.T, dim=-1)
        ori_logit = F.softmax(100 * ori_pred @ query_embeddings.T, dim=-1)
        ori_bias_logit = F.softmax(100 * ori_pred @ bias_well_q.T, dim=-1)

        for i in range(iters):  # TODO argmax need to debias Y. So before argmax please use P0 Sigma-1
            adv_pred = model(x_adv)  # 64 256 -> 64 10
            adv_pred = adv_pred / adv_pred.norm(dim=-1, keepdim=True)

            # x_set
            adv_debais_logit = F.log_softmax(100 * adv_pred @ debais_well_q.T, dim=-1)
            adv_ori_logit = F.log_softmax(100 * adv_pred @ query_embeddings.T, dim=-1)
            adv_bias_logit = F.log_softmax(100 * adv_pred @ bias_well_q.T, dim=-1)

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
                    # loss = F.cross_entropy(adv_ori_logit[group], target_y[group]) - F.cross_entropy(adv_debais_logit[group], target_y[group])
                    loss = 2 * F.cross_entropy(adv_ori_logit[group], target_y[group]) - F.cross_entropy(adv_debais_logit[group], target_y[group])
                else:
                    loss = F.cross_entropy(adv_ori_logit, target_y) - F.cross_entropy(adv_debais_logit, target_y)
                    #loss = 2 * F.cross_entropy(adv_ori_logit, target_y) - F.cross_entropy(adv_debais_logit, target_y)
            loss.backward(retain_graph=True)

            grad_sign = x_adv.grad.data.detach().sign()
            x_adv = self.inverse_normalize(x_adv) + grad_sign * step  # a sign( d( L (x_adv)))
            x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)
            model.zero_grad()

        return x_adv.detach().to(device)

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
