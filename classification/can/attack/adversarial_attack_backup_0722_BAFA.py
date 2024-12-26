import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions


class PGD():
    def __init__(self, model=None, bound=None, step=None, iters=None, norm=False, random_start=False, discrete=True,
                 device=None, P=None, P_info=None, no_label=False, using=None, mode=None, learn_mode=None, **kwargs):
        self.model, self.norm, self.device, self.discrete = model, norm, device, discrete
        self.bound, self.step, self.iter = bound, step, iters
        self.rand = random_start
        self.P, self.no_label = P, no_label
        if P_info is not None:
            self.U, self.S, self.VT = P_info
        if using is not None:
            self.using = using
        try:
            self.bias = (torch.eye(self.P.size(0)) - self.P).to(device)
            self.bias_em = None
            self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        except AttributeError:
            pass
        self.clamp_delta = nn.Parameter(torch.zeros(1024))
        self.mode = mode
        self.learn_mode = learn_mode

    def clamper_prompt(self, x_adv, x_nat, bound=None, metric="inf", inverse_normalized=False):
        if not inverse_normalized:
            x_adv = self.inverse_normalize(x_adv)
            x_nat = self.inverse_normalize(x_nat)
        clamp_delta = torch.clamp(self.clamp_delta, -self.bound, self.bound)
        x_adv = x_nat + clamp_delta
        x_adv = torch.clamp(x_adv, 0., 1.)
        return self.normalize(self.discretize(x_adv)).clone().detach().requires_grad_(True)

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
        # x_adv = torch.clamp(x_adv, 0., 1.)
        # x_adv = self.discretize(x_adv)
        return x_adv.clone().detach().requires_grad_(True)

    def kl_loss_(self, outputs, target, T=1):
        kl_loss = nn.KLDivLoss()(F.log_softmax(outputs / T, dim=1), F.softmax(target / T, dim=1))
        return kl_loss

    def rbf_kernel(self, X, sigma=1.0):
        pairwise_dists = torch.cdist(X, X, p=2) ** 2
        return torch.exp(-pairwise_dists / (2 * sigma ** 2))

    def perturb_bafa(self, x, query_embeddings, debais_well_q=None, target_y=None, model=None, y_best_group=None, mode=None,
                     **kwargs):
        bound, step, iters, device = self.bound, self.step, self.iter, self.device
        if mode is None:
            mode = self.mode
        model.zero_grad()
        x_nat = self.inverse_normalize(x.detach().clone().to(device))
        x_adv = x.detach().clone().requires_grad_(True).to(device)
        if self.rand:
            rand_perturb_dist = torch.distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, self.inverse_normalize(x_nat),
                                 bound=bound, inverse_normalized=True)
        ori_pred = model(x).float()
        ori_pred = ori_pred / ori_pred.norm(dim=-1, keepdim=True)
        if not debais_well_q is None:
            ori_logit_debias = 100 * ori_pred @ debais_well_q.T
        ori_logit = 100 * ori_pred @ query_embeddings.T

        for i in range(iters):  # TODO argmax need to debias Y. So before argmax please use P0 Sigma-1
            adv_pred = model(x_adv).float()
            adv_pred = adv_pred / adv_pred.norm(dim=-1, keepdim=True)

            with torch.no_grad():
                adv_pred_t = model(x_adv).float()
                adv_pred_t = adv_pred_t / adv_pred_t.norm(dim=-1, keepdim=True)


            adv_ori = 100 * adv_pred @ query_embeddings.T
            if not debais_well_q is None:
                adv_debais = 100 * adv_pred @ debais_well_q.T
                chat_adv = 100 * adv_pred @ (query_embeddings - debais_well_q).T

            # y_set
            if self.no_label:
                target_y = torch.argmax(ori_logit, dim=-1)
                term = torch.mean(chat_adv ** 2) / chat_adv.size(0)
                loss = (F.cross_entropy(adv_ori[y_best_group], target_y[y_best_group])
                        - F.cross_entropy(adv_debais[y_best_group], target_y[y_best_group])) + term

            else:
                if mode is None:
                    loss = F.cross_entropy(adv_ori, target_y)
                    # loss = F.cross_entropy(chat_adv, target_y) - F.cross_entropy(adv_debais, target_y)
                    # loss = F.cross_entropy(adv_debais, target_y) - F.cross_entropy(chat_adv, target_y)
                elif mode == 'bafa':
                    loss = F.cross_entropy(adv_ori, target_y) - F.cross_entropy(adv_debais, target_y)
                elif mode == 'bafa_kl':
                    loss_kl = F.kl_div(F.softmax(adv_debais, dim=1).log(), F.softmax(ori_logit_debias, dim=1),
                                       reduction='batchmean')
                    loss = F.cross_entropy(adv_ori, target_y) - loss_kl
                elif mode == 'bafa2':
                    # loss = F.cross_entropy(chat_adv, target_y) - F.cross_entropy(adv_debais, target_y)
                    loss = F.cross_entropy(adv_ori, target_y) - 0.5 * F.cross_entropy(adv_debais, target_y)
                elif mode == 'mse':
                    mse_loss = 0
                    feat_num = adv_pred.size(1)
                    for i in range(feat_num):
                        mse_loss += F.mse_loss(adv_pred[:, i], ori_pred[:, i], reduction="none").mean()
                    loss = F.cross_entropy(adv_ori, target_y) - mse_loss
                elif mode == 'bafa_mse':
                    mse_loss = 0
                    feat_num = adv_pred.size(1)
                    for i in range(feat_num):
                        mse_loss += F.mse_loss(adv_pred[:, i], ori_pred[:, i], reduction="none").mean()
                    loss = F.cross_entropy(adv_ori, target_y) - F.cross_entropy(adv_debais, target_y) - mse_loss
                elif mode == 'bafa2_mse':
                    mse_loss = 0
                    feat_num = adv_pred.size(1)
                    for i in range(feat_num):
                        mse_loss += F.mse_loss(adv_pred[:, i], ori_pred[:, i], reduction="none").mean()
                    loss = F.cross_entropy(adv_ori, target_y) - 0.1 * F.cross_entropy(adv_debais, target_y) - mse_loss
                elif mode == 'kd':
                    # mse
                    mse_loss = 0
                    feat_num = adv_pred.size(1)
                    for i in range(feat_num):
                        mse_loss += F.mse_loss(adv_pred[:, i], ori_pred[:, i], reduction="none").mean()
                    #
                    kl_loss = nn.KLDivLoss()(F.log_softmax(adv_debais / 4, dim=1),
                                             F.softmax(ori_logit_debias / 4, dim=1)) * (4 * 4 * 2.0)
                    ce_loss = F.cross_entropy(adv_ori, target_y)
                    print(ce_loss, kl_loss, mse_loss)
                    loss = ce_loss - 0.5 * kl_loss - 0.1 * mse_loss
                elif mode == 'mse_term2':
                    log_sum_de = torch.log(torch.sum(torch.exp(adv_debais), dim=1, keepdim=True))
                    log_sum_cha = torch.log(torch.sum(torch.exp(chat_adv), dim=1, keepdim=True))
                    term2 = log_sum_cha.mean() / log_sum_de.mean()
                    mse_loss = 0
                    feat_num = adv_pred.size(1)
                    for i in range(feat_num):
                        mse_loss += F.mse_loss(adv_pred[:, i], ori_pred[:, i], reduction="none").mean()
                    loss = F.cross_entropy(adv_debais, target_y) - mse_loss + term2
                elif mode == 'ori':
                    loss = F.cross_entropy(adv_ori, target_y)
                # loss = F.cross_entropy(chat_adv, target_y) - F.cross_entropy(adv_debais, target_y) + 1/mahalanobis_loss
                # loss = F.cross_entropy(chat_adv, target_y) - F.cross_entropy(adv_debais, target_y) + mahalanobis_loss

            loss.backward(retain_graph=True)
            grad_sign = x_adv.grad.data.detach().sign()
            x_adv = self.inverse_normalize(x_adv) + grad_sign * step  # a sign( d( L (x_adv)))
            x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)  # , metric=1,2
            model.zero_grad()

        return x_adv.detach().to(device)

    def perturb_txt(self, pre_embedding, target_image_set, target_y=None, model=None, bound=None, mode=None,
                    num_sample=10, **kwargs):
        if bound is None:
            bound = self.bound
        step, iters, device = self.step, self.iter, self.device
        if mode is None:
            mode = self.mode
        I = target_image_set.detach()
        target_y = target_y.to(device)
        model.zero_grad()
        x_nat = self.inverse_normalize(pre_embedding.detach().clone().to(device))
        x_adv = pre_embedding.detach().clone().requires_grad_(True).to(device)

        # ori_pred = model(pre_embedding).float()
        # ori_pred = ori_pred / ori_pred.norm(dim=-1, keepdim=True)
        # ori_logit = 100 * I @ ori_pred.T

        txt_adv_list = []
        for j in range(num_sample):
            if self.rand:
                rand_perturb_dist = torch.distributions.uniform.Uniform(-bound, bound)
                rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
                x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, self.inverse_normalize(x_nat),
                                     bound=bound, inverse_normalized=True)
            for i in range(iters):  # TODO argmax need to debias Y. So before argmax please use P0 Sigma-1
                adv_pred = model(x_adv).float()
                adv_pred = adv_pred / adv_pred.norm(dim=-1, keepdim=True)
                adv_logit = 100 * I @ adv_pred.T
                loss = F.cross_entropy(adv_logit, target_y)
                loss.backward(retain_graph=True)
                grad_sign = x_adv.grad.data.detach().sign()
                x_adv = self.inverse_normalize(x_adv) + grad_sign * step  # a sign( d( L (x_adv)))
                x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)  # metric=1,2
                model.zero_grad()

            txt_adv_list.append(x_adv.detach().to(device))
        return torch.cat(txt_adv_list, dim=0)

    def perturb_img(self, img_embedding, target_txt_set, target_y=None, model=None, y_best_group=None, mode=None,
                    num_sample=10, **kwargs):
        bound, step, iters, device = self.bound, self.step, self.iter, self.device
        if mode is None:
            mode = self.mode
        T = target_txt_set.detach()
        target_y = target_y.to(device)
        model.zero_grad()
        x_nat = self.inverse_normalize(img_embedding.detach().clone().to(device))
        x_adv = img_embedding.detach().clone().requires_grad_(True).to(device)

        # ori_pred = model(pre_embedding).float()
        # ori_pred = ori_pred / ori_pred.norm(dim=-1, keepdim=True)
        # ori_logit = 100 * I @ ori_pred.T

        if self.rand:
            rand_perturb_dist = torch.distributions.uniform.Uniform(-bound, bound)
            rand_perturb = rand_perturb_dist.sample(sample_shape=x_adv.shape).to(device)
            x_adv = self.clamper(self.inverse_normalize(x_adv) + rand_perturb, self.inverse_normalize(x_nat),
                                 bound=bound, inverse_normalized=True)
        for i in range(iters):  # TODO argmax need to debias Y. So before argmax please use P0 Sigma-1
            adv_pred = model(x_adv).float()
            adv_pred = adv_pred / adv_pred.norm(dim=-1, keepdim=True)
            adv_logit = 100 * adv_pred @ T.T
            loss = F.cross_entropy(adv_logit, target_y)
            loss.backward(retain_graph=True)
            grad_sign = x_adv.grad.data.detach().sign()
            x_adv = self.inverse_normalize(x_adv) + grad_sign * step  # a sign( d( L (x_adv)))
            x_adv = self.clamper(x_adv, x_nat, bound=bound, inverse_normalized=True)  # metric=1,2
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

# # Each pair of logits is treated as mix binary classification problem
# logits_pairs = torch.cat((adv_mix_logit[group][:, :5].unsqueeze(2), adv_mix_logit[group][:, 5:].unsqueeze(2)),
#                          dim=2)  # Shape (128, 5, 2)
# logits_flat = logits_pairs.view(-1, 2)  # Shape (640, 2)
# labels_expanded = target_y[group].repeat_interleave(5)
# new_loss = F.cross_entropy(logits_flat, labels_expanded)
# loss += new_loss
