import numpy as np

train_list = ['Actor', 'Architect', 'Audiologist', 'Author', 'Baker', 'Barber', 'Blacksmith', 'Bricklayer',
              'Bus Driver', 'Butcher', 'Chef', 'Chemist', 'Cleaner', 'Coach', 'Comedian', 'Computer Programmer',
              'Construction Worker', 'Consultant', 'Counselor', 'Dancer', 'Dentist', 'Designer', 'Dietitian', 'DJ',
              'Doctor', 'Driver', 'Economist', 'Electrician', 'Engineer', 'Entrepreneur', 'Farmer', 'Florist',
              'Graphic Designer', 'Hairdresser', 'Historian', 'Journalist', 'Judge', 'Lawyer', 'Librarian', 'Magician',
              'Makeup Artist', 'Mathematician', 'Marine Biologist', 'Mechanic', 'Model', 'Musician', 'Nanny', 'Nurse',
              'Optician', 'Painter', 'Pastry Chef', 'Pediatrician', 'Photographer', 'Plumber', 'Police Officer',
              'Politician', 'Professor', 'Psychologist', 'Real Estate Agent', 'Receptionist', 'Recruiter', 'Researcher',
              'Sailor', 'Salesperson', 'Surveyor', 'Singer', 'Social Worker', 'Software Developer', 'Statistician',
              'Surgeon', 'Teacher', 'Technician', 'Therapist', 'Tour Guide', 'Translator', 'Vet', 'Videographer',
              'Waiter', 'Writer', 'Zoologist', 'Housekeeper']
test_list = ['Accountant', 'Astronaut', 'Biologist', 'Carpenter', 'Civil Engineer', 'Clerk', 'Detective', 'Editor',
             'Firefighter', 'Interpreter', 'Manager', 'Nutritionist', 'Paramedic', 'Pharmacist', 'Physicist', 'Pilot',
             'Reporter', 'Security Guard', 'Scientist', 'Web Developer']

train_list_background = ['landbird', 'waterbird']
test_list_background = ['landbird', 'waterbird']
bias_list_p = ['land', 'wood', 'forest', 'mountain']
bias_list_n = ['water', 'ocean', 'beach']
spurious_prompt = ['This is a land background.', 'This is a picture of a forest.', 'This is a picture of a mountain.',
                   'This is a picture of a wood.', 'This is a water background.', 'This is a picture of an ocean.',
                   'This is a picture of a beach.', 'This is a picture of a port.']

train_list_glass = ['nerd', 'cheerleader', 'person']
g_bias_list_p = ['female', 'glasses off']
g_bias_list_n = ['male', 'glasses']
retain_prompt = ['a photo of a nerd.', 'a photo of a cheerleader.']
bias_prompt = ['a photo of glasses being worn', 'a photo of glasses', 'a photo of glasses in use',
               'a photo of glasses in action']

templates = ['this is a picture of', 'a photo of ']
retain_content = ['a nerd', 'a basketball player', 'a computer developer', 'a soccer player',
                  'a cheerleader']
bias_content = ['glasses', 'a monitor', 'a keyboard', 'a megaphone', 'a pom-poms', 'a whiteboard',
                'a stack of books']
bias_content = ['glasses']


def get_A(z_i, z_j):
    z_i = z_i[:, None]
    z_j = z_j[:, None]
    return (np.matmul(z_i, z_i.T) + np.matmul(z_j, z_j.T) - np.matmul(z_i, z_j.T) - np.matmul(z_j, z_i.T))


def get_M(embeddings, S):
    d = embeddings.shape[1]
    M = np.zeros((d, d))
    for s in S:
        M += get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(S)


def get_proj_matrix_torch(embeddings):
    U, S, V = torch.svd(embeddings)
    basis = V[:, :len(embeddings)]
    proj = torch.inverse(basis.T @ basis)
    proj = basis @ proj
    proj = proj @ basis.T
    proj = torch.eye(proj.shape[0]).to(embeddings.device) - proj
    return proj


# def create_prompts_and_index_mapping(templates, retain_content, bias_content, interaction_pre=None,
#                                      interaction_post=None):
#     retain_prompt = []
#     bias_prompt = []
#     candidate_prompt = []
#     index_mapping = {}
#
#     for template_idx, template in enumerate(templates):
#         for retain_idx, retain in enumerate(retain_content):
#             retain_prompt.append('{} {}'.format(template, retain))
#
#         for bias_idx, bias in enumerate(bias_content):
#             if interaction_post is not None:
#                 for inter_po in interaction_post:
#                     bias_prompt.append('{} a {} {}'.format(template, bias, inter_po))
#             else:
#                 bias_prompt.append('{} {}'.format(template, bias))
#         for retain_idx, retain in enumerate(retain_content):
#             for bias_idx, bias in enumerate(bias_content):
#                 for inter_pr in interaction_pre:
#                     if interaction_post is not None:
#                         for inter_po in interaction_post:
#                             prompt = '{} {} {} {} {}'.format(template, retain, inter_pr, bias, inter_po)
#                     else:
#                         prompt = '{} {} {} {}'.format(template, retain, inter_pr, bias)
#                     candidate_prompt.append(prompt)
#                     # Use the index of the new prompt as the key
#                     prompt_index = len(candidate_prompt) - 1
#                     # Store the indices of template, retain, and bias
#                     index_mapping[prompt_index] = [template_idx, retain_idx, bias_idx]
#
#     return retain_prompt, bias_prompt, candidate_prompt, index_mapping

def create_prompts_and_index_mapping(templates, retain_content, bias_content, interaction_pre=None,
                                     interaction_post=None, type=None):
    retain_prompt = []
    bias_prompt = []
    candidate_prompt = []
    index_mapping = {}

    for template_idx, template in enumerate(templates):
        for retain_idx, retain in enumerate(retain_content):
            retain_prompt.append('{} {}'.format(template, retain))

        for bias_idx, bias in enumerate(bias_content):
            if interaction_post is not None:
                for inter_po in interaction_post:
                    bias_prompt.append('{} a {} {}'.format(template, bias, inter_po))
            else:
                if type == 'gender':
                    bias_prompt.append('{} a {}'.format(template, bias))
                else:
                    bias_prompt.append('{} {}'.format(template, bias))
        for retain_idx, retain in enumerate(retain_content):
            for bias_idx, bias in enumerate(bias_content):
                if interaction_pre is not None:
                    for inter_pr in interaction_pre:
                        if interaction_post is not None:
                            for inter_po in interaction_post:
                                prompt = '{} {} {} {} {}'.format(template, retain, inter_pr, bias, inter_po)
                        else:
                            prompt = '{} {} {} {}'.format(template, retain, inter_pr, bias)
                        candidate_prompt.append(prompt)
                        # Use the index of the new prompt as the key
                        prompt_index = len(candidate_prompt) - 1
                        # Store the indices of template, retain, and bias
                        index_mapping[prompt_index] = [template_idx, retain_idx, bias_idx]
                else:
                    if interaction_post is not None:
                        for inter_po in interaction_post:
                            prompt = '{} {} {} {} {}'.format(template, retain, inter_pr, bias, inter_po)
                    else:
                        if type == 'gender':
                            prompt = '{} a {} {}'.format(template, bias, retain.replace('a ', ''))
                        else:
                            prompt = '{} {} {}'.format(template, retain, bias)
                    candidate_prompt.append(prompt)
                    # Use the index of the new prompt as the key
                    prompt_index = len(candidate_prompt) - 1
                    # Store the indices of template, retain, and bias
                    index_mapping[prompt_index] = [template_idx, retain_idx, bias_idx]

    return retain_prompt, bias_prompt, candidate_prompt, index_mapping


# # Example usage
# templates = ['this is a picture of', 'a photo of']
# retain_content = ['a waterbird', 'a land bird']
# bias_content = ['water background', 'land background']

# retain_prompt, bias_prompt, candidate_prompt, index_mapping = create_prompts_and_index_mapping(templates, retain_content, bias_content)
# print("Retain Prompts:")
# for idx, prompt in enumerate(retain_prompt):
#     print(f"{idx}: {prompt}")

# print("\nBias Prompts:")
# for idx, prompt in enumerate(bias_prompt):
#     print(f"{idx}: {prompt}")

# print("\nCandidate Prompts:")
# for idx, prompt in enumerate(candidate_prompt):
#     print(f"{idx}: {prompt}")

# print("\nIndex Mapping:")
# for idx, indices in index_mapping.items():
#     print(f"{idx}: {indices}")

def debias_vl_set(args, got_bias_pair=False):
    # Construct Positive Pair
    candidate_prompt, S, counter = [], [], 0
    if args.bias == 'background':

        for train_cls_i in train_list_background:
            train_cls_i = train_cls_i.lower()

            bias_p_indices = range(counter, counter + len(bias_list_p))
            bias_n_indices = range(counter + len(bias_list_p), counter + len(bias_list_p) + len(bias_list_n))

            for i, p in enumerate(bias_list_p):
                candidate_prompt.append('a photo of a {} with {} background'.format(train_cls_i, p))

            for j, n in enumerate(bias_list_n):
                candidate_prompt.append('a photo of a {} with {} background'.format(train_cls_i, n))

            for i in bias_p_indices:
                for j in bias_n_indices:
                    S.append([i, j])

            num_bias, S_bias = len(bias_n_indices) + len(bias_p_indices), list()
            for i in range(num_bias):
                S_bias.append([i, i + num_bias])

            counter += len(bias_list_p) + len(bias_list_n)
            if got_bias_pair:
                return candidate_prompt, S, S_bias
    elif args.bias == 'glasses':
        for train_cls_i in train_list_glass:
            train_cls_i = train_cls_i.lower()

            bias_p_indices = range(counter, counter + len(g_bias_list_p))
            bias_n_indices = range(counter + len(g_bias_list_p), counter + len(g_bias_list_p) + len(g_bias_list_n))

            for i, p in enumerate(g_bias_list_p):
                candidate_prompt.append('a photo of a {} {} '.format(p, train_cls_i, ))

            for j, n in enumerate(g_bias_list_n):
                candidate_prompt.append('a photo of a {} {}'.format(n, train_cls_i))
            for i in bias_p_indices:
                for j in bias_n_indices:
                    S.append([i, j])

            counter += len(g_bias_list_p) + len(g_bias_list_n)
    else:
        for train_cls_i in train_list:
            train_cls_i = train_cls_i.lower()
            candidate_prompt += ['a photo of a male {}.'.format(train_cls_i),
                                 'a photo of a female {}.'.format(train_cls_i)]
            S += [[counter, counter + 1]]
            counter += 2

    return candidate_prompt, S


def bias_prompt(two_prompt=True):
    prompt, S = [], []
    for p in bias_list_p:
        prompt.append('a photo of a {} background'.format(p))
        if two_prompt:
            prompt.append('this is a picture of a {}'.format(p))

    for n in bias_list_n:
        prompt.append('a photo of a {} background'.format(n))
        if two_prompt:
            prompt.append('this is a picture of a {}'.format(n))
    if two_prompt:
        bias_p_indices = range(0, 2 * len(bias_list_p), 2)
        bias_n_indices = range(2 * len(bias_list_p), 2 * (len(bias_list_p) + len(bias_list_n)), 2)
    else:
        bias_p_indices = range(0, len(bias_list_p))
        bias_n_indices = range(len(bias_list_p), (len(bias_list_p) + len(bias_list_n)))
    for i in bias_p_indices:
        for j in bias_n_indices:
            S.append([i, j])
            if two_prompt:
                S.append([i + 1, j + 1])
    return prompt, S


def bias_prompt_normal():
    prompt, S = [], []
    for p in g_bias_list_p:
        prompt.append('this is a picture of a {}'.format(p))

    for n in g_bias_list_n:
        prompt.append('this is a picture of a {}'.format(n))
    bias_p_indices = range(0, len(g_bias_list_p), 1)
    bias_n_indices = range(len(g_bias_list_p), (len(g_bias_list_p) + len(g_bias_list_n)), 1)
    for i in bias_p_indices:
        for j in bias_n_indices:
            S.append([i, j])
    return prompt, S


import torch


def txt_process(prompt, text_encoder, tokenizer, device, get_token=False):
    input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                      truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = text_encoder(input.input_ids.to(device))[0]  # .cpu().numpy()
    return embeddings if not get_token else (embeddings, input.input_ids.to(device))


import torch.nn.functional as F


def pgd_attack(text_encoder, embeds, z_dif, z_dif_b, check_, epsilon, step, num_iter, device, bias_model=None):
    delta = torch.zeros_like(embeds, requires_grad=True).to(embeds.device)
    text_encoder.zero_grad()
    for _ in range(num_iter):
        outputs = text_encoder(embeds + delta)  # TODO z_re, z_dif, z_dif_b, device
        z_re_adv = F.normalize(outputs[torch.arange(outputs.shape[0]), check_], dim=-1)
        loss = sud_loss_txt(z_re_adv, z_dif, z_dif_b)
        if bias_model is not None:
            with torch.no_grad():
                bias_output = bias_model(embeds + delta)
            z_re_adv_bias = F.normalize(bias_output[torch.arange(bias_output.shape[0]), check_], dim=-1)
            loss = sud_loss_txt(z_re_adv_bias, z_dif) - loss  # clip wrong, target right
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + step * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        text_encoder.zero_grad()
    delta.grad.zero_()
    return delta.detach()


from copy import deepcopy as dc


def pgd_attack_(target_model, tar_p, ret_p, bias_p, mix_p, mix_n, tokenizer, num_iter=10, step=4e-3, bound=0.4):
    device = target_model.device
    with torch.no_grad():
        token_, ret_mask = target_model.tokenizer(tokenizer, tar_p)
        tar_t = token_['input_ids'].argmax(-1)
        token_r, _ = target_model.tokenizer(tokenizer, ret_p)
        ret_t = token_r['input_ids'].argmax(-1)
        token_b, _ = target_model.tokenizer(tokenizer, bias_p)
        bias_t = token_b['input_ids'].argmax(-1)
        z_ret = target_model.text_encoder(token_r.input_ids.to(target_model.device))[0]
        ret_eos = F.normalize(z_ret[torch.arange(z_ret.shape[0]), ret_t], dim=-1)
        z_b = target_model.text_encoder(token_b.input_ids.to(target_model.device))[0]
        zb_eos = F.normalize(z_b[torch.arange(z_b.shape[0]), bias_t], dim=-1)
        z = target_model.get_feature(token_.input_ids, ret_mask)
        z_ = target_model(z)
        z_eos = F.normalize(z_[torch.arange(z_.shape[0]), tar_t], dim=-1)
        anchor_ret_pair = z_eos @ ret_eos.t()

        token_, _ = target_model.tokenizer(tokenizer, mix_p)
        mix_p_t = token_['input_ids'].argmax(-1)
        z_mp = target_model.text_encoder(token_.input_ids.to(target_model.device))[0]
        zmp_eos = F.normalize(z_mp[torch.arange(z_mp.shape[0]), mix_p_t], dim=-1)

        token_, _ = target_model.tokenizer(tokenizer, mix_n)
        mix_n_t = token_['input_ids'].argmax(-1)
        z_np = target_model.text_encoder(token_.input_ids.to(target_model.device))[0]
        zmn_eos = F.normalize(z_np[torch.arange(z_np.shape[0]), mix_n_t], dim=-1)

    sub = torch.tensor(0, device=device).float()
    retain_loss = dc(sub)
    target_model.zero_grad()
    z_nat = z.detach().clone().to(z.device)
    z_adv = z.detach().clone().requires_grad_(True).to(z.device)

    for _ in range(num_iter):
        adv_emb = target_model(z_adv)  # TODO z_re, z_dif, z_dif_b, device
        tar_eos = F.normalize(adv_emb[torch.arange(adv_emb.shape[0]), tar_t], dim=-1)
        retain_pair = tar_eos @ ret_eos.t()
        # for i in range(len(retain_pair)):
        #     retain_loss += F.mse_loss(retain_pair[i], anchor_ret_pair[i])
        for i in range(len(z_eos)):
            retain_loss += F.mse_loss(z_eos[i], tar_eos[i])

        logits = torch.stack([
            (tar_eos[0] @ zmp_eos[:4].t()).mean(), (tar_eos[0] @ zmp_eos[4:].t()).mean(),
            (tar_eos[1] @ zmn_eos[:4].t()).mean(), (tar_eos[1] @ zmn_eos[4:].t()).mean()
        ]).view(2, 2)
        entropy_loss = F.cross_entropy(logits, torch.tensor([0, 1], device='cuda'))

        loss = entropy_loss - retain_loss #  - ce_loss  # + attack - retain
        print(entropy_loss)
        loss.backward(retain_graph=True)
        z_adv_new = z_adv + z_adv.grad.data.detach().sign() * step
        z_adv = clamper(z_adv_new, z_nat, bound=bound)
        target_model.zero_grad()
        z_adv.grad = None

    return z_adv.detach().to(z.device)


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


def pgd_attack_test(text_encoder, embeds, z_dif, check_, epsilon, step, num_iter):
    delta = torch.zeros_like(embeds, requires_grad=True).to(embeds.device)
    text_encoder.zero_grad()
    for _ in range(num_iter):
        outputs = text_encoder(embeds + delta)
        z_re_adv = F.normalize(outputs[torch.arange(outputs.shape[0]), check_], dim=-1)
        target = torch.tensor([[0, 0], [0, 0]], device=z_re_adv.device, dtype=z_re_adv.dtype)
        s_z_d = (z_re_adv @ z_dif.T).view(2, -1, 7).sum(dim=2)
        loss = F.mse_loss(s_z_d, target)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + step * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        text_encoder.zero_grad()
    delta.grad.zero_()
    return delta.detach()


def pgd_attack_bias(target_model, feature, z_dif, check_bias, mask, target_b, epsilon, step, num_iter):
    delta = torch.zeros_like(feature, requires_grad=True).to(feature.device)
    target_model.zero_grad()
    for _ in range(num_iter):
        outputs = target_model(feature + delta)  # TODO z_re, z_dif, z_dif_b, device
        z_ = F.normalize(outputs[torch.arange(outputs.shape[0]), check_bias], dim=-1)
        loss = F.mse_loss((z_ @ z_dif.T) * mask, target_b)
        loss.backward()
        with torch.no_grad():
            delta.data = (delta + step * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
            delta.grad.zero_()
        target_model.zero_grad()
    delta.grad.zero_()
    return delta.detach()


def sud_loss_txt(z_re, z_dif, s_z_d_b=None):
    target = torch.tensor([[1, 0], [0, 1]], device=z_re.device, dtype=z_re.dtype)
    s_z_d = (z_re @ z_dif.T).view(2, -1, len(z_dif) // 2).mean(dim=2)
    # loss_z_dif -> This is main
    loss_z_dif = torch.mean((s_z_d - s_z_d.min()) ** 2) + F.mse_loss(s_z_d, target)
    if s_z_d_b is not None:
        bias_mean = torch.mean((s_z_d_b - s_z_d_b.min()) ** 2)
        target_loss = loss_z_dif + bias_mean
    else:
        target_loss = loss_z_dif
    return target_loss


import torch.nn as nn


class AttackTarget(nn.Module):
    def __init__(self, text_encoder, target_layer_num=1, device=None):
        super(AttackTarget, self).__init__()
        self.text_encoder = text_encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.text_model.encoder.layers[-target_layer_num:].parameters():
            param.requires_grad = True
        self.target_layer_num = target_layer_num
        self.device = device

    def tokenizer(self, tokenizer, prompt):
        token_ = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                           return_tensors="pt").to(self.device)
        return token_, create_attention_mask(token_.input_ids.shape)

    def get_feature(self, input_ids, attention_mask=None, causal_attention_mask=None):
        hidden_states = self.text_encoder.text_model.embeddings(input_ids=input_ids)
        for layer in self.text_encoder.text_model.encoder.layers[:-self.target_layer_num]:
            hidden_states = layer(hidden_states, attention_mask=attention_mask, causal_attention_mask=None)[0]
        return hidden_states

    def forward(self, features, attention_mask=None):
        for layer in self.text_encoder.text_model.encoder.layers[-self.target_layer_num:]:
            features = layer(features, attention_mask=attention_mask, causal_attention_mask=None)[0]
        return self.text_encoder.text_model.final_layer_norm(features)


def calculate_sim():
    first_class_indices = torch.cat([torch.triu_indices(7, 7, 1), torch.triu_indices(7, 7, 1) + 7], dim=1)
    second_class_indices = (torch.arange(7), torch.arange(7) + 7)

    all_indices = torch.triu_indices(14, 14, 1)
    all_pairs = set(zip(all_indices[0].tolist(), all_indices[1].tolist())) | set(
        zip(all_indices[1].tolist(), all_indices[0].tolist()))

    excluded_indices = set(zip(first_class_indices[0].tolist(), first_class_indices[1].tolist())) | \
                       set(zip(first_class_indices[1].tolist(), first_class_indices[0].tolist())) | \
                       set(zip(second_class_indices[0].tolist(), second_class_indices[1].tolist())) | \
                       set(zip(torch.arange(14).tolist(), torch.arange(14).tolist())) | \
                       set(zip(second_class_indices[1].tolist(), second_class_indices[0].tolist()))

    other_pairs = all_pairs - excluded_indices
    other_indices = (torch.tensor([i for i, j in other_pairs]), torch.tensor([j for i, j in other_pairs]))

    return first_class_indices, second_class_indices, other_indices


def create_attention_mask(input_shape):
    # Create a causal attention mask for CLIP
    batch_size, seq_len = input_shape
    mask = torch.full((batch_size, 1, seq_len, seq_len), float("-inf"))
    mask = torch.triu(mask, diagonal=1)
    mask = mask.to('cuda' if torch.cuda.is_available() else 'cpu')
    return mask


def create_bias_label(size, p_len, device):
    mask = torch.eye(size)
    target = torch.cat([mask, mask], dim=1)
    mask[:p_len, p_len:] = mask[p_len:, :p_len] = 1.0
    mask = torch.cat([mask, mask], dim=1)
    return mask.to(device), target.to(device)


def bias_loss(size, p_len, value):
    positive = torch.eye(size)
    positives = torch.cat([positive, positive], dim=1).to(value.device)
    negative = torch.zeros(positive.shape)
    negative[:p_len, p_len:] = negative[p_len:, :p_len] = 1.0
    negatives = torch.cat([negative, negative], dim=1).to(value.device)

    # Calculate the sum and count for positives
    positive_sum = torch.sum(value * positives, dim=1)
    positive_count = torch.sum(positives, dim=1)
    mean_positives = positive_sum / positive_count

    # Calculate the sum and count for negatives
    negative_sum = torch.sum(value * negatives, dim=1)
    negative_count = torch.sum(negatives, dim=1)
    mean_negatives = negative_sum / negative_count

    result = torch.stack([mean_positives, mean_negatives], dim=1)

    return result


def got_matrix_via_debias_vl(output, token, S, lam=500):
    output = output[torch.arange(output.shape[0]), token.argmax(-1)]
    output = F.normalize(output, dim=-1).cpu().numpy()

    # Compute Calibration Matrix
    M = get_M(output, S)
    G = lam * M + np.eye(M.shape[0])
    P = np.linalg.inv(G)
    P = torch.tensor(P).cuda()
    return P
