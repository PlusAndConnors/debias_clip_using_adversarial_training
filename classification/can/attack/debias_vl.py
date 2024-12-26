import torch


def debais_vl_s_c(opts):
    if opts.dataset == 'waterbirds':
        text_descriptions = ['This is a picture of a landbird.', 'This is a picture of a waterbird.']
        spurious_prompt = ['This is a land background.', 'This is a picture of a forest.',
                           'This is a picture of a moutain.', 'This is a picture of a wood.',
                           'This is a water background.', 'This is a picture of an ocean.',
                           'This is a picture of a beach.', 'This is a picture of a port.']
        candidate_prompt = ['This is a picture of a landbird with land background.',  # 0
                            'This is a picture of a landbird with water background.',  # 1
                            'This is a picture of a landbird in the ocean',  # 2
                            'This is a picture of a landbird in the water.',  # 3
                            'This is a picture of a landbird in the forest.',  # 4
                            'This is a picture of a waterbird with land background.',  # 5
                            'This is a picture of a waterbird with water background.',  # 6
                            'This is a picture of a waterbird in the ocean',  # 7
                            'This is a picture of a waterbird in the water.',  # 8
                            'This is a picture of a waterbird in the forest.']  # 9
        S = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4],
             [5, 6], [5, 7], [5, 8], [5, 9], [6, 7], [6, 8], [6, 9], [7, 8], [7, 9], [8, 9]]
        # same bias
        B = [[0, 5], [0, 9], [4, 5], [4, 9], [5, 9], [1, 2], [1, 3], [1, 6], [1, 7], [1, 8],
             [2, 3], [2, 6], [2, 7], [2, 8], [3, 6], [3, 7], [3, 8], [6, 7], [6, 8], [7, 8]]

    if opts.dataset == 'celebA':
        text_descriptions = ['A photo of a celebrity with dark hair.', 'A photo of a celebrity with blond hair.']
        spurious_prompt = ['A photo of a male.', 'A photo of a male celebrity.', 'A photo of a man.',
                           'A photo of a female.', 'A photo of a female celebrity.', 'A photo of a woman.']

        candidate_prompt = ['A photo of a male celebrity with dark hair.',
                            'A photo of a female celebrity with dark hair.',
                            'A photo of a male celebrity with blond hair.',
                            'A photo of a female celebrity with blond hair.']
        S = [[0, 1], [2, 3]]
        B = [[0, 2], [1, 3]]  # same bias
    return spurious_prompt, candidate_prompt, S, B, text_descriptions


def debias_vl(spurious_embeddings, candidate_embeddings, S):
    P0 = get_proj_matrix(spurious_embeddings)  # 4.2 remove pseudo way
    M = get_M(candidate_embeddings, S)  # S is opposite index in candidate -> M = zizj -zjzi
    # Regularization ensures that the eigenvalues are bounded away from zero, thus making the matrix invertible and the system stable.
    G = 1000 * M + torch.eye(M.shape[0])
    P = P0 @ torch.inverse(G)
    return P


def bias_vl(spurious_embeddings, candidate_embeddings, B, lamda=1000):
    P0, proj_sup = get_proj_matrix(spurious_embeddings, bias='bias')
    shape_ = proj_sup.shape[0]
    P_set = []
    for b in B:
        M = get_A(candidate_embeddings[b[0]], candidate_embeddings[b[1]])
        # Regularization ensures that the eigenvalues are bounded away from zero, thus making the matrix invertible and the system stable.
        G = lamda * M + torch.eye(shape_)
        P = P0 @ torch.inverse(G)
        P_set.append(P)
    return torch.stack(P_set)


def get_proj_matrix(embeddings, bias=None):
    # Perform SVD
    U, S, V = torch.svd(embeddings)

    # Use all components
    basis = V

    # Orthogonal projection
    proj = torch.inverse(basis.T @ basis)
    proj = basis @ proj
    proj_sup = proj @ basis.T
    proj = torch.eye(proj.shape[0]) - proj_sup
    if bias is not None:
        return proj, proj_sup
    return proj


def get_A(z_i, z_j):
    z_i = z_i[:, None]
    z_j = z_j[:, None]
    return z_i @ z_i.T + z_j @ z_j.T - z_i @ z_j.T - z_j @ z_i.T


def get_M(embeddings, S):
    d = embeddings.shape[1]
    M = torch.zeros((d, d), device=embeddings.device)
    for s in S:
        M += get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(S)
