# build_kernel.py

import torch
import hal.models as models
import hal.utils.misc as misc
import hal.kernels as kernels

__all__ = ['KernelMethodY']


class KernelMethodY:
    def __init__(self, opts, modal):
        self.hparams = opts
        if modal == 'image':
            self.lambda_z = opts.tau_z_i  # 0.7
            self.lambda_s = opts.tau_i  # 0.7
            self.gamma = opts.gamma_i
        else:
            self.lambda_z = opts.tau_z_t  # 0.1
            self.lambda_s = opts.tau_t  # 0.1
            self.gamma = opts.gamma_t

        # X(Text or Image),y(target 0,1), S - Bias, Z - Debias
        # K_x, K_y, K_s, K_z
        self.kernel_x = getattr(kernels, self.hparams.kernel_x)(**self.hparams.kernel_x_options)
        self.kernel_y = getattr(kernels, self.hparams.kernel_y)(**self.hparams.kernel_y_options)
        self.kernel_s = getattr(kernels, self.hparams.kernel_s)(**self.hparams.kernel_s_options)
        self.kernel_z = getattr(kernels, self.hparams.kernel_z)(**self.hparams.kernel_z_options)

    def solver(self, X, Y, S, Z=None):
        '''
        Z = theta_D * R_xD
        '''

        device = 'cuda'
        dtype = torch.float

        s = S.to(dtype)

        n = len(X)

        # Kx, Ky, Ks, Kz
        R_x = self.kernel_x(X).to(dtype=dtype, device=device)
        R_x_c = misc.mean_center(R_x, dim=0).to(dtype=dtype, device=device)
        dtype = R_x.dtype

        R_y = self.kernel_y(Y).to(dtype=dtype, device=device)
        R_y_c = misc.mean_center(R_y, dim=0).to(dtype=dtype, device=device)

        # H = I - 1/n (1@1.t() )
        R_s = self.kernel_s(s).to(dtype=dtype, device=device)
        R_s_c = misc.mean_center(R_s, dim=0).to(dtype=dtype, device=device)

        if not Z is None:
            Z_k = self.kernel_z(Z).to(dtype=dtype, device=device)
            Z_k_c = misc.mean_center(Z_k, dim=0).to(dtype=dtype, device=device)

        # Dep(X,Y), Dep(X,Z), Dep(X,S)
        b_y = torch.mm(R_x.t(), R_y_c)
        b_y = torch.mm(b_y, b_y.t())

        if not Z is None:
            b_z = torch.mm(R_x.t(), Z_k_c)
            b_z = torch.mm(b_z, b_z.t())

        b_s = torch.mm(R_x.t(), R_s_c)
        b_s = torch.mm(b_s, b_s.t())

        # L2 Norm
        norm2_b_y = torch.linalg.norm(b_y, 2)
        norm2_b_s = torch.linalg.norm(b_s, 2)

        lamba_s = self.lambda_s / (1. - self.lambda_s)  # Image 0.7/0.3 , Text 0.1/0.9
        if not Z is None:
            norm2_b_z = torch.linalg.norm(b_z, 2)
            self.norm2_b_z = norm2_b_z
            lamba_z = self.lambda_z / (1. - self.lambda_z)

            b = b_y / norm2_b_y - lamba_s * b_s / norm2_b_s + lamba_z * b_z / norm2_b_z
        else:
            # b = Dep(X,Y) - l_s/(1-l_s) Dep(X,S)
            b = b_y / norm2_b_y - lamba_s * b_s / norm2_b_s

        self.norm2_b_y = norm2_b_y
        self.norm2_b_s = norm2_b_s

        b = (b + b.t()) / 2

        c = torch.mm(R_x_c.t(), R_x_c) + n * self.gamma * torch.eye(R_x.shape[1], device=R_x.device)

        c = (c + c.t()) / 2

        eigs, V = torch.linalg.eig(torch.mm(torch.linalg.inv(c), b))
        eigs = torch.real(eigs)
        V = torch.real(V)

        sorted, indeces = torch.sort(eigs, descending=True)

        U = V[:, indeces[0:self.hparams.dim_z]]

        #########################################
        r0 = self.hparams.dim_z
        if self.lambda_s == 0:
            r = r0
        else:
            r1 = min((sorted > 0).sum(), r0)

            ###### Energy Thresholding ######
            if r1 > 0:
                for k in range(1, r1 + 1):
                    if torch.linalg.norm(sorted[0:k]) ** 2 / torch.linalg.norm(sorted[0:r1]) ** 2 >= 0.95:
                        r = k
                        break
            else:
                r = 0
        ######################################################
        if self.lambda_s >= 0.999999999:
            r = 0
        ######################################################
        U[:, r:self.hparams.dim_z] = 0

        encoder = models.KernelizedEncoder(U=n ** 0.5 * U, w=self.kernel_x.w, b=self.kernel_x.b)

        if not Z is None:
            Z_enc = encoder(X)
            if ((Z_enc / Z) < 0).sum() / Z.numel() > 0.5:
                U *= -1
                encoder = models.KernelizedEncoder(U=n ** 0.5 * U, w=self.kernel_x.w, b=self.kernel_x.b)

        self.encoder = encoder

        return encoder


def encod(self, X):
    return self.encoder(X)
