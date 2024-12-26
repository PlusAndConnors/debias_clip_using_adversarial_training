import torch
import torch.nn as nn


class LinearLoRA(nn.Module):
    def __init__(self, in_features, out_features, r=4):
        super(LinearLoRA, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.lora_A = nn.Linear(in_features, r, bias=True)
        self.lora_B = nn.Linear(r, out_features, bias=True)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x))


class TargetLoRA(nn.Module):
    def __init__(self, mha_layer, r=4):
        super(TargetLoRA, self).__init__()
        self.mha = mha_layer
        embed_dim = self.mha.embed_dim
        self.q_proj_lora = LinearLoRA(embed_dim, embed_dim, r=r)
        self.k_proj_lora = LinearLoRA(embed_dim, embed_dim, r=r)
        self.v_proj_lora = LinearLoRA(embed_dim, embed_dim, r=r)
        self.out_proj_lora = LinearLoRA(embed_dim, embed_dim, r=r)

        for param in self.mha.parameters():
            param.requires_grad = False

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False, attn_mask=None):
        query = self.q_proj_lora(query)
        key = self.k_proj_lora(key)
        value = self.v_proj_lora(value)
        attn_output, attn_weights = self.mha(query, key, value, key_padding_mask=key_padding_mask,
                                             need_weights=need_weights, attn_mask=attn_mask)
        attn_output = self.out_proj_lora(attn_output)
        return attn_output, attn_weights


class TargetVPT(nn.Module):
    def __init__(self, block, prompt_size=20):
        super(TargetVPT, self).__init__()
        self.block = block
        self.mask = self.block.attn_mask
        dtype = next(self.block.attn.parameters()).dtype
        self.prompt_size = prompt_size

        # Xavier uniform initialization for prompt embeddings
        val = torch.sqrt(torch.tensor(6. / float(prompt_size + self.block.attn.embed_dim)))
        self.prompt_embed = nn.Parameter(torch.zeros(prompt_size, 1, self.block.attn.embed_dim).to(dtype))
        nn.init.uniform_(self.prompt_embed.data, -val, val)
        # self.prompt_embed = nn.Parameter(torch.zeros(prompt_size, 1, self.block.attn.embed_dim).to(dtype))

    def forward(self, x, *args, **kwargs):
        if self.mask is None:
            T, B, E = x.shape
            cls_token = x[:1, :, :]
            patch_tokens = x[1 + self.prompt_size:, :, :]
            prompt = self.prompt_embed.expand(-1, B, E)
            x = torch.cat([cls_token, prompt, patch_tokens], dim=0)

        # if self.mask is not None:
        #     P = prompt.shape[0]
        #     expand_mask = torch.zeros((T + P, T + P), dtype=self.mask.dtype).to(self.mask.device)
        #     expand_mask[-T:, -T:] = self.mask
        #     self.block.attn_mask = expand_mask
        return self.block(x, *args, **kwargs)
