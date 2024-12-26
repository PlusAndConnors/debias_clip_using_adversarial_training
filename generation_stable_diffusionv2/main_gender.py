import torch
import torch.nn.functional as F

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, AdamW
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler
from itertools import combinations as combi
import argparse
import os
from generation_utils import txt_process, pgd_attack, AttackTarget
from generation_utils import create_prompts_and_index_mapping
from copy import deepcopy as dc

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debiased Diffusion Models')
    parser.add_argument('--cls', default="a photo of a waterbird", type=str,
                        help='target class name')  # landbird
    parser.add_argument('--lam', default=100, type=float, help='regualrizer constant')
    parser.add_argument('--mode', default='base', type=str, help='mode')  # ['debias_vl, base, debias_vl_bias]
    parser.add_argument('--use', default='base', type=str, help='')  # ['debias_vl, base, fairerclip, ]
    parser.add_argument("--revision", type=str, default=None, required=False,
                        help="Revision of pretrained model identifier from huggingface.co/models.", )
    parser.add_argument("--bias", type=str, default='background')
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--alpha", type=int, default=0.3)
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="stabilityai/stable-diffusion-2-1-base",
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--config-file", default="configs/debias_waterbird_mode.yaml", metavar="FILE",
                        help="path to config file", type=str)  # debias_waterbird_mode.yaml
    parser.add_argument('--model', default="clip_ViTL14", type=str)  # bias or None or bafa

    parser.add_argument(
        "--pretrained_vae_name_or_path", type=str, default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    args = parser.parse_args()
    height = 512  # default height of Stable Diffusion
    width = 512  # default width of Stable Diffusion
    num_inference_steps = 100  # Number of denoising steps
    guidance_scale = 7.5  # Scale for classifier-free guidance
    batch_size = 1
    alpha = args.alpha

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,
                                              subfolder="tokenizer", revision=args.revision, )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, )
    text_encoder = text_encoder.to(torch_device)

    target_model = AttackTarget(text_encoder, 1, torch_device).to(torch_device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, target_model.parameters()), lr=5e-5)
    # peft tuning
    sub = torch.tensor(0, device=torch_device).float()

    templates = ['a photo of']  # 'this is a picture of',
    retain_content = ['a doctor', 'a house keeper']
    bias_content = ['male', 'female']
    retain_prompt, bias_prompt, candidate_prompt, index_mapping = (
        create_prompts_and_index_mapping(templates, retain_content, bias_content, type='gender'))

    with torch.no_grad():
        z0 = txt_process(retain_prompt, text_encoder, tokenizer, torch_device)
        z_diff = txt_process(candidate_prompt, text_encoder, tokenizer, torch_device)
        z_diff_bias = txt_process(bias_prompt, text_encoder, tokenizer, torch_device)
    candidate_set = {
        idx: tokenizer(text=candidate_prompt[idx], padding='max_length', return_tensors="pt", truncation=True).to(
            torch_device) for idx in range(len(candidate_prompt))}
    use_ = 'bias'


    def create_attention_mask(input_shape):
        # Create a causal attention mask for CLIP
        batch_size, seq_len = input_shape
        mask = torch.full((batch_size, 1, seq_len, seq_len), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = mask.to('cuda' if torch.cuda.is_available() else 'cpu')
        return mask


    with torch.no_grad():
        token_, ret_mask = target_model.tokenizer(tokenizer, retain_prompt)
        check_ = token_['input_ids'].argmax(-1)
        wz_feature = target_model.get_feature(token_.input_ids, ret_mask)
        candi_token_, cand_mask = target_model.tokenizer(tokenizer, candidate_prompt)
        z_diff_feature = target_model.get_feature(candi_token_.input_ids, cand_mask)
        bias_token_, bias_mask = target_model.tokenizer(tokenizer, bias_prompt)
        z_diff_bias_feature = target_model.get_feature(bias_token_.input_ids, bias_mask)

    target_len, token_len, dim = z0.shape
    mix_len, bias_len = z_diff.size(0), z_diff_bias.size(0)
    ind_target, ind_bias, ind_mix = token_['input_ids'], bias_token_['input_ids'], candi_token_['input_ids']
    z0_tar_eos = F.normalize(z0[torch.arange(target_len), ind_target.argmax(-1)], dim=-1)
    z_dif = F.normalize(z_diff[torch.arange(z_diff.shape[0]), candi_token_['input_ids'].argmax(-1)], dim=-1)
    z_dif_b = F.normalize(z_diff_bias[torch.arange(z_diff_bias.shape[0]), bias_token_['input_ids'].argmax(-1)], dim=-1)
    min_value = z0_tar_eos[0].detach() @ z0_tar_eos[1].detach().T

    for epoch in range(args.epochs):  # Number of epochs
        equalization_loss, distil_loss, target_loss = dc(sub), dc(sub), dc(sub)
        equal_loss_adv, distil_loss_adv, cnt = dc(sub), dc(sub), 0
        optimizer.zero_grad()

        z_tar = target_model(wz_feature)
        for wz_, z0_ in zip(z_tar, z0):
            distil_loss += F.mse_loss(wz_, z0_).to(torch_device)
        z_tar_eos = F.normalize(z_tar[torch.arange(z_tar.shape[0]), check_], dim=-1)
        uq_pair = [v[2] for v in index_mapping.values() if v[1] == 0]
        for i, cls_emb in enumerate(z_tar):
            for j, (f, s) in enumerate(combi(uq_pair, 2)):
                cnt += 1
                equalization_loss += F.mse_loss(cls_emb @ z_dif[f].T, cls_emb @ z_dif[s].T).to(torch_device)
                equalization_loss += F.mse_loss(cls_emb @ z_dif[bias_len + f].T, cls_emb @ z_dif[bias_len + s].T).to(
                    torch_device)
        target_loss += equalization_loss / (i + 1) / (j + 1) * 5

        print(f'{epoch}epoch |' 'distil_loss : ',
              round(distil_loss.item(), 4), 'target_loss : ', round(target_loss.item(), 4), )
        loss = (1 - alpha) * distil_loss + alpha * target_loss
        loss.backward()
        optimizer.step()

    target_model.eval()
    # debias process with text_embedding

    prompt = ['a photo of doctor', 'a photo of female doctor', 'a photo of house keeper',
              'a photo of male house keeper']
    # Define Text Embedding
    with torch.no_grad():
        text_embeddings = txt_process(prompt, target_model.text_encoder, tokenizer, torch_device)
        uncond_embeddings = txt_process([""] * batch_size, target_model.text_encoder, tokenizer, torch_device)

    token_, ret_mask = target_model.tokenizer(tokenizer, retain_prompt)
    check_ = token_['input_ids'].argmax(-1)
    wz_feature = target_model.get_feature(token_.input_ids, ret_mask)
    delta = pgd_attack(target_model, wz_feature, z_dif, z_dif_b, check_, epsilon=0.1, step=1e-2, num_iter=5,
                       device=torch_device)
    z_adv = target_model(wz_feature + delta)
    z_tar = target_model(wz_feature)

    '''

    '''
    # uncond_embeddings[0][1:5] = torch.zeros(uncond_embeddings[0][1:5].shape)
    # text_embeddings[0][1:5] = torch.zeros(uncond_embeddings[0][1:5].shape)

    text_embeddings_water = torch.cat([uncond_embeddings, text_embeddings[0, None]])
    text_embeddings_water_land = torch.cat([uncond_embeddings, text_embeddings[1, None]])
    text_embeddings_land = torch.cat([uncond_embeddings, text_embeddings[2, None]])
    text_embeddings_land_water = torch.cat([uncond_embeddings, text_embeddings[3, None]])

    # generator
    generator = torch.manual_seed(12345)  # Seed generator to create the inital latent noise

    vae = AutoencoderKL.from_pretrained(
        args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
        subfolder=None if args.pretrained_vae_name_or_path else "vae",
        revision=None if args.pretrained_vae_name_or_path else args.revision,
    )

    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", revision=args.revision, )
    scheduler = LMSDiscreteScheduler.from_config(args.pretrained_model_name_or_path, subfolder="scheduler")

    vae = vae.to(torch_device)
    unet = unet.to(torch_device)

    # Diffusion Sampling
    save_dir = (f"241017_gender" + prompt[0] + "_lam" + str(args.lam)
                + args.mode + args.use + str(args.epochs) + '_' + str(args.alpha))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    img_list = [2, 3, 4, 6, 10]
    for i in tqdm(range(20)):
        # Generate Initial Noise
        latents = torch.randn((batch_size, unet.in_channels, height // 8, width // 8), generator=generator, )
        if not i in img_list:
            continue
        latents = latents.to(torch_device)

        def generate_image(latents, text_embeddings):
            scheduler.set_timesteps(num_inference_steps)
            latents = latents * scheduler.init_noise_sigma
            for t in tqdm(scheduler.timesteps):
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

                with torch.no_grad():
                    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                latents = scheduler.step(noise_pred, t, latents).prev_sample

            latents = 1 / 0.18215 * latents

            with torch.no_grad():
                image = vae.decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            images = (image * 255).round().astype("uint8")
            return images


        img_orig_1 = generate_image(latents.clone(), text_embeddings_water)
        img_orig_0 = generate_image(latents.clone(), text_embeddings_water_land)
        img_orib_1 = generate_image(latents.clone(), text_embeddings_land)
        img_orib_0 = generate_image(latents.clone(), text_embeddings_land_water)
        for idx in range(len(img_orig_0)):
            combined_image = Image.new('RGB', (img_orig_0[idx].shape[1] * 2, img_orig_0[idx].shape[0] * 2))
            combined_image.paste(Image.fromarray(img_orig_1[idx]), (0, 0))
            combined_image.paste(Image.fromarray(img_orib_1[idx]), (img_orig_0[idx].shape[1], 0))
            combined_image.paste(Image.fromarray(img_orig_0[idx]), (0, img_orig_0[idx].shape[0]))
            combined_image.paste(Image.fromarray(img_orib_0[idx]), (img_orib_1[idx].shape[1], img_orig_0[idx].shape[0]))

            combined_image.save(f"{save_dir}/img_{args.cls}_{i}_{idx}.jpg")
