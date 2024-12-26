def train_sud_loss_both_tuning(self, num_iters, model_save=False, att_mode=None, learn_mode=None,
                               pre_embedding=None):
    logit_scale, batch, device, check_unlearned, self.pre_embedding = 100, 128, self.cfg.device, True, pre_embedding
    student_layer_img, student_layer_txt = self.student_layer_img, self.student_layer_txt
    optimizer_img = torch.optim.SGD(student_layer_img.parameters(), lr=self.cfg.lr, momentum=0.9)
    optimizer_txt = torch.optim.SGD(student_layer_txt.parameters(), lr=self.cfg.lr, momentum=0.9)
    # teacher_layer = dc(student_layer)
    bound, step, iters = 0.4, 1e-2, 5

    case = ['wrong_best', 'wrong']
    case = case[1]
    # teacher_layer = dc(student_layer)
    adv = PGD(student_layer_img, bound, step, iters, False, True, True, device,
              no_label=self.no_label, using=self.text_model, mode=att_mode, learn_mode=learn_mode)
    # All right -> KD
    # All wrong -> Img, txt Tuning
    # All right & Txt PGD Wrong -> Text tuning
    # All right & Img PGD Wrong -> Img tuning
    for epo in range(num_iters):
        teacher_layer_txt, teacher_layer_img = dc(student_layer_txt), dc(student_layer_img)
        mse_ori_, ce_ori_, mse_adv_, ce_adv_ = 0, 0, 0, 0
        optimizer_img.zero_grad(), optimizer_txt.zero_grad()
        teacher_layer_txt.eval(), teacher_layer_img.eval()

        with torch.no_grad():
            text_embeddings = student_layer_txt(pre_embedding)
        student_layer_img, mse_adv, ce_adv, mse_ori, ce_ori = (
            self.img_tuning(student_layer_img, teacher_layer_img, adv, optimizer_img, device, text_embeddings, case,
                            100))
        mse_ori_ += mse_ori.item()
        ce_ori_ += ce_ori.item()
        mse_adv_ += mse_adv.item()
        ce_adv_ += ce_adv.item()
        with torch.no_grad():
            ori_target = student_layer_img(self.target_image_set)
        student_layer_txt, mse_adv, ce_adv, mse_ori, ce_ori = (
            self.txt_tuning(pre_embedding, ori_target, student_layer_img, student_layer_txt, teacher_layer,
                            adv, optimizer_txt, device, case, 100))
        mse_ori_ += mse_ori.item()
        ce_ori_ += ce_ori.item()
        mse_adv_ += mse_adv.item()
        ce_adv_ += ce_adv.item()

        self.text_model.target_layer = student_layer_txt
        self.image_model.target_layer = student_layer_img
        student_layer_txt.zero_grad()
        student_layer_img.zero_grad()
        if not epo % 3:
            pre_acc, our_acc, pre_acc_de, our_acc_de, att_suc, att_suc_de = 0, 0, 0, 0, 0, 0
            new_text_embeddings = student_layer_txt(pre_embedding)
            with torch.no_grad():
                val_image_set = student_layer_img(self.val_image_set)
            before_logit = logit_scale * val_image_set @ self.text_embeddings.t()
            after_logit = logit_scale * val_image_set @ new_text_embeddings.t()
            before_predict = torch.argmax(before_logit, dim=-1)
            after_predict = torch.argmax(after_logit, dim=-1)

            x_adv = adv.perturb_img(self.val_image_set, text_embeddings, target_y=self.val_y,
                                    model=student_layer_img, device=device)
            new_img_adv = student_layer_img(x_adv)
            att_y = torch.argmax(logit_scale * new_img_adv @ new_text_embeddings.t(), dim=-1)

            att_suc += (att_y != after_predict).sum()
            pre_acc += (before_predict == self.val_y).sum()
            our_acc += (after_predict == self.val_y).sum()
            if torch.isnan(new_text_embeddings).any():
                print('*' * 50, "Tensor contains NaN values.")
            student_layer_txt.zero_grad()
            student_layer_img.zero_grad()
            del new_text_embeddings
            print(f'val - att_suc = {att_suc} | pre_acc = {pre_acc} | our_acc = {our_acc} | ')
        if not att_suc:
            break
        if not epo % 5:
            if model_save:
                import os
                name = f'both_{self.data}'
                os.makedirs(name, exist_ok=True)
                torch.save(student_layer_txt.state_dict(),
                           f'{name}/student_layer{epo}_{bound}_{step}_{iters}_{learn_mode}_txt.pth')
                torch.save(student_layer_img.state_dict(),
                           f'{name}/student_layer{epo}_{bound}_{step}_{iters}_{learn_mode}_img.pth')
            if epo >= 4:
                self.test(False, learn_mode='CAN')
        print(
            f'epoch: {epo} | mse_ori : {round(mse_ori_, 2)} | ce_ori: {round(ce_ori_, 2)}'
            f' | mse_adv : {round(mse_adv_, 2)} | | ce_adv : {round(ce_adv_, 2)}')
    self.text_model.target_layer = student_layer_txt
    self.image_model.target_layer = student_layer_img