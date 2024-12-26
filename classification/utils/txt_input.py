def type_ck(type):
    temp = ['a photo of']  # 'this is a picture of',
    main_bias = type.split('_')
    if main_bias[0] == 'waterbird':
        ret = ['a landbird', 'a waterbird']
        bias = ['land', 'wood', 'forest', 'mountain', 'water', 'ocean', 'beach']
    elif main_bias[0] == 'job':
        ret = ['a doctor', 'a nurse']
        if main_bias[1] == 'gender':
            bias = ['male', 'female']
    elif main_bias[0] == 'celebA':
        temp = ['a photo of']  # 'this is a picture of',
        ret = ['dark hair', 'blond hair']
        bias = ['male', 'male celebrity', 'man', 'gentleman', 'female', 'female celebrity', 'woman', 'lady']
    else:
        ret = ['a landbird', 'a waterbird']
        bias = ['land', 'wood', 'forest', 'mountain', 'water', 'ocean', 'beach']

    return temp, ret, bias


def mk_prompt_mapping(type='waterbirds'):
    inter_pre = ['with'] if type == 'waterbirds' else None
    inter_post = ['background'] if type == 'waterbirds' else None
    main_prompt, bias_prompt, mix_prompt, mapping = [], [], [], {}
    templates, retain_content, bias_content = type_ck(type)
    for template_idx, template in enumerate(templates):
        for retain in retain_content:  # a photo of ID
            if type == 'celebA':
                main_prompt.append('{} a celebrity with {}.'.format(template, retain))
            else:
                main_prompt.append('{} {}.'.format(template, retain))

        for bias in bias_content:  # a photo of Bias
            if inter_post is not None:
                for inter_po in inter_post:  # a photo of a [Bias] background
                    bias_prompt.append('{} a {} {}.'.format(template, bias, inter_po))
            else:
                if type == 'gender' or type == 'celebA':  # a photo of a male
                    bias_prompt.append('{} a {}.'.format(template, bias))
                else:
                    bias_prompt.append('{} {}.'.format(template, bias))

        for retain_idx, retain in enumerate(retain_content):
            for bias_idx, bias in enumerate(bias_content):
                if inter_pre is not None:
                    for inter_pr in inter_pre:
                        if inter_post is not None:
                            for inter_po in inter_post:
                                prompt = '{} {} {} {} {}.'.format(template, retain, inter_pr, bias, inter_po)
                                mix_prompt.append(prompt)
                                prompt_index = len(mix_prompt) - 1
                                mapping[prompt_index] = [template_idx, retain_idx, bias_idx]
                        else:
                            prompt = '{} {} {} {}.'.format(template, retain, inter_pr, bias)
                            mix_prompt.append(prompt)
                            prompt_index = len(mix_prompt) - 1
                            mapping[prompt_index] = [template_idx, retain_idx, bias_idx]
                else:
                    if inter_post is not None:
                        for inter_po in inter_post:
                            prompt = '{} {} {} {} {}.'.format(template, retain, bias, inter_po)
                            mix_prompt.append(prompt)
                            prompt_index = len(mix_prompt) - 1
                            mapping[prompt_index] = [template_idx, retain_idx, bias_idx]
                    else:
                        if type == 'gender':
                            prompt = '{} a {} {}.'.format(template, bias, retain.replace('a ', ''))
                        else:
                            prompt = '{} {} {}.'.format(template, retain, bias)
                        mix_prompt.append(prompt)
                        prompt_index = len(mix_prompt) - 1
                        mapping[prompt_index] = [template_idx, retain_idx, bias_idx]

    return main_prompt, bias_prompt, mix_prompt, mapping
