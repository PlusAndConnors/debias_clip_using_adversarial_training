import numpy as np

def get_parameter_count(model, verbose=True):

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in model_parameters])
    if verbose:
        print(f'-> Number of parameters: {num_params}')
    return num_params
