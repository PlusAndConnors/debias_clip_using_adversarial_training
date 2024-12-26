from yacs.config import CfgNode
def update_args(args, cfg):
    for key, value in cfg.items():
        setattr(args, key, update_args(args, value) if isinstance(value, CfgNode) else value)