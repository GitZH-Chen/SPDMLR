import hydra
from omegaconf import DictConfig

from spdnets.training.spdnet_training import training

class Args:
    """ a Struct Class  """
    pass
args=Args()
args.config_name='SPDNetMLR.yaml'

@hydra.main(config_path='./conf/SPDNet/', config_name=args.config_name, version_base='1.1')
def main(cfg: DictConfig):
    training(cfg,args)

if __name__ == "__main__":
    main()