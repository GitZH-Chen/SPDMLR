import hydra
from omegaconf import DictConfig
import warnings
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from library.utils.hydra import hydra_helpers

from spdnets.training.eeg_training import training

warnings.filterwarnings("ignore", category=FitFailedWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Args:
    """ a Struct Class  """
    pass
args=Args()
args.config_name='TSMNetMLR.yaml'
args.architecture='[40,20]'

@hydra_helpers
@hydra.main(config_path='./conf/TSMNet/', config_name=args.config_name, version_base='1.1')
def main(cfg: DictConfig):
    training(cfg,args)

if __name__ == '__main__':

    main()
