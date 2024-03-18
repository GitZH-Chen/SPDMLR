import random
import numpy as np
import torch as th

def set_seed_only(seed):
    seed = seed
    random.seed(seed)
    # th.cuda.set_device(args.gpu)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

def set_seed_thread(seed,threadnum):
    th.set_num_threads(threadnum)
    seed = seed
    random.seed(seed)
    # th.cuda.set_device(args.gpu)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)

def set_up(args):
    set_seed_thread(args.seed, args.threadnum)
    print('begin model {}'.format(args.modelname))
    print('writer path {}'.format(args.writer_path))
