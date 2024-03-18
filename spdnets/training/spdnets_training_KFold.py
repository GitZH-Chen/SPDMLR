from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
import torch as th
import torch.nn as nn
import numpy as np
import fcntl

from spdnets.utils.spdnet import Get_Model
from spdnets.utils.spdnet.utils import get_dataset_settings,optimzer,parse_cfg,train_per_epoch,val_per_epoch
import spdnets.utils.spdnet.utils as spdnet_utils
from spdnets.utils.common_utils import set_seed_thread

from spdnets.training.spdnet_training import training_loop

def training_KFold(cfg,args):
    args = parse_cfg(args, cfg)
    args.is_save=False
    args.folds=cfg.fit.folds

    # set logger
    logger = logging.getLogger(args.modelname)
    logger.setLevel(logging.INFO)
    args.logger = logger
    logger.info('begin model {} on dataset: {}'.format(args.modelname, args.dataset))

    # set seed and threadnum
    set_seed_thread(args.seed, args.threadnum)

    # begin K fold experiments
    acc_val_all = [];acc_val_last_k = [];acc_val_final = []
    for ith in range(args.folds):
        # set dataset, model and optimizer
        args.DataLoader = get_dataset_settings(args)
        model = Get_Model.get_model(args)
        loss_fn = nn.CrossEntropyLoss()
        args.loss_fn = loss_fn.cuda()
        args.opti = optimzer(model.parameters(), lr=args.lr, mode=args.optimizer, weight_decay=args.weight_decay)

        # begin training
        args.ith_fold = ith+1
        logger.info(f'{args.ith_fold:d}/{args.folds:d} folds begins')
        # begin training
        val_acc = training_loop_of_KFold(model, args)

        acc_val_all.append(val_acc)
        last_k = np.array(val_acc)[-10:]
        acc_val_final.append(val_acc[-1])
        acc_val_last_k.append(last_k.max())

    save_print_results(args,acc_val_all,acc_val_final,acc_val_last_k,logger)
    return acc_val_all


def training_loop_of_KFold(model, args):
    #setting tensorboard
    if args.is_writer:
        architecture=str(args.architecture)
        if args.folds>1:
            args.writer_path = os.path.join('./tensorboard_logs/SPDNet', architecture, f"{args.modelname}_{args.ith_fold}")
        else:
            args.writer_path = os.path.join('./tensorboard_logs/', f"{args.modelname}_{args.ith_fold}")
        args.logger.info('writer path {}'.format(args.writer_path))
        args.writer = SummaryWriter(args.writer_path)

    acc_val = [];loss_val = [];acc_train = [];loss_train = [];training_time=[]
    logger = args.logger
    # training loop
    for epoch in range(0, args.epochs):
        # training
        elapse, epoch_loss_train, epoch_acc_train = train_per_epoch(model, args)
        training_time.append(elapse)
        acc_train.append(np.asarray(epoch_acc_train).mean() * 100)
        loss_train.append(np.asarray(epoch_loss_train).mean())

        # validation
        epoch_loss_val, epoch_acc_val = val_per_epoch(model, args)
        loss_val.append(np.asarray(epoch_loss_val).mean())
        acc_val.append(np.asarray(epoch_acc_val).mean() * 100)

        # save data into tensorboard
        if args.is_writer:
            args.writer.add_scalar('Loss/val', loss_val[epoch], epoch)
            args.writer.add_scalar('Accuracy/val', acc_val[epoch], epoch)
            args.writer.add_scalar('Loss/train', loss_train[epoch], epoch)
            args.writer.add_scalar('Accuracy/train', acc_train[epoch], epoch)

        # print results
        spdnet_utils.print_results(logger, training_time, acc_val, loss_val, epoch, args)
    logger.info(
        'Fold {}/{}: validation accuracy : {:.2f}% with average time: {:.2f} and average smallest time: {:.2f}'.format(
            args.ith_fold, args.folds, acc_val[-1], np.asarray(training_time[-5:]).mean(),
            np.asarray(sorted(training_time)[:5]).mean()))

    if args.is_writer:
        args.writer.close()
    return acc_val

def write_final_results(file_path,message):
    # Create a file lock
    with open(file_path, "a") as file:
        fcntl.flock(file.fileno(), fcntl.LOCK_EX)  # Acquire an exclusive lock

        # Write the message to the file
        file.write(message + "\n")

        fcntl.flock(file.fileno(), fcntl.LOCK_UN)
def save_print_results(args,acc_val_all,acc_val_final,acc_val_last_k,logger):

    mean = np.asarray(acc_val_last_k).mean()
    std = np.asarray(acc_val_last_k).std()
    final_results_last_k = '{} folds last 10 average result is: {:.2f}/{:.2f}'.format(args.folds,mean, std)
    final_results = '{} folds final_epoch average result is: {:.2f}/{:.2f}'.format(args.folds, np.asarray(acc_val_final).mean(), np.asarray(acc_val_final).std())
    logger.info(final_results_last_k)
    logger.info(final_results)

    final_results_last_k_path = os.path.join(os.getcwd(), 'final_results_last_k_' + args.dataset)
    final_results_path = os.path.join(os.getcwd(), 'final_results_' + args.dataset)
    logger.info("results file path: {} and {}, and saving the results".format(final_results_path,final_results_last_k_path))
    write_final_results(final_results_path, args.modelname + '- ' + final_results)
    write_final_results(final_results_last_k_path, args.modelname + '- ' + final_results_last_k)
    torch_results_dir = './torch_resutls'
    if not os.path.exists(torch_results_dir):
        os.makedirs(torch_results_dir)

    th.save({
        'acc_val_all': acc_val_all,
    }, os.path.join(torch_results_dir, args.modelname))