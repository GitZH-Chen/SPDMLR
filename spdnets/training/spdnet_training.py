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

def training(cfg,args):
    args=parse_cfg(args,cfg)

    #set logger
    logger = logging.getLogger(args.modelname)
    logger.setLevel(logging.INFO)
    args.logger = logger
    logger.info('begin model {} on dataset: {}'.format(args.modelname,args.dataset))

    #set seed and threadnum
    set_seed_thread(args.seed,args.threadnum)

    # set dataset, model and optimizer
    args.DataLoader = get_dataset_settings(args)
    model = Get_Model.get_model(args)
    loss_fn = nn.CrossEntropyLoss()
    args.loss_fn = loss_fn.cuda()
    args.opti = optimzer(model.parameters(), lr=args.lr, mode=args.optimizer,weight_decay=args.weight_decay)
    # begin training
    val_acc = training_loop(model,args)

    return val_acc

def training_loop(model, args):
    #setting tensorboard
    if args.is_writer:
        args.writer_path = os.path.join('./tensorboard_logs/',f"{args.modelname}")
        args.logger.info('writer path {}'.format(args.writer_path))
        args.writer = SummaryWriter(args.writer_path)

    acc_val = [];loss_val = [];acc_train = [];loss_train = [];training_time=[]
    logger = args.logger
    # training loop
    for epoch in range(0, args.epochs):
        # training
        elapse,epoch_loss_train,epoch_acc_train = train_per_epoch(model,args)
        training_time.append(elapse)
        acc_train.append(np.asarray(epoch_acc_train).mean() * 100)
        loss_train.append(np.asarray(epoch_loss_train).mean())

        # validation
        epoch_loss_val,epoch_acc_val = val_per_epoch(model, args)
        loss_val.append(np.asarray(epoch_loss_val).mean())
        acc_val.append(np.asarray(epoch_acc_val).mean() * 100)

        # save data into tensorboard
        if args.is_writer:
            args.writer.add_scalar('Loss/val', loss_val[epoch], epoch)
            args.writer.add_scalar('Accuracy/val', acc_val[epoch], epoch)
            args.writer.add_scalar('Loss/train', loss_train[epoch], epoch)
            args.writer.add_scalar('Accuracy/train', acc_train[epoch], epoch)

        # print results
        spdnet_utils.print_results(logger,training_time,acc_val,loss_val,epoch,args)

    # save final data
    save_results(logger,training_time, acc_val, args)

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

def save_results(logger,training_time,acc_val,args):
    if args.is_save:
        average_time = np.asarray(training_time[-10:]).mean()
        final_val_acc = acc_val[-1]
        final_results = f'Final validation accuracy : {final_val_acc:.2f}% with average time: {average_time:.2f}'
        final_results_path = os.path.join(os.getcwd(), 'final_results_' + args.dataset)
        logger.info(f"results file path: {final_results_path}, and saving the results")
        write_final_results(final_results_path, args.modelname + '- ' + final_results)
        torch_results_dir = './torch_resutls'
        if not os.path.exists(torch_results_dir):
            os.makedirs(torch_results_dir)
        th.save({
            'acc_val': acc_val,
        }, os.path.join(torch_results_dir,args.modelname.rsplit('-',1)[0]))