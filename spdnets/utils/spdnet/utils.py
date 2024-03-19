import datetime
import geoopt
import time
import torch as th

from datasets.spdnet.Radar_Loader import DataLoaderRadar
from datasets.spdnet.HDM05_Loader import DataLoaderHDM05

def get_dataset_settings(args):
    if args.dataset=='HDM05':
        pval = 0.5
        DataLoader = DataLoaderHDM05(args.path, pval, args.batchsize)
    elif args.dataset== 'RADAR' :
        pval = 0.25
        DataLoader = DataLoaderRadar(args.path,pval,args.batchsize)
    else:
        raise Exception('unknown dataset {}'.format(args.dataset))
    return DataLoader

def get_model_name(args):
    if args.classifier == 'SPDMLR':
        if args.metric == 'SPDLogEuclideanMetric':
            description = f'{args.metric}-[{args.alpha},{args.beta:.4f}]'
        elif args.metric == 'SPDLogCholeskyMetric':
            description = f'{args.metric}-[{args.power}]'

        description = '-' + description
    elif args.classifier == 'LogEigMLR':
        description=''
    else:
        raise NotImplementedError
    optim = f'{args.lr}-{args.optimizer}-{args.weight_decay}'
    name = f'{optim}-{args.model_type}-{args.init_mode}-{args.bimap_manifold}-{args.classifier}{description}-{args.architecture}-{datetime.datetime.now().strftime("%H_%M")}'
    return name

def optimzer(parameters,lr,mode='AMSGRAD',weight_decay=0.):
    if mode=='ADAM':
        optim = geoopt.optim.RiemannianAdam(parameters, lr=lr,weight_decay=weight_decay)
    elif mode=='SGD':
        optim = geoopt.optim.RiemannianSGD(parameters, lr=lr,weight_decay=weight_decay)
    elif mode=='AMSGRAD':
        optim = geoopt.optim.RiemannianAdam(parameters, lr=lr,amsgrad=True,weight_decay=weight_decay)
    else:
        raise Exception('unknown optimizer {}'.format(mode))
    return optim

def parse_cfg(args,cfg):
    # setting args from cfg

    #fit
    args.epochs = cfg.fit.epochs
    args.batchsize = cfg.fit.batch_size
    args.threadnum = cfg.fit.threadnum
    args.is_writer = cfg.fit.is_writer
    args.cycle = cfg.fit.cycle
    args.seed = cfg.fit.seed
    args.is_save = cfg.fit.is_save

    # model
    args.model_type = cfg.nnet.model.model_type
    args.init_mode = cfg.nnet.model.init_mode
    args.bimap_manifold = cfg.nnet.model.bimap_manifold
    args.architecture = cfg.nnet.model.architecture
    args.classifier = cfg.nnet.model.classifier
    args.metric = cfg.nnet.model.metric
    args.power = cfg.nnet.model.power
    args.alpha = cfg.nnet.model.alpha
    args.beta = eval(cfg.nnet.model.beta) if isinstance(cfg.nnet.model.beta, str) else cfg.nnet.model.beta

    #optimizer
    args.optimizer = cfg.nnet.optimizer.mode
    args.lr = cfg.nnet.optimizer.lr
    args.weight_decay = cfg.nnet.optimizer.weight_decay

    #dataset
    args.dataset = cfg.dataset.name
    args.class_num=cfg.dataset.class_num
    args.path = cfg.dataset.path

    # get model name
    args.modelname = get_model_name(args)

    return args

def train_per_epoch(model,args):
    start = time.time()
    epoch_loss, epoch_acc = [], []
    model.train()
    for local_batch, local_labels in args.DataLoader._train_generator:
        local_batch = local_batch.to(th.double)
        args.opti.zero_grad()
        out = model(local_batch)
        l = args.loss_fn(out, local_labels)
        acc, loss = (out.argmax(1) == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
        epoch_loss.append(loss)
        epoch_acc.append(acc)
        l.backward()
        args.opti.step()
    end = time.time()
    elapse = end - start
    return elapse,epoch_loss,epoch_acc

def val_per_epoch(model,args):
    epoch_loss, epoch_acc = [], []
    y_true, y_pred = [], []
    model.eval()
    with th.no_grad():
        for local_batch, local_labels in args.DataLoader._test_generator:
            local_batch = local_batch.to(th.double)
            out = model(local_batch)
            l = args.loss_fn(out, local_labels)
            predicted_labels = out.argmax(1)
            y_true.extend(list(local_labels.cpu().numpy()))
            y_pred.extend(list(predicted_labels.cpu().numpy()))
            acc, loss = (predicted_labels == local_labels).cpu().numpy().sum() / out.shape[0], l.cpu().data.numpy()
            epoch_acc.append(acc)
            epoch_loss.append(loss)
    return epoch_loss,epoch_acc

def print_results(logger,training_time,acc_val,loss_val,epoch,args):
    if epoch % args.cycle == 0:
        logger.info(f'Time: {training_time[epoch]:.2f}, Val acc: {acc_val[epoch]:.2f}, loss: {loss_val[epoch]:.2f} at epoch {epoch + 1:d}/{args.epochs:d}')