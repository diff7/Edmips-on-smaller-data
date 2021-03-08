import os
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import models as models

from utils import (
    save_checkpoint,
    Logger,
    ProgressMeter,
    adjust_learning_rate,
    accuracy,
)

from omegaconf import OmegaConf as omg
from data_loader import DataReader

SEARCH_CONFIG = "./train_config.yaml"

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(models.__dict__[name])
)


def main():
    cfg = omg.load(SEARCH_CONFIG)
    print(cfg)

    os.makedirs(cfg.results_dir, exist_ok=True)

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    if cfg.gpu is not None:
        warnings.warn(
            "You have chosen a specific GPU. This will completely "
            "disable data parallelism."
        )

    if cfg.dist_url == "env://" and cfg.world_size == -1:
        cfg.world_size = int(os.environ["WORLD_SIZE"])

    cfg.distributed = cfg.world_size > 1 or cfg.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if cfg.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        cfg.world_size = ngpus_per_node * cfg.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, cfg=(ngpus_per_node, cfg))
    else:
        # Simply call main_worker function
        main_worker(cfg.gpu, ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, cfg):
    global best_acc1
    best_acc1 = 0
    cfg.gpu = gpu

    if cfg.gpu is not None:
        print("Use GPU: {} for training".format(cfg.gpu))

    if cfg.distributed:
        if cfg.dist_url == "env://" and cfg.rank == -1:
            cfg.rank = int(os.environ["RANK"])
        if cfg.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            cfg.rank = cfg.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank,
        )
    # create model
    print("=> creating model '{}'".format(cfg.arch))
    if len(cfg.arch_cfg) > 0:
        if os.path.isfile(cfg.arch_cfg):
            print(
                "=> loading architecture config from '{}'".format(cfg.arch_cfg)
            )
        else:
            print("=> no architecture found at '{}'".format(cfg.arch_cfg))
    model = models.__dict__[cfg.arch](cfg.arch_cfg)

    if cfg.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if cfg.gpu is not None:
            torch.cuda.set_device(cfg.gpu)
            model.cuda(cfg.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            cfg.batch_size = int(cfg.batch_size / ngpus_per_node)
            cfg.workers = int(
                (cfg.workers + ngpus_per_node - 1) / ngpus_per_node
            )
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[cfg.gpu]
            )
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif cfg.gpu is not None:
        torch.cuda.set_device(cfg.gpu)
        model = model.cuda(cfg.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if "alex" in cfg.arch or "vgg" in cfg.arch:
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(cfg.gpu)

    optimizer = torch.optim.SGD(
        model.parameters(),
        cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )

    # optionally resume from a checkpoint
    if cfg.resume:
        if os.path.isfile(cfg.resume):
            print("=> loading checkpoint '{}'".format(cfg.resume))
            if cfg.gpu is None:
                checkpoint = torch.load(cfg.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = "cuda:{}".format(cfg.gpu)
                checkpoint = torch.load(cfg.resume, map_location=loc)
            cfg.start_epoch = checkpoint["epoch"]
            best_acc1 = checkpoint["best_acc1"]
            if cfg.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(cfg.gpu)
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    cfg.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(cfg.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(cfg.data, "train.txt")
    valdir = os.path.join(cfg.data, "val.txt")
    train_dataset = DataReader(traindir, True, cfg.crop_size)

    if cfg.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset
        )
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        num_workers=cfg.workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    val_dataset = DataReader(valdir, False, cfg.crop_size)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.workers,
        pin_memory=True,
    )

    if cfg.evaluate:
        validate(val_loader, model, criterion, cfg)
        return

    # START RUNNGIN OUR TRAINING LOOP HERE
    logger_eval = init_logger(cfg, search=False, type="eval")
    logger_train = init_logger(cfg, search=False, type="train")

    best_epoch = cfg.start_epoch
    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, None, epoch, cfg)

        # train for one epoch
        train(
            train_loader, model, criterion, optimizer, epoch, cfg, logger_train
        )

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, cfg, epoch, logger_eval)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch = epoch

        print("========= architecture info =========")
        if hasattr(model, "module"):
            bitops, bita, bitw = model.module.fetch_arch_info()
        else:
            bitops, bita, bitw = model.fetch_arch_info()
        print(
            "model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M".format(
                bitops, bita, bitw
            )
        )

        if not cfg.multiprocessing_distributed or (
            cfg.multiprocessing_distributed and cfg.rank % ngpus_per_node == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": cfg.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                epoch,
                cfg.step_epoch,
                cfg.results_dir,
                search=True,
            )

    print("Best Acc@1 {0} @ epoch {1}".format(best_acc1, best_epoch))


# A MODEL TRAINER - NOTHING FANCY HERE


def init_logger(cfg, search, type, best_arch=None):

    logger = Logger(
        cfg.results_dir,
        search=search,
        type=type,
    )

    logger.add("time", ":6.3f")
    # logger.add("data", ":6.3f")
    logger.add("loss", ":.4e")
    logger.add("acc1", ":6.2f")
    logger.add("acc5", ":6.2f")

    if best_arch:
        for key in best_arch:
            logger.set(key, best_arch[key])

    return logger


def train(train_loader, model, criterion, optimizer, epoch, cfg, logger):
    curr_lr = optimizer.param_groups[0]["lr"]
    progress = ProgressMeter(
        len(train_loader),
        [logger.time, logger.loss, logger.acc1, logger.acc5],
        prefix="Epoch: [{}/{}]\t" "LR: {}\t".format(epoch, cfg.epochs, curr_lr),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time

        if cfg.gpu is not None:
            images = images.cuda(cfg.gpu, non_blocking=True)
        target = target.cuda(cfg.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        logger.time.update(time.time() - end)
        logger.loss.update(loss.item(), images.size(0))
        logger.acc1.update(acc1[0].item(), images.size(0))
        logger.acc5.update(acc5[0].item(), images.size(0))
        logger.save(batch=i, epoch=epoch)

        end = time.time()

        if i % cfg.print_freq == 0:
            progress.display(i)


# A MODEL VALIDATION - NOTHING FANCY HERE


def validate(val_loader, model, criterion, cfg, epoch, logger):

    progress = ProgressMeter(
        len(val_loader),
        [logger.time, logger.loss, logger.acc1, logger.acc5],
        prefix="Test: ",
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if cfg.gpu is not None:
                images = images.cuda(cfg.gpu, non_blocking=True)
            target = target.cuda(cfg.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            # measure elapsed time
            logger.time.update(time.time() - end)
            logger.loss.update(loss.item(), images.size(0))
            logger.acc1.update(acc1[0].item(), images.size(0))
            logger.acc5.update(acc5[0].item(), images.size(0))
            end = time.time()
            if i % cfg.print_freq == 0:
                progress.display(i)

        logger.save(batch=i, epoch=epoch)

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {acc1:.3f} Acc@5 {acc5:.3f}".format(
                acc1=logger.acc1.avg, acc5=logger.acc5.avg
            )
        )

    return logger.acc1.avg


if __name__ == "__main__":
    main()