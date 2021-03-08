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
    ProgressMeter,
    adjust_learning_rate,
    accuracy,
    Logger,
)

from omegaconf import OmegaConf as omg
from data_loader import DataReader

SEARCH_CONFIG = "./search_config.yaml"

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(models.__dict__[name])
)


best_acc1 = 0


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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg))
    else:
        # Simply call main_worker function
        main_worker(cfg.gpu, ngpus_per_node, cfg)


def main_worker(gpu, ngpus_per_node, cfg):
    global best_acc1
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
    # print("\n".join([k for k in models.__dict__]))
    model = models.__dict__[cfg.arch](num_classes=cfg.num_classes)

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

    # group model/architecture parameters
    params, alpha_params = [], []
    for name, param in model.named_parameters():
        if "alpha" in name:
            alpha_params += [param]
        else:
            params += [param]
    optimizer = torch.optim.SGD(
        params, cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
    )
    arch_optimizer = torch.optim.SGD(
        alpha_params,
        cfg.lra,
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
            arch_optimizer.load_state_dict(checkpoint["arch_optimizer"])
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

    logger_eval = init_logger(cfg, search=True, type="eval")

    if cfg.evaluate:
        validate(val_loader, model, criterion, cfg, 0, logger_eval)
        return

    best_arch = get_best_arch(model)

    best_epoch = cfg.start_epoch

    logger_train = init_logger(
        cfg,
        search=True,
        type="train",
    )
    logger_val = init_logger(cfg, search=True, type="val", best_arch=best_arch)

    for epoch in range(cfg.start_epoch, cfg.epochs):
        if cfg.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, arch_optimizer, epoch, cfg)

        # train for one epoch
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            arch_optimizer,
            epoch,
            cfg,
            logger_train,
        )

        best_arch = get_best_arch(model)
        for k in best_arch:
            logger_val.set(k, best_arch[k])
        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, cfg, epoch, logger_val)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            best_epoch = epoch

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
                    "arch_optimizer": arch_optimizer.state_dict(),
                },
                is_best,
                epoch,
                cfg.step_epoch,
                cfg.results_dir,
                search=True,
            )

    print("Best Acc@1 {0} @ epoch {1}".format(best_acc1, best_epoch))


def init_logger(cfg, search, type, best_arch=None):

    logger = Logger(
        cfg.results_dir,
        search=True,
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


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    arch_optimizer,
    epoch,
    cfg,
    logger,
):

    curr_lr = optimizer.param_groups[0]["lr"]
    curr_lra = arch_optimizer.param_groups[0]["lr"]
    progress = ProgressMeter(
        len(train_loader),
        [logger.time, logger.loss, logger.acc1, logger.acc5],
        prefix="Epoch: [{}/{}]\t"
        "LR: {}\t"
        "LRA: {}\t".format(epoch, cfg.epochs, curr_lr, curr_lra),
    )

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        # logger.data.update(time.time() - end)

        if cfg.gpu is not None:
            images = images.cuda(cfg.gpu, non_blocking=True)
        target = target.cuda(cfg.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        # complexity penalty
        if cfg.complexity_decay != 0:
            if hasattr(model, "module"):
                loss_complexity = (
                    cfg.complexity_decay * model.module.complexity_loss()
                )
            else:
                loss_complexity = cfg.complexity_decay * model.complexity_loss()
            loss += loss_complexity

        # compute gradient and do SGD step
        optimizer.zero_grad()
        arch_optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        arch_optimizer.step()

        # measure elapsed time
        logger.time.update(time.time() - end)
        logger.loss.update(loss.item(), images.size(0))
        logger.acc1.update(acc1[0].item(), images.size(0))
        logger.acc5.update(acc5[0].item(), images.size(0))
        logger.save(batch=i, epoch=epoch)
        end = time.time()

        if i % cfg.print_freq == 0:
            progress.display(i)


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
            logger.time.update(time.time() - end)
            logger.loss.update(loss.item(), images.size(0))
            logger.acc1.update(acc1[0].item(), images.size(0))
            logger.acc5.update(acc5[0].item(), images.size(0))
            logger.time.update(time.time() - end)
            end = time.time()
            if i % cfg.print_freq == 0:
                progress.display(i)
        logger.save(batch=i, epoch=epoch)

        # measure elapsed time

        # TODO: this should also be done with the ProgressMeter
        print(
            " * Acc@1 {acc1:.3f} Acc@5 {acc5:.3f}".format(
                acc1=logger.acc1.avg, acc5=logger.acc5.avg
            )
        )

    return logger.acc1.avg


def get_best_arch(model):
    print("========= architecture =========")
    if hasattr(model, "module"):
        (
            best_arch,
            bitops,
            bita,
            bitw,
            mixbitops,
            mixbita,
            mixbitw,
        ) = model.module.fetch_best_arch()
    else:
        (
            best_arch,
            bitops,
            bita,
            bitw,
            mixbitops,
            mixbita,
            mixbitw,
        ) = model.fetch_best_arch()
    print(
        "best model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M".format(
            bitops, bita, bitw
        )
    )
    print(
        "expected model with bitops: {:.3f}M, bita: {:.3f}K, bitw: {:.3f}M".format(
            mixbitops, mixbita, mixbitw
        )
    )
    for key, value in best_arch.items():
        print("{}: {}".format(key, value))

    return {
        "best_arch_activ": best_arch["best_activ"],
        "best_arch_weight": best_arch["best_weight"],
        "bitops": bitops,
        "bita": bita,
        "bitw": bitw,
    }


if __name__ == "__main__":
    main()
