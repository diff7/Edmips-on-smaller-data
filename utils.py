import os
import shutil
import torch
import pandas as pd


def save_path(results_dir, search):
    return os.path.join(results_dir, str("search" if search else "train"))


def save_checkpoint(
    state,
    is_best,
    epoch,
    step_epoch,
    results_dir,
    search,
    filename="checkpoint.pth.tar",
):

    save_dir = save_path(results_dir, search)
    os.makedirs(save_dir, exist_ok=True)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(save_dir, "model_best.pth.tar"))
    if (epoch + 1) % step_epoch == 0:
        shutil.copyfile(
            filename,
            os.path.join(save_dir, f"checkpoint_ep{epoch + 1}.pth.tar"),
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = round(self.sum / self.count, 4)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class Logger:
    def __init__(self, results_dir, search=True, type="train"):
        self.path = save_path(results_dir, search) + f"/result_{type}.csv"
        os.makedirs(save_path(results_dir, search), exist_ok=True)
        print("Saving logs to: ", self.path)
        self.type = type
        self.metrics = []
        self.params = dict()
        self.head = []
        self.records = []
        self.all_batches = 0

    def add(self, name, params):
        self.metrics.append(name)
        setattr(self, name, AverageMeter(params))

    def set(self, name, params):
        if isinstance(params, list):
            params = [str(x) for x in params]
        else:
            params = [str(params)]

        self.params[name] = params

    def _fill_head(self):
        self.head = (
            ["epoch", "batch"] + self.metrics + list(sorted(self.params.keys()))
        )

    def _get_values(self):
        values = []
        for name in self.metrics:
            values.append(str(getattr(self, name).avg))
            getattr(self, name).reset()
        for name in sorted(self.params):
            values.append(" ".join(self.params[name]))
        return values

    def save(self, batch, epoch):
        self.all_batches += batch
        if len(self.head) == 0:
            self._fill_head()
            self.records.append(self.head)

        self.records.append(
            [str(epoch), str(self.all_batches)] + self._get_values()
        )

        with open(self.path, "w") as f:
            for row in self.records:
                f.write(",".join(row) + "\n")


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


# def adjust_learning_rate(optimizer, epoch, cfg):
#     """Sets the learning rate to the initial LR decayed by 10 every step_epochs"""
#     lr = cfg.lr * (0.1 ** (epoch // cfg.step_epoch))
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr


def adjust_learning_rate(optimizer, arch_optimizer, epoch, cfg):
    """Sets the learning rate to the initial LR decayed by 10 every step_epochs"""
    lr = cfg.lr * (0.1 ** (epoch // cfg.step_epoch))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    if arch_optimizer is not None:
        lra = cfg.lra * (0.1 ** (epoch // cfg.step_epoch))
        for param_group in arch_optimizer.param_groups:
            param_group["lr"] = lra


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# def save_checkpoint(
#     state, is_best, epoch, step_epoch, filename="arch_checkpoint.pth.tar"
# ):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, "arch_model_best.pth.tar")
#     if (epoch + 1) % step_epoch == 0:
#         shutil.copyfile(
#             filename, "arch_checkpoint_ep{}.pth.tar".format(epoch + 1)
#         )


# class AverageMeter(object):
#     """Computes and stores the average and current value"""

#     def __init__(self, name, fmt=":f"):
#         self.name = name
#         self.fmt = fmt
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count

#     def __str__(self):
#         fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
#         return fmtstr.format(**self.__dict__)


# class ProgressMeter(object):
#     def __init__(self, num_batches, meters, prefix=""):
#         self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
#         self.meters = meters
#         self.prefix = prefix

#     def display(self, batch):
#         entries = [self.prefix + self.batch_fmtstr.format(batch)]
#         entries += [str(meter) for meter in self.meters]
#         print("\t".join(entries))

#     def _get_batch_fmtstr(self, num_batches):
#         num_digits = len(str(num_batches // 1))
#         fmt = "{:" + str(num_digits) + "d}"
#         return "[" + fmt + "/" + fmt.format(num_batches) + "]"


# def adjust_learning_rate(optimizer, arch_optimizer, epoch, cfg):
#     """Sets the learning rate to the initial LR decayed by 10 every step_epochs"""
#     lr = cfg.lr * (0.1 ** (epoch // cfg.step_epoch))
#     lra = cfg.lra * (0.1 ** (epoch // cfg.step_epoch))
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr
#     for param_group in arch_optimizer.param_groups:
#         param_group["lr"] = lra


# def accuracy(output, target, topk=(1,)):
#     """Computes the accuracy over the k top predictions for the specified values of k"""
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)

#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.reshape(1, -1).expand_as(pred))

#         res = []
#         for k in topk:
#             correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res
