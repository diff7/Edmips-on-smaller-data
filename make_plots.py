import os
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import OmegaConf as omg

SEARCH_CONFIG = "./search_config.yaml"

DSET = "CIFAR10"

cfg = omg.load(SEARCH_CONFIG)

path_search = os.path.join(cfg.results_dir, "search", "result_val.csv")
path_train = os.path.join(cfg.results_dir, "train", "result_eval.csv")

result__val = pd.read_csv(path_search)
result__train = pd.read_csv(path_train)

f, ax = plt.subplots(2, 2, figsize=(15, 10))


ax[0, 0].scatter(result__val["epoch"], result__val["acc1"], alpha=0.4)
ax[0, 0].set_xlabel("epoch")
ax[0, 0].set_ylabel("acc1")

ax[0, 0].set_title(f"Search  epoch - acc {DSET}")

ax[0, 1].scatter(result__val["epoch"], result__val["bitops"], alpha=0.4)
ax[0, 1].set_xlabel("epoch")
ax[0, 1].set_ylabel("bitops")

ax[0, 1].set_title(f"Search  epoch - bitops (Mb) {DSET}")

ax[1, 1].scatter(result__val["acc1"], result__val["bitops"], alpha=0.4)
ax[1, 1].set_xlabel("acc1")
ax[1, 1].set_ylabel("bitops (MB)")

ax[1, 1].set_title(f"Search  acc1 - bitops (Mb) {DSET}")

ax[1, 0].scatter(result__train["epoch"], result__train["acc1"], alpha=0.4)
ax[1, 0].set_xlabel("epoch")
ax[1, 0].set_ylabel("acc1")

ax[1, 0].set_title(f"Train final  epoch - acc1 {DSET}")

f.savefig("./plots.png")
