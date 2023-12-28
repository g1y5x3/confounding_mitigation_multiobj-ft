import wandb
import argparse
import numpy as np
from tsai.all import *
from fastai.callback.wandb import *
from fastai.layers import *
from sklearn.metrics import accuracy_score
from mlconfound.stats import partial_confound_test
from util.sEMGhelpers import load_datafile, partition

# environment variable for the experiment
WANDB = os.getenv("WANDB", False)
NAME  = os.getenv("NAME",  "Confounding-Mitigation-In-Deep-Learning")
GROUP = os.getenv("GROUP", "MLP-sEMG")

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="sEMG MLP experiments")
  parser.add_argument('-bs', type=int, default=128, help="batch size")
  parser.add_argument('-nw', type=int, default=2,   help="number of workers")
  args = parser.parse_args()
  # X - FEAT_N
  # Y - LABEL
  # C - SUBJECT_SKINFOLD
  FEAT_N, LABEL, SUBJECT_SKINFOLD, VFI_1, SUBJECT_ID = load_datafile("data/subjects_40_v6")

  # NOTE
  # For the neural networks implementation, a high-level API was used in order to minimize
  # implementation tsai is wrapped around fastai's API but it has a better numpy interface
  # more reference can be found in https://timeseriesai.github.io/tsai/
  train_acc = np.zeros(40)
  test_acc  = np.zeros(40)
  p_value   = np.zeros(40)
  for sub_test in range(40):
    sub_txt = "R%03d"%(int(SUBJECT_ID[sub_test][0][0]))
    sub_group = "Fatigued" if int(VFI_1[sub_test][0][0][0]) > 10 else "Healthy"
    print('\n===No.%d: %s===\n'%(sub_test+1, sub_txt))
    print('VFI-1:', (VFI_1[sub_test][0][0]))

    cbs = None
    if WANDB:
      run = wandb.init(project=NAME,
                       group=GROUP,
                       name=sub_txt,
                       tags=[sub_group],
                       reinit=True)
      cbs = WandbCallback(log_preds=False)

    print("Loading training and testing set")
    X_Train, Y_Train, C_Train, X_Test, Y_Test = partition(FEAT_N, LABEL, SUBJECT_SKINFOLD, sub_test)

    # Setting "stratify" to True ensures that the relative class frequencies are approximately preserved in each train and validation fold.
    splits = get_splits(Y_Train, valid_size=.1, stratify=True, random_state=123, shuffle=True, show_plot=False)
    tfms   = [None, [Categorize()]]
    dsets       = TSDatasets(X_Train, Y_Train, tfms=tfms, splits=splits)
    dsets_train = TSDatasets(X_Train, Y_Train, tfms=tfms) # keep an unsplit copy for computing the p-value
    dsets_test  = TSDatasets(X_Test,  Y_Test,  tfms=tfms)
    dls       = TSDataLoaders.from_dsets(dsets.train, dsets.valid, shuffle_train=True, bs=args.bs, num_workers=args.nw)
    dls_train = dls.new(dsets_train)
    dls_test  = dls.new(dsets_test)

    # This model is pre-defined in https://timeseriesai.github.io/tsai/models.mlp.html
    model = MLP(c_in=dls.vars, c_out=dls.c, seq_len=48, layers=[50, 50, 50], use_bn=True)
    print(model)

    # Training loop is abstracted by the fastai API
    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy, cbs=cbs)

    # Basically apply some tricks to make it converge faster
    # https://docs.fast.ai/callback.schedule.html#learner.lr_find
    # https://docs.fast.ai/callback.schedule.html#learner.fit_one_cycle
    learn.lr_find()
    learn.fit_one_cycle(50, lr_max=1e-3)

    train_preds, train_targets = learn.get_preds(dl=dls_train)
    train_acc[sub_test] = accuracy_score(train_targets, train_preds.argmax(dim=1))
    print(f"Training acc: {train_acc[sub_test]}")

    ret = partial_confound_test(train_targets.numpy(), train_preds.argmax(dim=1).numpy(), C_Train,
                                cat_y=True, cat_yhat=True, cat_c=False,
                                cond_dist_method='gam',
                                progress=True)
    p_value[sub_test] = ret.p
    print(f"P Value     : {p_value[sub_test]}")

    test_preds, test_targets = learn.get_preds(dl=dls_test)
    test_acc[sub_test] = accuracy_score(test_targets, test_preds.argmax(dim=1))
    print(f"Testing acc : {test_acc[sub_test]}")

    if WANDB: wandb.log({"subject_info/vfi_1" : int(VFI_1[sub_test][0][0]),
                         "metrics/train_acc"  : train_acc[sub_test],
                         "metrics/test_acc"   : test_acc[sub_test],
                         "metrics/p_value"    : p_value[sub_test]})