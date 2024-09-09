import os, wandb, argparse
import scipy as sio
import numpy as np
from multiprocessing.pool import ThreadPool
from mlconfound.stats import partial_confound_test
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from util.sEMGhelpers import load_features, partition_features

WANDB = os.getenv("WANDB", False)
NAME  = os.getenv("NAME",  "Confounding-Mitigation-In-Deep-Learning")
GROUP = os.getenv("GROUP", "SVM-sEMG-Leave-One-Subject-Out")

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="sEMG GA-SVM experiments")
  parser.add_argument('-s',      type=int, default=0,   help="start of the subjects")
  parser.add_argument('-nsub',   type=int, default=1,   help="number of subjects to be executed")
  args = parser.parse_args()

  # X - FEAT_N
  # Y - LABEL
  # C - SUBJECT_SKINFOLD
  FEAT_N, LABEL, SUBJECT_SKINFOLD, VFI_1, SUBJECT_ID = load_features("data/subjects_40_v6")

  testing_acc  = np.zeros(40)
  training_acc = np.zeros(40)
  p_value      = np.zeros(40)

  testing_acc_ga  = np.zeros(40)
  training_acc_ga = np.zeros(40)
  p_value_ga      = np.zeros(40)

  start_sub    = args.s
  num_sub      = args.nsub
  for sub_test in range(start_sub, start_sub + num_sub):
    print(SUBJECT_ID[sub_test][0][0])
    sub_txt = "R%03d"%(int(SUBJECT_ID[sub_test][0][0]))
    sub_group = "Fatigued" if int(VFI_1[sub_test][0][0]) > 10 else "Healthy"
    print('\n===No.%d: %s===\n'%(sub_test+1, sub_txt))
    print('VFI-1:', (VFI_1[sub_test][0][0]))


    if WANDB:
      run = wandb.init(project   = NAME,
                        group    = GROUP,
                        name     = sub_txt,
                        tags     = [sub_group],
                        settings = wandb.Settings(_disable_stats=True, _disable_meta=True),
                        reinit   = True)
      wandb.log({"subject_info/vfi_1"  : int(VFI_1[sub_test][0][0])})

    X, Y, C, X_Test, Y_Test, _ = partition_features(FEAT_N, LABEL, SUBJECT_SKINFOLD, sub_test)

    # Split training and validation (mainly for shuffle validation set technicially ot used here)
    X_Train, X_Valid, YC_Train, YC_Valid = train_test_split(X, np.transpose([Y, C]),
                                                            test_size=0.1,
                                                            random_state=42)
    Y_Train, C_Train = YC_Train[:,0], YC_Train[:,1]

    clf = SVC(C=1.0, gamma='scale', kernel='rbf', class_weight='balanced', max_iter=1000, tol=0.001)
    clf.fit(X_Train, Y_Train)

    # training acc
    label_predict = clf.predict(X_Train)
    training_acc[sub_test] = accuracy_score(label_predict, Y_Train)
    print('Training Acc: ', training_acc[sub_test])

    # p value
    ret = partial_confound_test(Y_Train, label_predict, C_Train,
                                cat_y=True, cat_yhat=True, cat_c=False,
                                cond_dist_method='gam',
                                progress=True)
    p_value[sub_test] = ret.p
    print('P Value     : ', p_value[sub_test])

    # testing acc
    label_predict = clf.predict(X_Test)
    testing_acc[sub_test] = accuracy_score(label_predict, Y_Test)
    print('Testing  Acc: ', testing_acc[sub_test])

    if WANDB: 
      wandb.log({"metrics/train_acc" : training_acc[sub_test],
                 "metrics/test_acc"  : testing_acc[sub_test],
                 "metrics/p_value"   : p_value[sub_test]})
      run.finish()