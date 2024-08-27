import numpy as np
import scipy.io as sio

def partition_features(FEAT, LABEL, SUBJECT_SKINFOLD, sub_test, SUBJECT_ID=None):
  # Load testing samples
  X_Test = FEAT[sub_test,0]
  Y_Test = LABEL[sub_test,0].flatten()
  C_Test = np.mean(np.mean(SUBJECT_SKINFOLD[sub_test,:]), axis=1)
  if SUBJECT_ID is not None: ID_Test = SUBJECT_ID[sub_test,0].flatten()
  print(f'# of Testing Samples {len(Y_Test)}')

  # Load training samples
  X_Train = np.zeros((0,48))
  Y_Train = np.zeros(0)    
  C_Train = np.zeros(0)
  if SUBJECT_ID is not None: ID_Train = np.zeros(0)
  for sub_train in range(40):
    if sub_train != sub_test:
      x_s = FEAT[sub_train,0]
      X_Train = np.concatenate((X_Train, x_s), axis=0)

      y_s = LABEL[sub_train,0].flatten()
      Y_Train = np.concatenate((Y_Train, y_s), axis=0)

      c_s = np.mean(np.mean(SUBJECT_SKINFOLD[sub_train,:]), axis=1)
      C_Train = np.concatenate((C_Train, c_s), axis=0)

      if SUBJECT_ID is not None:
        id_s = SUBJECT_ID[sub_train,0].flatten()
        ID_Train = np.concatenate((ID_Train, id_s), axis=0)

  print('# of Healthy Samples: %d'%(np.sum(Y_Train == -1)))
  print('# of Fatigued Samples: %d'%(np.sum(Y_Train == 1)))

  if SUBJECT_ID is not None:
    return X_Train, Y_Train, C_Train, ID_Train, X_Test, Y_Test, C_Test, ID_Test
  else:
    return X_Train, Y_Train, C_Train, X_Test, Y_Test, C_Test

def partition_features_pair(FEAT, LABEL, SUBJECT_SKINFOLD, sub_test, SUBJECT_ID=None):
  sub_pair = [44, 41, 81, 85, 8 , 24, 34, 29, 52, 39, 88, 16, 2 , 40, 37, 90, 61, 10,
              57, 58, 11, 19, 21, 30, 32, 43, 45, 47, 83, 55, 50, 56, 59, 69, 46, 49]

  # Load testing samples
  sub_test_1 = sub_test
  sub_test_2 = sub_test + 20 if sub_test < 20 else sub_test - 20

  X_Test = np.concatenate((FEAT[sub_test_1, 0], FEAT[sub_test_2, 0]), axis=0)
  Y_Test = np.concatenate((LABEL[sub_test_1, 0].flatten(), LABEL[sub_test_2, 0].flatten()), axis=0)
  C_Test = np.concatenate((np.mean(np.mean(SUBJECT_SKINFOLD[sub_test_1,:]), axis=1),
                           np.mean(np.mean(SUBJECT_SKINFOLD[sub_test_2,:]), axis=1)), axis=0)
  print(X_Test.shape)
  print(Y_Test.shape)
  print(C_Test.shape)

  if SUBJECT_ID is not None: ID_Test = SUBJECT_ID[sub_test,0].flatten()
  print(f'# of Testing Samples {len(Y_Test)}')

  # Load training samples
  X_Train = np.zeros((0,48))
  Y_Train = np.zeros(0)    
  C_Train = np.zeros(0)
  if SUBJECT_ID is not None: ID_Train = np.zeros(0)
  for sub_train in range(40):
    if sub_train != sub_test_1 and sub_train != sub_test_2:
      x_s = FEAT[sub_train,0]
      X_Train = np.concatenate((X_Train, x_s), axis=0)

      y_s = LABEL[sub_train,0].flatten()
      Y_Train = np.concatenate((Y_Train, y_s), axis=0)

      c_s = np.mean(np.mean(SUBJECT_SKINFOLD[sub_train,:]), axis=1)
      C_Train = np.concatenate((C_Train, c_s), axis=0)

      if SUBJECT_ID is not None:
        id_s = SUBJECT_ID[sub_train,0].flatten()
        ID_Train = np.concatenate((ID_Train, id_s), axis=0)

  print('# of Healthy Samples: %d'%(np.sum(Y_Train == -1)))
  print('# of Fatigued Samples: %d'%(np.sum(Y_Train == 1)))

  if SUBJECT_ID is not None:
    return X_Train, Y_Train, C_Train, ID_Train, X_Test, Y_Test, C_Test, ID_Test
  else:
    return X_Train, Y_Train, C_Train, X_Test, Y_Test, C_Test

# mainly just for the sake of not keeping the copy of DATA_ALL
def load_features(file):
  DATA_ALL = sio.loadmat(file)
  FEAT_N           = DATA_ALL['FEAT_N']            # Normalized features
  LABEL            = DATA_ALL['LABEL']             # Labels
  VFI_1            = DATA_ALL['SUBJECT_VFI']       # VFI-1 Score
  SUBJECT_ID       = DATA_ALL['SUBJECT_ID']        # Sujbect ID
  SUBJECT_SKINFOLD = DATA_ALL['SUBJECT_SKINFOLD']  # Subject Skinfold Thickness
  return FEAT_N, LABEL, SUBJECT_SKINFOLD, VFI_1, SUBJECT_ID

def load_raw_signals(file):
  data = sio.loadmat(file)
  signals = data['DATA']
  labels = data['LABEL']
  VFI1 = data['SUBJECT_VFI']       # VFI-1 Score
  sub_id = data['SUBJECT_ID']        # Sujbect ID
  sub_skinfold = data['SUBJECT_SKINFOLD']  # Subject Skinfold Thickness
  return signals, labels, VFI1, sub_id, sub_skinfold