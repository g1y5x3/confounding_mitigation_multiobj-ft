import numpy as np
import scipy.io as sio

def LoadTrainTestFeatures(FEAT, LABEL, SUBJECT_SKINFOLD, sub_test):
  # Load testing samples
  X_Test     = FEAT[sub_test,0]
  Y_Test     = LABEL[sub_test,0].flatten()
  print(f'# of Testing Samples {len(Y_Test)}')

  # Load training samples
  X_Train = np.zeros((0,48))
  Y_Train = np.zeros(0)    
  C_Train = np.zeros(0)
  for sub_train in range(40):
    if sub_train != sub_test:
      x_s = FEAT[sub_train,0]
      y_s = LABEL[sub_train,0].flatten()
      c_s = np.mean(np.mean(SUBJECT_SKINFOLD[sub_train,:]), axis=1)
      X_Train = np.concatenate((X_Train, x_s), axis=0)
      Y_Train = np.concatenate((Y_Train, y_s), axis=0)
      C_Train = np.concatenate((C_Train, c_s), axis=0)

  print('# of Healthy Samples: %d'%(np.sum(Y_Train == -1)))
  print('# of Fatigued Samples: %d'%(np.sum(Y_Train == 1)))   
  
  return X_Train, Y_Train, C_Train, X_Test, Y_Test

# mainly just for the sake of not keeping the copy of DATA_ALL
def load_datafile(file):
  DATA_ALL = sio.loadmat(file)
  FEAT_N           = DATA_ALL['FEAT_N']            # Normalized features
  LABEL            = DATA_ALL['LABEL']             # Labels
  VFI_1            = DATA_ALL['SUBJECT_VFI']       # VFI-1 Score
  SUBJECT_ID       = DATA_ALL['SUBJECT_ID']        # Sujbect ID
  SUBJECT_SKINFOLD = DATA_ALL['SUBJECT_SKINFOLD']  # Subject Skinfold Thickness
  return FEAT_N, LABEL, SUBJECT_SKINFOLD, VFI_1, SUBJECT_ID
