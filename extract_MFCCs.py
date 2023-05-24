import numpy as np
import gc
import os 

from python_speech_features import mfcc

current_path = os.getcwd()

# ==================================================================================
# Load the dataset
# ==================================================================================

train_normal_x = np.load(os.path.join(current_path, "Dataset_as_numpy", "train_x.npy"))
train_normal_y = np.load(os.path.join(current_path, "Dataset_as_numpy", "train_y.npy"))

valid_normal_x = np.load(os.path.join(current_path, "Dataset_as_numpy", "valid_x.npy"))
valid_normal_y = np.load(os.path.join(current_path, "Dataset_as_numpy", "valid_y.npy"))

test_normal_x = np.load(os.path.join(current_path, "Dataset_as_numpy", "test_x.npy"))
test_normal_y = np.load(os.path.join(current_path, "Dataset_as_numpy", "test_y.npy"))

# ==================================================================================
# MFCC exctraction if no tensorflow end to end model
# ==================================================================================

print("computing MFCCs")

X_train_features = []

for i in range(len(train_normal_x[:, 0])):
    X_train_features.append(mfcc(train_normal_x[i, :], 16000, numcep=32, nfilt=32))

X_valid_features = []

for i in range(len(valid_normal_x[:, 0])):
    X_valid_features.append(mfcc(valid_normal_x[i, :], 16000, numcep=32, nfilt=32))

X_test_features = []

for i in range(len(test_normal_x[:, 0])):
    X_test_features.append(mfcc(test_normal_x[i, :], 16000, numcep=32, nfilt=32))

train_normal_x = 0
valid_normal_x = 0
test_normal_x = 0
gc.collect()

X_train_features = np.expand_dims(np.array(X_train_features), axis=-1)
X_valid_features = np.expand_dims(np.array(X_valid_features), axis=-1)
X_test_features = np.expand_dims(np.array(X_test_features), axis=-1)

# ==================================================================================
# Save files
# ==================================================================================

np.save(os.path.join(current_path, "MFCCs", "train_MFCCs.npy"), X_train_features)
np.save(os.path.join(current_path, "MFCCs", "valid_MFCCs.npy"), X_valid_features)
np.save(os.path.join(current_path, "MFCCs", "test_MFCCs.npy"), X_test_features)
