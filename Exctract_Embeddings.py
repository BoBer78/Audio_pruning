from Utility_functions import exctract_wav2vec

import os
import numpy as np

current_path = os.getcwd()

train_normal_x = np.load(os.path.join(current_path, "Dataset_as_numpy", "train_x.npy"))
train_normal_y = np.load(os.path.join(current_path, "Dataset_as_numpy", "train_y.npy"))

# sanity check
print("There are {} classes in the dataset.".format(np.max(train_normal_y + 1)))

# ==================================================================================
# Features exctraction with Wav2Vec
# ==================================================================================

X_train_features = exctract_wav2vec(train_normal_x)

# ==================================================================================
# Save the features
# ==================================================================================

np.save(
    os.path.join(current_path, "wav2vec_features", "x_wav2vec"),
    X_train_features,
)
np.save(
    os.path.join(current_path, "wav2vec_features", "y_wav2vec"),
    train_normal_y,
)
