import tensorflow as tf

import numpy as np 
import argparse
import csv 
import random
import os 

current_path = os.getcwd()

# ==================================================================================
# To use tensor cores and speed up training (uncomment only if available)
# ==================================================================================

# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)
# print('Compute dtype: %s' % policy.compute_dtype)
# print('Variable dtype: %s' % policy.variable_dtype)

# ==================================================================================
# This is used to launch all the runs with the required parameters.
# ==================================================================================

argParser = argparse.ArgumentParser()
argParser.add_argument('--N_samples', type = int, required = True)
argParser.add_argument('--K', type = int, required = True)
argParser.add_argument('--type_prune', type = str, required = True)
argParser.add_argument('--ratio', type = float, required = True)

args = argParser.parse_args()
N_samples = args.N_samples
K = args.K
type_prune = args.type_prune
ratio = args.ratio

# for sanity checking
print('Number of samples used : {}'.format(N_samples))
print('K = {}'.format(K))
print('type_prune : {}'.format(type_prune))
print('Prune_ratio = {}'.format(ratio))

# ==================================================================================
# We load directly the MFCCs into the memory, again, to speed up the process. 
# ==================================================================================

X_train_features = np.load(os.path.join(current_path, 'MFCCs' , 'train_MFCCs.npy'))
train_normal_y = np.load(os.path.join(current_path , 'Dataset_as_numpy', 'train_y.npy'))

X_valid_features = np.load(os.path.join(current_path, 'MFCCs' , 'valid_MFCCs.npy'))
valid_normal_y = np.load(os.path.join(current_path , 'Dataset_as_numpy', 'valid_y.npy'))

X_test_features = np.load(os.path.join(current_path, 'MFCCs' , 'test_MFCCs.npy'))
test_normal_y = np.load(os.path.join(current_path , 'Dataset_as_numpy', 'test_y.npy'))

# ==================================================================================
# select parts of the trainig set only, pruned using Wav2vec + kmeans + number final 
# ==================================================================================

# If the pruning is not "normal" ie strictly random pruning we select according to our pruning results
if type_prune != 'normal': 

    y_pruned = np.load(os.path.join(current_path, 'y_pruned', 'y_{}_pruning_FULL_k{}_{}.npy'.format(type_prune , K , ratio)))

    print('There are ', len(y_pruned), 'samples in the pruned dataset')

    # prune the dataset according to the labels previously selected 
    X_train_features = X_train_features[y_pruned,:]
    train_normal_y = train_normal_y[y_pruned]

# randomly select a certain number of the dataset, random pruning always occurs, even if we pruned the dataset before
labels_random_prune = random.sample(range(1, len(train_normal_y)), N_samples)

X_train_features = X_train_features[labels_random_prune,:]
train_normal_y = train_normal_y[labels_random_prune]

print('There are ', len(train_normal_y), 'samples in the final train set')


# ==================================================================================
# Tensorflow classifiers, uncomment the desired model
# ==================================================================================

Batch_size = 512
epoch = 170
initial_learn_rate = 5e-3
Callback_epoch_number = 15
dropout_rate = 0.1
N_classes = (np.max(train_normal_y)+1)
print('there are {} classes in the dataset'.format(N_classes))


# # LeNet original model 
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=(2, 2), activation='relu', input_shape=(99,32,1)))
# model.add(tf.keras.layers.AveragePooling2D())
# model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 3), activation='relu'))
# model.add(tf.keras.layers.AveragePooling2D())
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(units=120, activation='relu'))
# model.add(tf.keras.layers.Dropout(dropout_rate))
# model.add(tf.keras.layers.Dense(units=84, activation='relu'))
# model.add(tf.keras.layers.Dropout(dropout_rate))
# model.add(tf.keras.layers.Dense(units=N_classes, activation = 'softmax'))
# model.summary()

# # Smallest model
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.Conv2D(filters=3, kernel_size=(2, 2), activation='relu', input_shape=(99,32,1)))
# model.add(tf.keras.layers.AveragePooling2D())
# model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(5, 3), activation='relu'))
# model.add(tf.keras.layers.AveragePooling2D())
# model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 3), activation='relu'))
# model.add(tf.keras.layers.AveragePooling2D())
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(units=84, activation='relu'))
# model.add(tf.keras.layers.Dropout(dropout_rate))
# model.add(tf.keras.layers.Dense(units=N_classes, activation = 'softmax'))
# model.summary()


# Tiny model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(2, 2), activation='relu', input_shape=(99,32,1)))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(2, 2), activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 2), activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(4, 2), activation='relu'))
model.add(tf.keras.layers.AveragePooling2D())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(dropout_rate))
model.add(tf.keras.layers.Dense(units=N_classes, activation = 'softmax'))
model.summary()


model.compile(optimizer=tf.optimizers.Adam(lr=initial_learn_rate),loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_delta=0.0001)
Stopping_condition = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=Callback_epoch_number, restore_best_weights=True, min_delta=0.0001)

history = model.fit(X_train_features, train_normal_y , batch_size=Batch_size,  shuffle=True, callbacks = [lr_reduce,Stopping_condition],  epochs=epoch, validation_data=(X_valid_features, valid_normal_y), verbose = 1)

test_loss, test_acc = model.evaluate(X_test_features, test_normal_y, verbose=2)

print("The final accuracy is : ")
print(test_acc)

with open('results/history_tiny_prune_{}_K{}_{}_{}.csv'.format(N_samples, K, type_prune, ratio), 'a') as f_object:
    writer_object = csv.writer(f_object)
    # writer_object.writerow(['test_loss', 'test_acc', 'N_epochs'])
    writer_object.writerow([test_loss, test_acc, len(history.history['loss'])])