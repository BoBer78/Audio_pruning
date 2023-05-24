## Requirements and packages 

- Tensorflow (for classification)
- PyTorch (to exctract features)
- transformers (for wav2vec)
- sklearn (for pruning, PCA, k-means)
- Numpy 
- Matplotlib
- python_speech_features (for MFCCs)
- (Julia for plotting)

## Folders

#### Dataset_as_Numpy

Contains the original dataset. The original google speech commands was normalised and put into the .npy files. 
It has (x,y) pairs (for data and labels) for each train / validation / test set. 
What is really important is to never shuffle the data (like random batches or dataloader), the original order is the only "global label" for all scripts. 
You can put your audio files here as .npy files to process other datasets : 
 - X has size (# samples , sample lenght)
 - y has size (# samples, )
 
For the moment, sample lenght = 16000, some things have to be changed to a different lenght. 
All the used files are : train_x.npy , train_y.npy, valid_x.npy, valid_y.npy, test_x.npy , test_y.npy

#### wav2vec_features 

Contains the features exctracted by wav2vec ran on the original training set. 
It comes empty, to populate it you can run "python Exctract_Embeddings.py". 
If the samples have another size, but the same for all of them, this should still work. 

#### MFCCs

Contains the pre-computed MFCCs. This is then fed to the CNN based classifier to make the classifications. 
This speeds up the computation by a lot since the MFCCs are static and do not need to be computed at each step. 
If the folder is empy, populate it by running "python exctract_MFCCS.py"

#### y_pruned 

Contains the results of the prunig, as simpe lists of labels. This is the reason that nothing has to be shuffle around.
We rely on the fact that the order of the dataset does not move so it is easy to store the pruning results. 
It comes empty at first, it will contains pruning results after running "python pruning.py"

#### results

Contains the results of the classification in different .csv files. 
The csv titles should be self, explanatory. They always contain the loss in the first collumn, the accuracy in the second, and the number of epochs in the last one. 

#### Plots 

Experimental. Little julia script that allows to read and plot the different results. 


## Python scripts 

 - Exctract_Embeddings.py : take the dataset present in Dataset_as_numpy and apply the pre-trained wav2vec model to obtain features. 
 - extract_MFCCs.py : computes the MFCCs of the audio data present in Dataset_as_numpy. 
 - pruning.py : selectively prune the training set wav2vec features, for complete dataset requires 64 Gb of Ram. 
 - Classification_pruning.py : does the classification for all the different pruning ratio / methods. 
 - Run_all_classifications.py : automatically scans all the parameters to generate the final graphs. 

## How to use the code 

 - put your data in Dataset_as_numpy following the required sizes. it should contain : train_x , train_y, valid_x, valid_y, test_x , test_y (as .npy files)
 - change directory to be in this folder.
 - run "python Exctract_Embeddings.py" to get the wav2vec features. 
 - run "python exctract_MFCCs.py" to get the MFCCs
 - check the parameters inside pruning.py, change them to match your projects / dataset's need (number of cluster, clustering method, pruning ratios). 
 - run "python pruning.py" to generate the desired pruned datasets. 
 - the wav2vec_features / MFCCs / y_pruned  should now be populated. 
 - If your audio files do not have a size of 16000, change the size of the model's input in Classification_pruning.py. You can (should) adapt the training parameter according to the task at hand.  
 - Change the parameters you want to scan on in Run_all_Classification.py.
 - run "python Run_all_Classification.py", it will scan all parameters, and output all results to the results folder. 
 - (Navigate to the Plots folder and run the julia script to get the plots)
 - All the results are clearly presented in the results file. The .csv are simple 3 collumn documents, with each column representing Loss, Accuracy and number of epohs. Taking  the average will get you to the final plots. 