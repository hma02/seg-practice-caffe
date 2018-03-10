import caffe
import lmdb
from PIL import Image
import numpy as np
import glob
from random import shuffle

# Initialize the Image set:

NumberTrain = 3 # Number of Training Images

NumberTest = 1 # Number of Testing Images

Rheight = 224 # Required Height

Rwidth = 224 # Required Width

LabelHeight = 224 # Downscaled height of the label

LabelWidth = 224 # Downscaled width of the label


# Read the files in the Data Folder

inputs_data_train = sorted(glob.glob("/home/allanma/data/hair/toydata/TrainData/*.jpg"))
inputs_label_train = sorted(glob.glob("//home/allanma/data/hair/toydata/TrainData/*.png"))

inputs_data_valid = sorted(glob.glob("/home/allanma/data/hair/toydata/ValData/*.jpg"))
inputs_label_valid = sorted(glob.glob("/home/allanma/data/hair/toydata/ValData/*.png"))

target_dir = '/home/allanma/data/hairlmdb/toy'
# shuffle(inputs_data_train) # Shuffle the DataSet
# shuffle(inputs_data_valid) # Shuffle the DataSet

inputs_Train = inputs_data_train[:NumberTrain] # Extract the training data from the complete set

inputs_Test = inputs_data_valid[:NumberTest] # Extract the testing data from the complete set

inputs_Train_label = inputs_label_train[:NumberTrain] # Extract the training data from the complete set

inputs_Test_label = inputs_label_valid[:NumberTest] # Extract the testing data from the complete set


# Creating LMDB for Training Data

print("Creating Training Data LMDB File ..... ")

in_db = lmdb.open(target_dir+'/Train_Data_lmdb',map_size=10485760)

with in_db.begin(write=True) as in_txn:

    for in_idx, in_ in enumerate(inputs_Train):
        print in_idx
        im = np.array(Image.open(in_)) # or load whatever ndarray you need
        Dtype = im.dtype
        im = im[:,:,::-1]
        im = Image.fromarray(im)
        im = im.resize([Rheight, Rwidth], Image.ANTIALIAS)
        im = np.array(im,Dtype) 
        
        im = im/255.0 

        im = im.transpose((2,0,1))
        print im.shape, im.dtype, Dtype
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx),im_dat.SerializeToString())

in_db.close()

# Creating LMDB for Training Labels

print("Creating Training Label LMDB File ..... ")

in_db = lmdb.open(target_dir+'/Train_Label_lmdb',map_size=10485760)

with in_db.begin(write=True) as in_txn:

    for in_idx, in_ in enumerate(inputs_Train_label):
        print in_idx
        L = np.array(Image.open(in_)) # or load whatever ndarray you need
        
        Dtype = L.dtype
        # L = L[:,:,::-1]
        Limg = Image.fromarray(L)
        Limg = Limg.resize([LabelHeight, LabelWidth],Image.NEAREST) # To resize the Label file to the required size 
        L = np.array(Limg,Dtype)
        if len(im.shape)==3:
            L = (L > 200) * 1.0
            L = L[:,:,0]
            
        L = L.reshape(L.shape[0],L.shape[1],1).astype(Dtype)

        L = L.transpose((2,0,1))
        print L.shape, L.dtype,Dtype
        L_dat = caffe.io.array_to_datum(L)
        in_txn.put('{:0>10d}'.format(in_idx),L_dat.SerializeToString())

in_db.close()

# Creating LMDB for Testing Data

print("Creating Testing Data LMDB File ..... ")

in_db = lmdb.open(target_dir+'/Test_Data_lmdb',map_size=10485760)

with in_db.begin(write=True) as in_txn:

    for in_idx, in_ in enumerate(inputs_Test):
        print in_idx    
        im = np.array(Image.open(in_)) # or load whatever ndarray you need
        Dtype = im.dtype
        im = im[:,:,::-1]
        im = Image.fromarray(im)
        im = im.resize([Rheight, Rwidth], Image.ANTIALIAS)
        im = np.array(im,Dtype)
        
        im = im/255.0     

        im = im.transpose((2,0,1))
        print im.shape, im.dtype, Dtype
        im_dat = caffe.io.array_to_datum(im)
        in_txn.put('{:0>10d}'.format(in_idx),im_dat.SerializeToString())

in_db.close()

# Creating LMDB for Testing Labels

print("Creating Testing Label LMDB File ..... ")

in_db = lmdb.open(target_dir+'/Test_Label_lmdb',map_size=10485760)

with in_db.begin(write=True) as in_txn:

    for in_idx, in_ in enumerate(inputs_Test_label):
        print in_idx    

        L = np.array(Image.open(in_)) # or load whatever ndarray you need
        # L = (L > 200) * 1.0
        Dtype = L.dtype
        # L = L[:,:,::-1]
        Limg = Image.fromarray(L)
        Limg = Limg.resize([LabelHeight, LabelWidth],Image.NEAREST) # To resize the Label file to the required size 

        L = np.array(Limg,Dtype)
        if len(im.shape)==3:
            L = (L > 200) * 1.0
            L = L[:,:,0]

        L = L.reshape(L.shape[0],L.shape[1],1).astype(Dtype)

        L = L.transpose((2,0,1))
        print L.shape, L.dtype, Dtype
        L_dat = caffe.io.array_to_datum(L)
        in_txn.put('{:0>10d}'.format(in_idx),L_dat.SerializeToString())

in_db.close()