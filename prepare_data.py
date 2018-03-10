import caffe
import lmdb
from PIL import Image
import numpy as np
import glob
from random import shuffle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# Initialize the Image set:

Rheight = 128 # Required Height

Rwidth = 128 # Required Width

LabelHeight = 128 # Downscaled height of the label

LabelWidth = 128 # Downscaled width of the label


# Read the files in the Data Folder

toy=False

if toy:
    folder='realdata'
    print 'generating toydata'
    NumberTrain = 3 # Number of Training Images
    NumberTest = 1 # Number of Testing Images
    
else:
    folder='realdata'
    print 'generating realdata'
    NumberTrain = -1 # Number of Training Images
    NumberTest = -1 # Number of Testing Images
    

inputs_data_train = sorted(glob.glob("/home/allanma/data/hair/"+folder+"/train/*.jpg"))
inputs_label_train = sorted(glob.glob("//home/allanma/data/hair/"+folder+"/train/*.png"))

inputs_data_valid = sorted(glob.glob("/home/allanma/data/hair/"+folder+"/test/*.jpg"))
inputs_label_valid = sorted(glob.glob("/home/allanma/data/hair/"+folder+"/test/*.png"))

target_dir = '/home/allanma/data/hairlmdb/'+folder+'/'
# shuffle(inputs_data_train) # Shuffle the DataSet
# shuffle(inputs_data_valid) # Shuffle the DataSet

inputs_Train = inputs_data_train[:NumberTrain] # Extract the training data from the complete set

inputs_Test = inputs_data_valid[:NumberTest] # Extract the testing data from the complete set

inputs_Train_label = inputs_label_train[:NumberTrain] # Extract the training data from the complete set

inputs_Test_label = inputs_label_valid[:NumberTest] # Extract the testing data from the complete set


def create_data_lmdb(inputs=inputs_Train,target_folder='train_data'):
    print("Creating "+target_folder+" LMDB File ..... ")

    if not os.path.exists(target_dir+target_folder):
        os.makedirs(target_dir+target_folder)
    
    in_db = lmdb.open(target_dir+target_folder,map_size=int(1e14))

    with in_db.begin(write=True) as in_txn:

        for in_idx, in_ in enumerate(tqdm(inputs)):
            # print in_idx
            im = np.array(Image.open(in_)) # or load whatever ndarray you need
            Dtype = im.dtype
            im = im[:,:,::-1]
            im = Image.fromarray(im)
            im = im.resize([Rheight, Rwidth], Image.ANTIALIAS)
            im = np.array(im,Dtype) 
        
            # im = im/255.0 # move this operation into preprocessing of the training

            im = im.transpose((2,0,1)).astype(Dtype)
            # print im.shape, im.dtype, Dtype
            im_dat = caffe.io.array_to_datum(im)
            in_txn.put('{:0>10d}'.format(in_idx),im_dat.SerializeToString())

    in_db.close()


def create_label_lmdb(inputs=inputs_Train_label,target_folder='train_label'):
    print("Creating "+target_folder+" LMDB File ..... ")

    if not os.path.exists(target_dir+target_folder):
        os.makedirs(target_dir+target_folder)
    
    in_db = lmdb.open(target_dir+target_folder,map_size=int(1e14))

    with in_db.begin(write=True) as in_txn:

        for in_idx, in_ in enumerate(tqdm(inputs)):
            # print in_idx
            L = np.array(Image.open(in_)) # or load whatever ndarray you need
        
            Dtype = L.dtype
            # L = L[:,:,::-1]
            Limg = Image.fromarray(L)
            Limg = Limg.resize([LabelHeight, LabelWidth],Image.NEAREST) # To resize the Label file to the required size 
            L = np.array(Limg,Dtype)
            
            # print L[L.shape[0]/2:]
            
            L = (L > 200) * 255
            
            # print L[L.shape[0]/2:]

            if len(L.shape)==3:
                L = L[:,:,0] # when it is close to black or close to white, the RGB pixels are all very large or all very small, so taking one of them is close to taking the mean of them
            
            L = L.reshape(L.shape[0],L.shape[1],1)

            L = L.transpose((2,0,1)).astype(Dtype)
            # print L.shape, L.dtype,Dtype
            L_dat = caffe.io.array_to_datum(L)
            in_txn.put('{:0>10d}'.format(in_idx),L_dat.SerializeToString())

    in_db.close()

# Creating LMDB for Training Data
create_data_lmdb(inputs=inputs_Train,target_folder='train_data')
# Creating LMDB for Training Labels
create_label_lmdb(inputs=inputs_Train_label,target_folder='train_label')
# # Creating LMDB for Testing Data
create_data_lmdb(inputs=inputs_Test,target_folder='test_data')
# # Creating LMDB for Testing Labels
create_label_lmdb(inputs=inputs_Test_label,target_folder='test_label')


def read_lmdb(lmdb_path=target_dir+'train_data'):
    
    
    """
        Loops over image data in the lmdb, and displays information about each datum
        Assumes that data dimensions are as follows: (channels, height, width)
    """
    ax = plt.subplot(111)
    plt.hold(False)
    
    
    lmdb_env = lmdb.open(lmdb_path, readonly=True)
    
    with lmdb_env.begin() as lmdb_txn :
        

        lmdb_cursor = lmdb_txn.cursor()
            
        datum = caffe.proto.caffe_pb2.Datum()
    
        for key, value in lmdb_cursor:
            
            datum.ParseFromString(value)
            
            image = np.zeros((datum.channels, datum.height, datum.width))
        
            image = caffe.io.datum_to_array(datum)
            
            image = np.transpose(image, (1, 2, 0))    # -> height, width, channels
            
            if len(image.shape)==3:
                
                # print np.histogram(image,bins=256)
                
                image = image[:,:,::-1]                   # BGR -> RGB
                
            
            print("key: ", key) 
            print("image shape: " + str(image.shape) + ", data type: " + str(image.dtype) + ", random pixel value: " +  str(image[image.shape[0]/2,image.shape[1]/2,0]))
            
            ax.imshow(np.squeeze(image))
            plt.draw()
            plt.waitforbuttonpress()
            
        # img_data = img_data.reshape(datum.channels, datum.height, datum.width)
#         if visualize:
#                     plt.imshow(img_data.transpose([1,2,0]))
#                     plt.show(block=False)

    plt.show() 
    lmdb_txn.abort()
    lmdb_env.close()
    
read_lmdb(lmdb_path=target_dir+'train_label')
