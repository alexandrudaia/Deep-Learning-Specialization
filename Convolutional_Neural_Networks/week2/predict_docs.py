import warnings
warnings.filterwarnings("ignore")
import numpy as np
import  scipy.misc
import glob
#ssh  daia@34.244.143.184
#9ccc2ef88ddea19c2f2539115142a6b9
train=np.empty(shape=(8000,300,300))
y=[]
counter=0
for filename in glob.iglob('/home/daia/classifier/trdom/*.*'):
     img=scipy.misc.imread(filename,mode='L')
     img=scipy.misc.imresize(img,(300,300))
     train[counter]=img
     counter+=1
     y.append(0)
for filename in glob.iglob('/home/daia/classifier/hrdom/*.*'):
     img=scipy.misc.imread(filename,mode='L')
     img=scipy.misc.imresize(img,(300,300))
     train[counter]=img
     counter+=1
     y.append(1)
for filename in glob.iglob('/home/daia/classifier/krdom/*.*'):
     img=scipy.misc.imread(filename,mode='L')
     img=scipy.misc.imresize(img,(300,300))
     train[counter]=img
     counter+=1
     y.append(2)
for filename in glob.iglob('/home/daia/classifier/frdom/*.*'):
     img=scipy.misc.imread(filename,mode='L')
     img=scipy.misc.imresize(img,(300,300))
     train[counter]=img
     counter+=1
     y.append(3)
y=np.array(y)
y=y.reshape(8000,1)
#Perfect   , now   having train and test  we also  need to shuffle  the data  as a good practice  for evaluation  and fiting 
from random import shuffle
N=train.shape[0]
ind_list = [i for i in range(N)]
shuffle(ind_list)

train_new  = train[ind_list, :,:]
target_new = y[ind_list,]
train=train_new
y=target_new
##########################################################end of shuffling#########################################################

#train test split in order  to  evaluate the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train, y, test_size=0.33, random_state=42)

#

import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding1D, BatchNormalization, Flatten, Conv1D
from keras.layers import AveragePooling1D, MaxPooling1D, Dropout, GlobalMaxPooling1D, GlobalAveragePooling1D 
from keras.models import Model, Sequential
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
 
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K
def DocumentModel(input_shape):
    """
    Implementation of the HappyModel.
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """
    
 
   
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)
    X=Sequential()

   
   
 
    # CONV -> BN -> RELU Block applied to X
    X = Conv1D(32, 7, strides = 1, name = 'conv1')(X_input)
    X = BatchNormalization(axis = 2, name = 'bn1')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(2, name='max_pool1')(X)
    
 
   
    # CONV -> BN -> RELU Block applied to X
    X = Conv1D(64, 5, strides = 1, name = 'conv2')(X)
    X = BatchNormalization(axis = 2, name = 'bn2')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(2, name='max_pool2')(X)
    
 
 
    # CONV -> BN -> RELU Block applied to X
    X = Conv1D(128, 3, strides = 1, name = 'conv3')(X)
    X = BatchNormalization(axis = 2, name = 'bn3')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(2, name='max_pool3')(X)
    
 
    # CONV -> BN -> RELU Block applied to X
    X = Conv1D(64, 1, strides = 1, name = 'conv4')(X)
    X = BatchNormalization(axis = 2, name = 'bn4')(X)
    X = Activation('relu')(X)

    #layer group5 4*4*32
  
 
    X = Conv1D(32, 3, strides = 1, name = 'conv5')(X)
    X = BatchNormalization(axis = 2, name = 'bn5')(X)
    X = Activation('relu')(X)
    X = MaxPooling1D(2, name='max_pool5')(X)
    
    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(128, activation='sigmoid', name='fc1')(X)
    X = Dense(32, activation='sigmoid', name='fc2')(X)
    X = Dense(4, activation='sigmoid', name='fc3')(X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='DocumentModel')
    
    ### END CODE HERE ###
    
    return model

happyModel = DocumentModel(((300,300)))
happyModel.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = ["accuracy"])

from keras.utils.np_utils import to_categorical

categorical_labels = to_categorical(y_train, num_classes=None)

happyModel.fit(x = X_train, y = categorical_labels, epochs = 1, batch_size = 32)
# serialize model to JSON
model_json = happyModel.to_json()
with open("happyModel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
pred=happyModel.predict(X_test)
pred
y_classes = pred.argmax(axis=-1)
from sklearn.metrics import   accuracy_score
print('accuracy ' , accuracy_score(y_classes,y_test))



import sys
test_img=scipy.misc.imread(str(sys.argv[1]),mode='L')
test_img=scipy.misc.imresize(test_img,(300,300))
test_df=np.empty(shape=(1,300,300))
test_df[0]=test_img
test_pred=happyModel.predict(test_df)
class_label=test_pred.argmax(axis=-1)
print("The picture queried  has  label : ",class_label)

print(test_pred)


