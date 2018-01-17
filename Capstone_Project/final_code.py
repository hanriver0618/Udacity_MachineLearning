## Python Package
import cv2
import numpy as np
from tqdm import tqdm
import pandas as pd
                 
from os.path import exists, expanduser
import random
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from os import listdir
from os.path import isfile, join

from keras.layers import *
from keras.models import *
from keras.applications import *
from keras.optimizers import *
from keras.optimizers import Adam
from keras.regularizers import *
from keras.applications.inception_v3 import preprocess_input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.preprocessing import image 
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils

from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.datasets import load_files    

%matplotlib inline
%config InlineBackend.figure_format = 'retina'

## 12 seedlings
categories = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 
              'Common wheat', 'Fat Hen', 'Loose Silky-bent',
              'Maize', 'Scentless Mayweed', 'Shepherds Purse', 
              'Small-flowered Cranesbill', 'Sugar beet']

## Preprocessing 1 : load files and convert the labels for each seeding to binary class matrix
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np_utils.to_categorical(np.array(data['target']), 12)
    return files, targets

## Preprocessing 2: reshape all input images to the appropriate tensors with dimension, (Number, 3, width, height) 
def img_to_tensor(img_path,size):
    img = image.load_img(img_path, target_size=(size,size))
    x = image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def imgs_to_tensor(img_paths,size):
    list_of_tensors = [img_to_tensor(img_path,size) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)



## train(80%)/validation(10%)/test(10%) split 
labels = listdir("./train")
train_files, train_targets = load_dataset('./train')

y_train = train_targets
train_tensors = imgs_to_tensor(train_files,47).astype('float32')/255

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
for train_index, test_index in sss.split(train_tensors, y_train):
    train_tensors, valid_tensors = train_tensors[train_index], train_tensors[test_index]
    y_train, y_valid = y_train[train_index], y_train[test_index]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42) 
for train_index, test_index in sss.split(valid_tensors, y_valid):
    valid_tensors, test_tensors = valid_tensors[train_index], valid_tensors[test_index]
    y_valid, y_test = y_valid[train_index], y_valid[test_index]
    
print(train_tensors.shape)
print(valid_tensors.shape)
print(test_tensors.shape)


## Model 1
## Build Model 1, adding each layer 
model1 = Sequential()
model1.add(Conv2D(filters=32,kernel_size=2, activation='relu',
                 input_shape=train_tensors.shape[1:]))
model1.add(Conv2D(filters=32,kernel_size=2, activation='relu'))
model1.add(MaxPooling2D())
model1.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
model1.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
model1.add(MaxPooling2D())
model1.add(Conv2D(filters=128,kernel_size=2, activation='relu'))
model1.add(Conv2D(filters=128,kernel_size=2, activation='relu'))
model1.add(MaxPooling2D())
model1.add(GlobalAveragePooling2D())
model1.add(Dense(1024,activation='relu'))
model1.add(Dense(1024,activation='relu'))
model1.add(Dense(12,activation='softmax'))

## Model 3
## Build Model 3, adding each layer
model3 = Sequential()
model3.add(Conv2D(filters=64,kernel_size=2, activation='relu',
                 input_shape=train_tensors.shape[1:]))
model3.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
model3.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model3.add(Dropout(0.2))

model3.add(Conv2D(filters=128,kernel_size=2, activation='relu'))
model3.add(Conv2D(filters=128,kernel_size=2, activation='relu'))
model3.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model3.add(Dropout(0.2))

model3.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
model3.add(Conv2D(filters=64,kernel_size=2, activation='relu'))
model3.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model3.add(Dropout(0.2))

model3.add(GlobalAveragePooling2D())
model3.add(Dense(512,activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(256,activation='relu'))
model3.add(Dropout(0.2))
model3.add(Dense(12,activation='softmax'))

## Final : Train Model 1 (rmsprop)
model1.compile(optimizer=RMSprop(), loss='categorical_crossentropy',metrics=['accuracy'])

## save the model weight using ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='weights.model1_rmsprop.hdf5', 
                           verbose=1, save_best_only=True)

## train the model1 with train dataset and validation dataset with epoch=20 and batch_size=20
model1.fit(train_tensors, y_train, 
          validation_data=(valid_tensors, y_valid),
          epochs=20, batch_size=20, callbacks=[checkpointer], verbose=1)

## load the weight and make a prediction using test data set.
model1.load_weights('weights.model1_rmsprop.hdf5')
predictions = [np.argmax(model1.predict(np.expand_dims(feature, axis=0))) 
               for feature in test_tensors]
y_pred = [labels[i] for i in predictions]
test_list = y_test.argmax(axis=1)

print(f1_score(test_list, predictions, average='macro')) 
print(accuracy_score(test_list,predictions))

## train model1 with image augmentation using ImageDataGenerator with epoch =50 and batch_size=50
epochs=50
batch_size=32
model1.compile(optimizer=RMSprop(), loss='categorical_crossentropy',metrics=['accuracy'])
checkpointer = [EarlyStopping(monitor='val_loss', patience=5, verbose=0), 
            ModelCheckpoint(filepath='weights.model1_rmsprop_with_aug.hdf5', monitor='val_loss', save_best_only=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]

datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True)

model1.fit_generator(datagen.flow(train_tensors, y_train, batch_size=batch_size),
                    steps_per_epoch=len(train_tensors)/batch_size, 
                    validation_data=datagen.flow(valid_tensors, y_valid, batch_size=batch_size), 
                    validation_steps=len(valid_tensors)/batch_size,
                    callbacks=checkpointer,
                    epochs=epochs, 
                    verbose=1)

## load the weight and make a prediction using test data set.
model1.load_weights('weights.model1_rmsprop_with_aug.hdf5')
predictions = [np.argmax(model1.predict(np.expand_dims(feature, axis=0))) 
               for feature in test_tensors]
y_pred = [labels[i] for i in predictions]
test_list = y_test.argmax(axis=1)

print(f1_score(test_list, predictions, average='weighted')) 
print(accuracy_score(test_list,predictions))


## Check a confusion matrix with the result using test dataset.
confusion = confusion_matrix(test_list,predictions)
abbreviation = ['BG', 'Ch', 'Cl', 'CC', 'CW', 'FH', 'LSB', 'M', 'SM', 'SP', 'SFC', 'SB']
pd.DataFrame({'class': categories, 'abbreviation': abbreviation})

## Plot Confusion Matrix
# import seaborn as sns
# fig, ax = plt.subplots(1)
# ax = sns.heatmap(confusion, ax=ax, cmap=plt.cm.Oranges, annot=True)
# ax.set_xticklabels(abbreviation)
# ax.set_yticklabels(abbreviation)
# plt.title('Confusion Matrix',size=20)
# plt.ylabel('True',size=16)
# plt.xlabel('Predicted',size=16)
# plt.show();

## Apply the model to the test file and save the result for kaggle posting
model1.load_weights('weights.model1_rmsprop_with_aug.hdf5')
data=load_files('./test')
final_X_test = np.array(data['filenames'])
final_test_tensors = imgs_to_tensor('./test/'+df_test.file.values,47).astype('float32')/255

predictions = [np.argmax(model1.predict(np.expand_dims(feature, axis=0))) 
               for feature in final_test_tensors]
y_pred = [labels[i] for i in predictions]

df = pd.DataFrame(data={'file': df_test['file'], 'species': y_pred})
df_sort = df.sort_values(by=['file'])
df_sort.to_csv('final.csv', index=False)

## Appendix : Model with Xception
## I used the image size with 128 x 128 using Xception (47 x 47 is not allowed to use Xception model)

## train(80%)/validation(10%)/test(10%) split
labels = listdir("./train")
train_files, train_targets = load_dataset('./train')

y_train = train_targets
train_tensors = imgs_to_tensor(train_files,128).astype('float32')/255

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42) 
for train_index, test_index in sss.split(train_tensors, y_train):
    train_tensors, valid_tensors = train_tensors[train_index], train_tensors[test_index]
    y_train, y_valid = y_train[train_index], y_train[test_index]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42) 
for train_index, test_index in sss.split(valid_tensors, y_valid):
    valid_tensors, test_tensors = valid_tensors[train_index], valid_tensors[test_index]
    y_valid, y_test = y_valid[train_index], y_valid[test_index]

## load Xception model from keras package
pre_train = Xception(input_shape=(128,128, 3), include_top=False, weights='imagenet', pooling='avg')

## complete the model with Xception with a fully connected layer (and dropout)
x = pre_train.output
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(12, activation='softmax')(x)
model_Xception = Model(inputs=pre_train.input, outputs=predictions)

## compile the model with Xception using adam algorithm as an optimizer.
model_Xception.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',metrics=['accuracy']) 

checkpointer = ModelCheckpoint(filepath='weights.Xception.hdf5', 
                               verbose=1, save_best_only=True)
## train the model
model_Xception.fit(train_tensors, y_train, 
          validation_data=(valid_tensors, y_valid),
          epochs=5, batch_size=32, callbacks=[checkpointer], verbose=1)

## load the model weights and predict with test dataset.
model.load_weights('weights.Xception.hdf5')
predictions = [np.argmax(model_Xception.predict(np.expand_dims(feature, axis=0))) 
               for feature in test_tensors]
y_pred = [labels[i] for i in predictions]
test_list = y_test.argmax(axis=1)

print(f1_score(test_list, predictions, average='weighted')) 
print(accuracy_score(test_list,predictions))

## train the model with Image Augmentation using ImageDataGenerator
model_Xception.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(), metrics=['accuracy'])

datagen = ImageDataGenerator( horizontal_flip=True, 
                              vertical_flip=True)
                                      
checkpointer = [ EarlyStopping(monitor='val_loss', patience=5, verbose=0), 
              ModelCheckpoint(filepath='weights.Xception_with_aug.hdf5', monitor='val_loss', save_best_only=True, verbose=0),
              ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)]
epochs=5
batch_size=32
model_Xception.fit_generator(datagen.flow(train_tensors, y_train, batch_size=batch_size),
                    steps_per_epoch=len(train_tensors)/batch_size, 
                    validation_data=datagen.flow(valid_tensors, y_valid, batch_size=batch_size), 
                    validation_steps=len(valid_tensors)/batch_size,
                    callbacks=checkpointer,
                    epochs=epochs, 
                    verbose=1)

## load the model weights and predict with test dataset
model_Xception.load_weights('weights.Xception_with_aug.hdf5')
predictions = [np.argmax(model_Xception.predict(np.expand_dims(feature, axis=0))) 
               for feature in test_tensors]
y_pred = [labels[i] for i in predictions]
test_list = y_test.argmax(axis=1)

print(f1_score(test_list, predictions, average='weighted')) 
print(accuracy_score(test_list,predictions))
