import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

data = pd.read_csv('C:/Users/jubai/Documents/CSE465/fer2013.csv')


data_train = data[data['Usage']=='Training'].copy()
data_val   = data[data['Usage']=='PublicTest'].copy()
data_test  = data[data['Usage']=='PrivateTest'].copy()
    
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

es = EarlyStopping(monitor='val_loss', patience = 10, mode = 'min', restore_best_weights=True)

class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

num_classes = 7 
width, height = 48, 48
num_epochs = 30
batch_size = 64
num_features = 64

def CRNO(df, dataName):
    df['pixels'] = df['pixels'].apply(lambda pixel_sequence: [int(pixel) for pixel in pixel_sequence.split()])
    data_X = np.array(df['pixels'].tolist(), dtype='float32').reshape(-1, width, height,1)/255.0   
    data_Y = to_categorical(df['emotion'], num_classes)  
    print(dataName, "_X shape: {}, ", dataName, "_Y shape: {}".format(data_X.shape, data_Y.shape))
    return data_X, data_Y

train_X, train_Y = CRNO(data_train, "train")
val_X, val_Y     = CRNO(data_val, "val")
test_X, test_Y   = CRNO(data_test, "test")

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
loaded_model = load_model('C:/Users/jubai/Documents/CSE465/model_file_50epochs.h5')
test_pred_prob = loaded_model.predict(test_X)
test_pred_labels = np.argmax(test_pred_prob, axis=1)
test_true_labels = np.argmax(test_Y, axis=1)
confusion_mat = confusion_matrix(test_true_labels, test_pred_labels)
accuracy = accuracy_score(test_true_labels, test_pred_labels)
precision = precision_score(test_true_labels, test_pred_labels, average='weighted')
recall = recall_score(test_true_labels, test_pred_labels, average='weighted')

