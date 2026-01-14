

import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from tensorflow import keras
from PIL import Image
from tensorflow.keras.optimizers import Adam


train_dir = 'C:\\Users\\Lenovo\\OneDrive\\Desktop\\TR-103\\tt.v1i.folder\\train'
val_dir = 'TR-103//tt.v1i.folder//valid'
test_dir = 'TR-103//tt.v1i.folder//test'

train_num_Fracture = len(os.listdir(os.path.join(train_dir, 'Fracture')))
train_num_Normal = len(os.listdir(os.path.join(train_dir, 'Normal')))
print(f"train_num_Fracture: {train_num_Fracture}")
print(f"train_num_Normal: {train_num_Normal}")

val_num_Fracture = len(os.listdir(os.path.join(val_dir, 'Fracture')))
val_num_Normal = len(os.listdir(os.path.join(val_dir, 'Normal')))
print(f"val_num_Fracture: {val_num_Fracture}")
print(f"val_num_Normal: {val_num_Normal}")

test_num_Fracture = len(os.listdir(os.path.join(test_dir, 'Fracture')))
test_num_Normal = len(os.listdir(os.path.join(test_dir, 'Normal')))
print(f"test_num_Fracture: {test_num_Fracture}")
print(f"test_num_Normal: {test_num_Normal}")

print("")
train_num = train_num_Fracture+train_num_Normal
val_num = val_num_Fracture+val_num_Normal
test_num = test_num_Fracture+test_num_Normal
print(f"train_num: {train_num}")
print(f"val_num: {val_num}")
print(f"test_num: {test_num}")

print("")
num_Fracture = train_num_Fracture+ test_num_Fracture + val_num_Fracture
num_Normal = train_num_Normal + test_num_Normal + val_num_Normal
print(f"num_Fracture: {num_Fracture}")
print(f"num_Normal: {num_Normal}")

labels = ["Normal", "Fracture"]
img_size = 224

def load_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
                
    return np.array(data, dtype="object")

train = load_data(train_dir)
test = load_data(test_dir)
val = load_data(val_dir)

X_train = [] 
y_train = []
X_test = [] 
y_test = []
X_val = [] 
y_val = []

def split_x_y(data, x, y):
    for feature, label in data:
        x.append(feature)
        y.append(label)
    return np.array(x), np.array(y)

X_train, y_train = split_x_y(train, X_train, y_train)
X_test, y_test = split_x_y(test, X_test, y_test)
X_val, y_val = split_x_y(val, X_val, y_val)

print(X_train)

def augmentation_layers(x):
    x = keras.layers.Rescaling(1./255)(x)
    x = keras.layers.RandomRotation(0.05)(x)
    return x

j=0
fig=plt.figure(figsize=(6, 6))
for i in (0,1,2,-2,-3,-4):
    fig.add_subplot(2,3,j+1)
    plt.imshow(train[i][0])
    plt.title(labels[train[i][1]])
    plt.tight_layout()
    j+=1
plt.show()

j=0
fig=plt.figure(figsize=(6, 6))
for i in (4,5,-1):
    fig.add_subplot(2,3,j+1)
    plt.imshow(train[i][0])
    plt.title("Original Image")
    x = augmentation_layers(train[i][0])
    fig.add_subplot(2,3,j+4)
    plt.imshow(x)
    plt.title("Augmented Image")
    plt.tight_layout()
    j+=1
plt.show()

weight_for_num_Fracture = num_Fracture / (num_Normal + num_Fracture)
weight_for_num_Normal = num_Normal / (num_Normal + num_Fracture)

class_weight = {0: weight_for_num_Fracture, 1: weight_for_num_Normal}

print(f"Weight for class Fracture: {weight_for_num_Fracture:.2f}")
print(f"Weight for class Normal: {weight_for_num_Normal:.2f}")

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=2, verbose=1)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, 
                                               mode='min')

def visualise_model_metrics(model, history):
    evaluation = model.evaluate(x=X_test,y=y_test)
    print(f"Test Accuracy: {evaluation[1] * 100:.2f}%")

    evaluation = model.evaluate(x=X_train,y=y_train)
    print(f"Train Accuracy: {evaluation[1] * 100:.2f}%")
    
    plt.figure(figsize=(20,15))

    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label = "Training accuracy",marker='o')
    plt.plot(history.history['val_accuracy'], label="Validation accuracy",marker='o')
    plt.legend()
    plt.title("Training vs validation accuracy")


    plt.subplot(2,2,2)
    plt.plot(history.history['loss'], label = "Training loss",marker='o')
    plt.plot(history.history['val_loss'], label="Validation loss",marker='o')
    plt.legend()
    plt.title("Training vs validation loss")

    plt.show()

def load_classification_report(y_pred):
    print('Classification report')
    print()
    print(classification_report(y_true=y_test,y_pred=y_pred, target_names=labels))

def plot_roc(y_pred):
    fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure(figsize=(4, 4))
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

base_resnet50 = tf.keras.applications.ResNet50(
                    include_top=False,
                    weights="imagenet",
                    input_shape=(224,224,3),
                    pooling='avg',
                    classes=1)

for layer in base_resnet50.layers:
        layer.trainable=False

inputs = keras.layers.Input(shape=(224,224,3))

x = augmentation_layers(inputs)

x = base_resnet50(inputs)

x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512,activation='relu')(x)
x = layers.Dense(256,activation='relu')(x)
x = layers.Dense(128,activation='relu')(x)
x = layers.Dropout(0.5)(x)

predictions = layers.Dense(1,activation='sigmoid')(x)
resnet50 = keras.Model(inputs, predictions)


resnet50.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= 3e-4),loss='binary_crossentropy',
                    metrics=['accuracy'])

checkpoint_resnet50 = 'C:\\Users\\Lenovo\\OneDrive\\Desktop\\TR-103\\resnet_50.hdf5'

model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_resnet50,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

history_res50 = resnet50.fit(
    x=X_train, y=y_train, 
    epochs=30,
    validation_data=(X_val,y_val),
    class_weight=class_weight,
    callbacks = [reduce_lr, early_stopping, model_checkpoint],
    batch_size=32, validation_batch_size=8)

resnet50.load_weights(checkpoint_resnet50)

# Evaluate the model on the test data
test_loss, test_accuracy = resnet50.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


pred = resnet50.predict(X_test)
resnet50_pred = list(map(lambda x: 0 if x<0.5 else 1, pred))

load_classification_report(resnet50_pred)
plot_roc(resnet50_pred)

base_densenet121 = tf.keras.applications.DenseNet121(
                    include_top=False,
                    weights="imagenet",
                    input_shape=(224,224,3),
                    pooling='avg',
                    classes=1)

for layer in base_densenet121.layers:
        layer.trainable=False

inputs = layers.Input(shape=(224,224,3))

x = augmentation_layers(inputs)

x = base_densenet121(inputs)

x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512,activation='relu')(x)
x = layers.Dense(256,activation='relu')(x)
x = layers.Dense(128,activation='relu')(x)
x = layers.Dropout(0.5)(x)

predictions = tf.keras.layers.Dense(1,activation='sigmoid')(x)
densenet121 = keras.Model(inputs, predictions)

densenet121.compile(optimizer=tf.keras.optimizers.Adam(learning_rate= 1e-3),loss='binary_crossentropy',
                    metrics=['accuracy'])



history_densenet121 = densenet121.fit(
    x=X_train, y=y_train, 
    epochs=25,
    validation_data=(X_val,y_val),
    class_weight=class_weight,
    callbacks = [reduce_lr, early_stopping],
    batch_size=32, validation_batch_size=8)


# Evaluate the model on the test data
test_loss, test_accuracy = densenet121.evaluate(X_test, y_test)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

pred = densenet121.predict(X_test)
densenet121_pred = list(map(lambda x: 0 if x<0.5 else 1, pred))

load_classification_report(densenet121_pred)
plot_roc(densenet121_pred)

import numpy as np
from tensorflow.keras.preprocessing import image

# Load the image using Keras' image module
img_path = '/content/drive/MyDrive/CSV files/tt.v1i.folder/train/Normal/11_jpg.rf.cbe56dba323836a3c8fe4ba619186055.jpg'  # Replace with the path to your image
img = image.load_img(img_path, target_size=(224, 224))  # Resizing to match the model's input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

# Preprocess the image similar to how it was done for training/validation data
img_array = img_array / 255.0  # Normalize pixel values to [0, 1] (assuming this was done during training)

# Now, make predictions using the trained model
predictions_resnet50 = resnet50.predict(img_array)
predictions_densenet121 = densenet121.predict(img_array)
# Assuming predictions_resnet50 and predictions_densenet121 are obtained from the previous code

# Printing predictions for ResNet50
print("Predictions for ResNet50:")
print(predictions_resnet50)

# Printing predictions for DenseNet121
print("\nPredictions for DenseNet121:")
print(predictions_densenet121)




# Assuming predictions_resnet50 and predictions_densenet121 are obtained from the pre-trained models

# Define a threshold for classification
threshold = 0.5  # Adjust the threshold value as needed

# Check if the predicted probability for fractured class is greater than the threshold
fracture_prob_resnet50 = predictions_resnet50 # Replace fractured_class_index with the index for fractured class
fracture_prob_densenet121 = predictions_densenet121 # Replace fractured_class_index with the index for fractured class

if fracture_prob_resnet50 > threshold:
    print("ResNet50 predicts this image is fractured.")
else:
    print("ResNet50 predicts this image is normal.")

if fracture_prob_densenet121 > threshold:
    print("DenseNet121 predicts this image is fractured.")
else:
    print("DenseNet121 predicts this image is normal.")


from tensorflow.keras.preprocessing import image
import numpy as np

# Load the image using Keras' image module
new_img_path ='/content/drive/MyDrive/CSV files/tt.v1i.folder/train/Normal/11_jpg.rf.cbe56dba323836a3c8fe4ba619186055.jpg'
img = image.load_img(new_img_path, target_size=(224, 224))  # Resizing to match the model's input size
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension

# Preprocess the image similar to how it was done for training/validation data
img_array = img_array / 255.0  # Normalize pixel values to [0, 1]

# Now, make predictions using the trained model
predictions_resnet = resnet50.predict(img_array)
predictions_densenet=densenet121.predict(img_array)

# The 'predictions' array contains the probability for the positive class (fractured)
fracture_probability_resnet = predictions_resnet[0][0]  # Probability for fractured class (assuming labels are encoded as 0 for normal and 1 for fractured)
normal_probability_resnet = 1 - fracture_probability_resnet  # Probability for normal class
fracture_probability_densenet=predictions_densenet[0][0] 
normal_probability_densenet=1 - fracture_probability_densenet 
print("Probability for fracture_probability_resnet:", fracture_probability_resnet)
print("Probability for normal_probability_resnet:", normal_probability_resnet)
print("Probability for fracture_probability_densenet:", fracture_probability_densenet)
print("Probability for normal_probability_densenet:", fracture_probability_densenet)
