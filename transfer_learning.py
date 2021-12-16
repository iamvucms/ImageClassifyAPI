import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import  ImageDataGenerator  
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

PATH = r'datasets/'

train_dir = os.path.join(PATH, 'train')
val_dir = os.path.join(PATH, 'validation')


#get so luong data co duoc
# train_cats_dir = os.path.join(train_dir,'cats')
# train_dogs_dir = os.path.join(train_dir,'dogs')
# val_cats_dir = os.path.join(val_dir,'cats')
# val_dogs_dir = os.path.join(val_dir,'dogs')

train_architecture_dir = os.path.join(train_dir, 'architecture')
train_art_dir = os.path.join(train_dir, 'art')
train_cosplay_dir = os.path.join(train_dir, 'cosplay')
train_decor_dir = os.path.join(train_dir, 'decor')
train_fashion_dir = os.path.join(train_dir, 'fashion')
train_food_dir = os.path.join(train_dir, 'food')
train_landscape_dir = os.path.join(train_dir, 'landscape')

val_architecture_dir = os.path.join(train_dir, 'architecture')
val_art_dir = os.path.join(train_dir, 'art')
val_cosplay_dir = os.path.join(train_dir, 'cosplay')
val_decor_dir = os.path.join(train_dir, 'decor')
val_fashion_dir = os.path.join(train_dir, 'fashion')
val_food_dir = os.path.join(train_dir, 'food')
val_landscape_dir = os.path.join(train_dir, 'landscape')

print(os.listdir(train_architecture_dir))

# num_cats_tr = len(os.listdir(train_cats_dir))
# num_dogs_tr = len(os.listdir(train_dogs_dir))
# num_cats_val = len(os.listdir(val_cats_dir))
# num_dogs_val = len(os.listdir(val_dogs_dir))

num_train_architecture = len(os.listdir(train_architecture_dir))
num_train_art = len(os.listdir(train_art_dir))
num_train_decor = len(os.listdir(train_decor_dir))
num_train_fashion = len(os.listdir(train_fashion_dir))
num_train_cosplay = len(os.listdir(train_cosplay_dir))
num_train_food = len(os.listdir(train_food_dir))
num_train_landscape = len(os.listdir(train_landscape_dir))

num_val_architecture = len(os.listdir(train_architecture_dir))
num_val_art = len(os.listdir(train_art_dir))
num_val_decor = len(os.listdir(train_decor_dir))
num_val_fashion = len(os.listdir(train_fashion_dir))
num_val_cosplay = len(os.listdir(train_cosplay_dir))
num_val_food = len(os.listdir(train_food_dir))
num_val_landscape = len(os.listdir(train_landscape_dir))

total_train = num_train_architecture + num_train_art + num_train_decor + num_train_fashion + num_train_cosplay + num_train_food + num_train_landscape
total_val  = num_val_architecture + num_val_art + num_val_decor + num_val_fashion + num_val_cosplay + num_val_food + num_val_landscape

batch_size = 128
epochs = 20
# IMG_HEIGHT = 150
# IMG_WIDTH = 150

IMG_HEIGHT = 299
IMG_WIDTH = 299
#3.generator data for train

image_gen_val = ImageDataGenerator(rescale=1./255)
#data aumentation with imagedataGenerator
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )

#if have some classes, please change class_mode='categorical'
train_data_gen = image_gen_train.flow_from_directory(batch_size=batch_size,
                                                       directory=train_dir,
                                                       shuffle=True,
                                                       target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                       class_mode='categorical')


val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                      directory=val_dir,
                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                      class_mode='categorical')

#import pretrained model
IMG_SIZE = 299
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
# base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
#                                                include_top=False,
#                                                weights='imagenet')

base_model = tf.keras.applications.Xception(input_shape=IMG_SHAPE,
                                            include_top=False,
                                            weights='imagenet')

base_model.trainable = False
base_model.summary()

# feature_extract = base_model(val_data_gen[1])
# print(feature_extract)

sample_training_image, label_test = next(train_data_gen)
print(sample_training_image[0])
# print(label_test)
# print(type(label_test))

# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(sample_training_image[i], cmap=plt.cm.binary)
#     plt.xlabel(str(label_test[i]))
# plt.show()


global_average_pooling = tf.keras.layers.GlobalAveragePooling2D()
predict_layer = tf.keras.layers.Dense(1)

model = tf.keras.Sequential([
    base_model,
    global_average_pooling,
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# model.compile(optimizer='adam',
#               loss=tf.losses.BinaryCrossentropy(from_logits=True),
#               metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])

# model.summary()
#checkpoint best model
filepath = 'model/weights.best.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

#early stop when converging
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)


history = model.fit(train_data_gen,
                              epochs=50,
                              validation_data=val_data_gen,
                              steps_per_epoch=total_train // batch_size,
                              validation_steps= total_val // batch_size,
                              callbacks=[checkpoint, es])


model.save("model.h5")

initial_epochs = 10
validation_steps = 20
print("[info] evaluate model")
loss0, accuracy0 = model.evaluate(val_data_gen, steps=validation_steps)

#model.save("model.h5")




