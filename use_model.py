from numpy import loadtxt
from keras.models import load_model
import tensorflow as tf
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from urllib.request import urlopen
from PIL import Image
model = load_model('model-colab.h5')
# model.summary()

def load_image(image_url, show=False):
    try:
        img_tensor = loadImageURL(image_url)               # (height, width, channels)
        img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
        img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

        if show:
            plt.imshow(img_tensor[0])
            plt.axis('off')
            plt.show()

        return img_tensor
    except Exception as e:
        print("Error")
# 
def loadImageURL(URL):
    with urlopen(URL) as url:
        img = Image.open(BytesIO(url.read()))
        img = img.convert('RGB')
        img = img.resize((150,150),Image.NEAREST)
        img = image.img_to_array(img)
    return image.img_to_array(img)
    
#predict an image
def classify(IMAGE_URI):
    try:
        new_image = load_image(IMAGE_URI)
        # check prediction
        pred = model.predict(new_image)
        # Positive numbers predict class 1, negative numbers predict class 0.
        class_name = ['architecture', 'art', 'cosplay', 'decor', 'fashion', 'food', 'landscape']
        class_predict = class_name[np.argmax(pred)]
        return class_predict
    except Exception as e:
        return None

#predit some images

