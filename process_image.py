import os

from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image

PATH_TRAIN = 'datasets/train/'
PATH_VAL = 'datasets/validation/'

architecture_path = PATH_TRAIN + 'architecture/'
art_path = PATH_TRAIN + 'art/'
cosplay_path = PATH_TRAIN + 'cosplay/'
decor_path = PATH_TRAIN + 'decor/'
fashion_path = PATH_TRAIN + 'fashion/'
food_path = PATH_TRAIN + 'food/'
landscape_path = PATH_TRAIN + 'landscape/'




def delete_error_image(class_path):
    list_images = os.listdir(class_path)
    # print(list_images)
    for image_name in list_images:
        link_an_image = os.path.join(class_path, image_name)
        try:
            pil_im = Image.open(link_an_image)
            #pil_im.show()
        except:
            os.remove(link_an_image)
            print("[DELETE] ", link_an_image)


delete_error_image(architecture_path)
delete_error_image(art_path)
delete_error_image(cosplay_path)
delete_error_image(decor_path)
delete_error_image(fashion_path)
delete_error_image(food_path)
delete_error_image(landscape_path)

#
# pil_im = Image.open('/home/phuc/Documents/code/AI/vucms/datasets/train/decor/img_75.jpeg')
# pil_im.show()
