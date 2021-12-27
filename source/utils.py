import matplotlib.pyplot as plt
import os
from os.path import exists
from PIL import Image
import cv2


def save_figures(history, dir):
    # Create folder to store figures (if not exist)
    if not exists(dir): os.makedirs(dir)

    # Features to visualize
    acc = history.history['acc']
    loss = history.history['loss']
    val_acc = history.history['val_acc']
    val_loss = history.history['val_loss']
    labels = ['training' , 'validation']

    # Accuracy
    plt.plot(acc); plt.plot(val_acc)
    plt.title('Training and Validation Accuracy')
    plt.ylabel('Accuracy'); plt.xlabel('Epoch'); plt.ylim([0.25,1])
    plt.legend(labels, loc='lower right')
    plt.savefig(dir + 'accuracy.jpg')
    plt.close()

    # Loss
    plt.plot(loss); plt.plot(val_loss)
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss'); plt.xlabel('Epoch'); plt.ylim([0,0.75])
    plt.legend(labels, loc='upper right')
    plt.savefig(dir + 'loss.jpg')
    plt.close()


def remove_file_with_extension(path, extension):
    for fname in os.listdir(path):
         if fname.endswith(extension):
            os.remove(os.path.join(path, fname))


def remove_small_image(path):
    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        parent_dir = os.path.dirname(path)
        img_name = os.path.basename(path)
        h, w = img.shape
        if(h < 80) or (w < 80):
            os.remove(os.path.join(parent_dir, img_name))


def remove_all_small_images(dir):
    for subdir, _, fileList in os.walk(dir):
        for f in fileList:
            try:
                full_path = os.path.join(subdir, f)
                remove_small_image(full_path)
            except Exception as e:
                print('Unable to remove %s. Skipping.' % full_path)


def resize_image(path, new_size):
    if os.path.isfile(path):
        im = Image.open(path).resize(new_size, Image.ANTIALIAS)
        parent_dir = os.path.dirname(path)
        img_name = os.path.basename(path).split('.')[0]
        #im.save(os.path.join(parent_dir, img_name + '.png'), 'PNG', quality=100)
        im.save(os.path.join(parent_dir, img_name + '.jpg'), 'JPEG', quality=100)


def resize_all_images(dir, new_size):
    for subdir, _, fileList in os.walk(dir):
        for f in fileList:
            try:
                full_path = os.path.join(subdir, f)
                resize_image(full_path, new_size)
            except Exception as e:
                print('Unable to resize %s. Skipping.' % full_path)
