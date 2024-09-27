import os
import shutil
import numpy as np
from os import path
import json
np.random.seed(123)
from configs.configuration import Config


def separate_dataset(input_data_path, output_train_path, output_test_path):

    imageFiles = os.listdir(input_data_path + 'images/')

    num_train_set = int(len(imageFiles) * 0.7)
    num_test_set = int(len(imageFiles)) - num_train_set
    train_set = np.random.choice(imageFiles, size=num_train_set, replace=False)

    # print(len(train_set))
    # print(train_set[0])

    for imageFile in train_set:

        image_src = input_data_path + '/images/' + imageFile
        image_dst = output_train_path + "/images/" + imageFile

        mask_src = input_data_path + '/masks/' + imageFile
        mask_dst = output_train_path + "masks/" + imageFile

        if os.path.isfile(image_src):
            shutil.copy(image_src, image_dst)
        else:
            print("Error at image: ", imageFile)

        if os.path.isfile(mask_src):
            shutil.copy(mask_src, mask_dst)
        else:
            print("Error at image: ", imageFile)

    test_set = np.random.choice(imageFiles, size=num_test_set, replace=False)

    for imageFile in test_set:

        image_src = input_data_path + '/images/' + imageFile
        image_dst = output_test_path + "/images/" + imageFile

        mask_src = input_data_path + '/masks/' + imageFile
        mask_dst = output_test_path + "/masks/" + imageFile

        if os.path.isfile(image_src):
            shutil.copy(image_src, image_dst)
        else:
            print("Error at image: ", imageFile)

        if os.path.isfile(mask_src):
            shutil.copy(mask_src, mask_dst)
        else:
            print("Error at image: ", imageFile)

if __name__ == '__main__':

    ROOT_DATA_DIR = Config.root_data_dir

    DATA_VERSION = Config.data_version

    DATA_DIR = ROOT_DATA_DIR

    INPUT_IMAGE_DIR = DATA_DIR + 'original/'

    OUTPUT_TRAIN_SET = DATA_DIR + 'train/'
    OUTPUT_TEST_SET = DATA_DIR + 'test/'

    OUTPUT_TRAIN_IMAGE = OUTPUT_TRAIN_SET + 'images/'
    OUTPUT_TRAIN_MASK = OUTPUT_TRAIN_SET + 'masks'

    OUTPUT_TEST_IMAGE = OUTPUT_TEST_SET + 'images/'
    OUTPUT_TEST_MASK = OUTPUT_TEST_SET + 'masks'

    os.makedirs(OUTPUT_TRAIN_IMAGE, exist_ok=True)
    os.makedirs(OUTPUT_TEST_IMAGE, exist_ok=True)
    os.makedirs(OUTPUT_TRAIN_MASK, exist_ok=True)
    os.makedirs(OUTPUT_TEST_MASK, exist_ok=True)

    separate_dataset(INPUT_IMAGE_DIR, OUTPUT_TRAIN_SET, OUTPUT_TEST_SET)

