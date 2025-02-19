import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


# 50x50 pixels
img_size = 50


# define folder locations
benign_training_loc = "melanoma_cancer_dataset/train/benign/"
malignant_training_loc = "melanoma_cancer_dataset/train/malignant/"
benign_testing_loc = "melanoma_cancer_dataset/test/benign/"
malignant_testing_loc = "melanoma_cancer_dataset/test/malignant/"

# np arrays with parsed data
benign_training_data = []
malignant_training_data = []
benign_testing_data = []
malignant_testing_data = []


# one-hot vectors
# [1,0] = benign
# [0,1] = melanoma


# benign training loader
for filename in os.listdir(benign_training_loc):
    try:
        path = benign_training_loc+filename
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # grayscale
        img = cv2.resize(img, (img_size, img_size))
        img_array = np.array(img)
        benign_training_data.append([img_array, np.array([1,0])]) # one-hot vector 1,0 = benign 
    
    except:
        pass # do nothing


# malignant training loader
for filename in os.listdir(malignant_training_loc):
    try:
        path = malignant_training_loc+filename
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # grayscale
        img = cv2.resize(img, (img_size, img_size))
        img_array = np.array(img)
        malignant_training_data.append([img_array, np.array([0,1])]) # one-hot vector 0,1 = malignant
    
    except:
        pass # do nothing


# benign testing loader
for filename in os.listdir(benign_testing_loc):
    try:
        path = benign_testing_loc+filename
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # grayscale
        img = cv2.resize(img, (img_size, img_size))
        img_array = np.array(img)
        benign_testing_loc.append([img_array, np.array([1,0])]) # one-hot vector 1,0 = benign 
    
    except:
        pass # do nothing


# malignant testing loader
for filename in os.listdir(malignant_testing_loc):
    try:
        path = malignant_testing_loc+filename
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) # grayscale
        img = cv2.resize(img, (img_size, img_size))
        img_array = np.array(img)
        malignant_testing_loc.append([img_array, np.array([0,1])]) # one-hot vector 0,1 = malignant 
    
    except:
        pass # do nothing

