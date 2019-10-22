#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
the evaluation program that evaluates the performance of the classifier.

@author: mannan
"""

import argparse
import numpy as np
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import model_from_json
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split
from load_cifar_10 import load_cifar_10_data
from display import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import glob
import os

def select_latest_model(path):
    # find the last saved file
    list_of_files = glob.glob(path + '/*.h5') # * means all if need specific format then *.h5
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def main():
    # Loading data
    print("[INFO] loading cifer 10 dataset...")
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = load_cifar_10_data(args.dataset)
    train_Y_one_hot = to_categorical(train_labels)
    test_Y_one_hot = to_categorical(test_labels)
    train_X, valid_X, train_label, valid_label = train_test_split(train_data, train_Y_one_hot, test_size=0.5, random_state=44)

    # load json and create model
    json_file = open(args.model_cl + '/model_classification_final.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    full_model = model_from_json(loaded_model_json)
    # load weights from highest accuracy epoch  
    full_model.load_weights(select_latest_model(args.model_cl))
    print("[INFO] Loaded model from disk")
     
    # evaluate loaded model on test data
    print("[INFO] evaluating classifier...")
    full_model.compile(loss=categorical_crossentropy, optimizer=Adam(), metrics=['accuracy'])
    score_val = full_model.evaluate(valid_X, valid_label, verbose=0)
    print("Validation set %s: %.2f%%" % (full_model.metrics_names[1], score_val[1]*100))
    
    score_test = full_model.evaluate(test_data, test_Y_one_hot, verbose=0)
    print("Test set %s: %.2f%%" % (full_model.metrics_names[1], score_test[1]*100))
    
    Y_pred = full_model.predict(test_data)
    Y_pred_classes = np.argmax(Y_pred, axis = 1)
    Y_true = np.argmax(test_Y_one_hot, axis = 1)
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, label_names, args.images) 
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="construct a autoencoder")
    
    parser.add_argument("-d", "--dataset", required=True,
	                   help="path to input dataset")
    
    parser.add_argument("-mc", "--model_cl", required=True,
	                   help="path to the classification model")
    
    parser.add_argument("-i", "--images", required=True,
	                   help="path to generated images")

    args = parser.parse_args()
    main()