#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
the training program that trains autoencoder, initial classifier and final classifier; also saves all the model and their corresponding weights.

@author: mannan
"""

import argparse
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD, RMSprop
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from load_cifar_10 import load_cifar_10_data
from display import loss_plot_classifier, aug_display
import numpy as np
from net.autoencodeclassifynet5 import AutoencodeClassifyNet5
from display import display_data, loss_plot_ac, compare


def main():
     
    input_img = Input(shape = (args.IMAGE_DIMS))
    
    print("[INFO] loading cifer 10 dataset...")
    train_data, _, train_labels, test_data, _, test_labels, _ = load_cifar_10_data(args.dataset)                   # load data
    
    s = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)                                         # spliting data for autoencoder training
    for train_index, test_index in s.split(train_data, train_labels):
        train_X, train_ground = train_data[train_index], train_data[train_index]
        valid_X, valid_ground = train_data[test_index], train_data[test_index]
        
    print("[INFO] building autoencoder...")
    autoencoder = AutoencodeClassifyNet5.build_ae(input_img)                                                       # defining autoencoder
    autoencoder.compile(optimizer=Adam(), loss='mean_squared_error')
    autoencoder.summary()
    
    print("[INFO] starting autoencoder training...")
    epochs_ae = 250
    autoencoder_train = autoencoder.fit(train_X, 
                                        train_ground, 
                                        batch_size = args.BS, 
                                        epochs = epochs_ae, 
                                        verbose = 1, 
                                        validation_data = (valid_X, valid_ground),
                                        shuffle = True)
      
    print("[INFO] evaluating autoencoder...")
    score = autoencoder.evaluate(test_data, test_data, verbose=1)                                                  # evaluating autoencoder
    print(score)
    loss_plot_ac(autoencoder_train, epochs_ae, args.images)   
    result_test = autoencoder.predict(test_data)
    result_val = autoencoder.predict(valid_X)
    compare(test_data, result_test, args.images+'/original_vs_generated_test.png')
    compare(valid_X, result_val, args.images+'/original_vs_generated_valid.png')
    autoencoder.save_weights(args.model_ae + '/autoencoder.h5')
    
    print("[INFO] preparing data for classifier...")  
    train_Y_one_hot = to_categorical(train_labels)                                                                 # preparing training data for classifier
    s = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=44)
    for train_index, test_index in s.split(train_data, train_Y_one_hot):   
        X_train, X_valid = train_data[train_index], train_data[test_index]
        Y_train, Y_valid = train_Y_one_hot[train_index], train_Y_one_hot[test_index]
    model_ae_path = args.model_ae + '/autoencoder.h5'
    
    print("[INFO] building classifier...")
    model_classify = AutoencodeClassifyNet5.build_classifier(input_img, args.num_classes, model_ae_path)          # define classifier
    aug = ImageDataGenerator(rotation_range = 1,
                             width_shift_range = 0.1,                                                             # data augmentation
                             height_shift_range = 0.1,
#                             shear_range = 0.1,
#                             zoom_range = 0.1,
                             horizontal_flip=True, fill_mode="nearest")    
    opt = Adam(lr = args.learning_rate)
    model_classify.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model_classify.summary()
    
    print("[INFO] initial training of classifier only for FC layers...")
    epochs_cl_ini = 350
    classify_train = model_classify.fit_generator(aug.flow(X_train, Y_train, batch_size=args.BS), 
                                                    steps_per_epoch=len(X_train) // args.BS,
                                                    epochs=epochs_cl_ini,
                                                    verbose=1,
                                                    validation_data=(X_valid, Y_valid))
    
    
    print("[INFO] saving initial classifier model...")
    model_classify.save_weights(args.model_cl + '/model_classification_initial.h5') 
    model_json = model_classify.to_json()
    with open(args.model_cl + "/model_classification_initial.json", "w") as json_file:
        json_file.write(model_json)
    
    loss_plot_classifier(classify_train, args.epochs, args.images+'/initial')                                      # plot the learning curves
    
    
    
    for layer in model_classify.layers[0:16]:                                                                      # make convolution layers trainable
        layer.trainable = True

       
    opt = Adam(lr = args.learning_rate)
    model_classify.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    
    chkpt = args.model_cl + '/classification_complete_Deep_weights.{epoch:03d}_{acc:.4f}_{val_acc:.4f}.h5'
    cp_cb = ModelCheckpoint(filepath = chkpt, monitor='val_acc', verbose = 1, save_best_only = True, mode='auto')
    
    print("[INFO] final training of classifier...")
    classify_train = model_classify.fit_generator(aug.flow(X_train, Y_train, batch_size=args.BS), 
                                              validation_data=(X_valid, Y_valid),
                                              steps_per_epoch=len(X_train) // args.BS,
                                              epochs = args.epochs,
                                              verbose=1,
                                              callbacks = [cp_cb])
    print("[INFO] saving final classifier model...") 
    model_json = model_classify.to_json()
    with open(args.model_cl + "/model_classification_final.json", "w") as json_file:
        json_file.write(model_json)
    
    loss_plot_classifier(classify_train, args.epochs, args.images)                                                 # plot the learning curves
    
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="construct a autoencoder")
    
    parser.add_argument("-d", "--dataset", required=True,
	                   help="path to input dataset")
    
    parser.add_argument("-m_ae", "--model_ae", required=True,
	                   help="path to autoencoder model")
    
    parser.add_argument("-mc", "--model_cl", required=True,
	                   help="path to classification model")
    
    parser.add_argument("-i", "--images", required=True,
	                   help="path to generated images")
    
    parser.add_argument('--epochs', type=float, default = 150,
                        help='number of epoch')

    parser.add_argument('--BS', type=int, default = 64, 
                        help='batch size')
    
    parser.add_argument('--learning_rate', type=float, default = 5e-4,
                        help='learning rate')
    
    parser.add_argument('--IMAGE_DIMS', type=int, default = (32, 32, 3), 
                        help='image dimension')

    parser.add_argument("-c", "--num_classes", type=str, default = 10,
                        help="number of classes")


    args = parser.parse_args()
    main()
