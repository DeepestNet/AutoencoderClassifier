#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""


@author: mannan
"""
from matplotlib import pyplot as plt
import numpy as np
import itertools

"""
display and save various graph and image into the disk

"""

def display_data(train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names):
    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)

    # display some random training images in a 25x25 grid
    num_plot = 5
    f, ax = plt.subplots(num_plot, num_plot)
    for m in range(num_plot):
        for n in range(num_plot):
            idx = np.random.randint(0, train_data.shape[0])
            ax[m, n].imshow(train_data[idx])
            ax[m, n].get_xaxis().set_visible(False)
            ax[m, n].get_yaxis().set_visible(False)
    f.subplots_adjust(hspace=0.1)
    f.subplots_adjust(wspace=0)
    plt.show()
    
def loss_plot_ac(model, epochs, path):
    #display autoencoder loss and validation loss and save image in images folder
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epochs = range(epochs)
    plt.figure()
    plt.plot(epochs, loss, 'k', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(path+'/autoencoder_loss.png')
    plt.show()
    
    
def compare(original, detected, path, num=10):
    # show some random images and corresponding generated images by autoencoder
    n = num
    plt.figure(figsize=(20, 4))

    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i+1)
        plt.imshow(original[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # display reconstruction
        ax = plt.subplot(2, n, i +1 + n)
        plt.imshow(detected[i].reshape(32, 32, 3))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.savefig(path)
    plt.show()
    
    
def loss_plot_classifier(model, epoch, path):
    # display classifier loss, validation loss and classifier accuracy, validation accuracy with epoch
    accuracy = model.history['acc']
    val_accuracy = model.history['val_acc']
    loss = model.history['loss']
    val_loss = model.history['val_loss']
    epoch = range(len(accuracy))
    plt.plot(epoch, accuracy, 'k', label='Training accuracy')
    plt.plot(epoch, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(path+'/classifier_accuracy.png')
    plt.figure()
    plt.plot(epoch, loss, 'k', label='Training loss')
    plt.plot(epoch, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(path+'/classifier_loss.png')
    plt.show()
        
def aug_display(aug, train_X, train_label):
        aug.fit(train_X)
    
        # configure batch size and retrieve one batch of images
        for X_batch, y_batch in aug.flow(train_X, train_label, batch_size=9):
            # create a grid of 3x3 images
            for i in range(0, 9):
                plt.subplot(330 + 1 + i)
                plt.imshow(X_batch[i].reshape(32, 32, 3), cmap=plt.get_cmap('gray'))             
            # show the plot
            plt.show()
            
            a = X_batch[0,:,:,0]
            break

def plot_confusion_matrix(cm, classes, path,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path+'/confusion_matrix.png')
    plt.show()
    