#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Supervised and unsupervised architecture

@author: mannan
"""

from keras import regularizers 
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, UpSampling2D, Activation
from keras.layers.normalization import BatchNormalization

class AutoencodeClassifyNet5:
    # define encoder
    @staticmethod
    def encoder(input_img):
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) 
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x) 
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) 
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(2, 2))(x) 
        x = Conv2D(128,  (3, 3), activation='relu', padding='same')(x) 
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        encoded = MaxPooling2D(pool_size=(2, 2))(x)
        return encoded 
    
    # define decoder
    @staticmethod
    def decoder(encoded):
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded) 
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x) 
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x) 
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x) 
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D((2,2))(x) 
        decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x) 
        return decoded
    
    # CNN + fully connected network for classification
    @staticmethod
    def cl(enco, num_classes):  
        
        enco = Conv2D(256, (3, 3), padding='same')(enco)
        enco = BatchNormalization()(enco)
        enco = Activation('relu')(enco)
        enco = Dropout(0.5)(enco)
        enco = Conv2D(256, (3, 3), padding='same')(enco)
        enco = BatchNormalization()(enco)
        enco = Activation('relu')(enco)
        enco = Dropout(0.5)(enco)     
        flat = Flatten()(enco)
        den = Dense(128, activation='relu')(flat)#, kernel_regularizer = regularizers.l2(0.001))(flat)
        den = BatchNormalization()(den)
        den = Dropout(0.5)(den)
        out = Dense(num_classes, activation='softmax')(den)
        return out
    
    @staticmethod
    def build_ae(input_img):
        encoder_out = AutoencodeClassifyNet5.encoder(input_img)
        decoder_out = AutoencodeClassifyNet5.decoder(encoder_out)
        #return autoencoder network 
        return Model(input_img, decoder_out)
    
    @staticmethod
    def build_classifier(input_img, num_classes, autoencoder_model_path):
        full_model = Model(input_img, AutoencodeClassifyNet5.cl(AutoencodeClassifyNet5.encoder(input_img), num_classes))
        autoencoder = AutoencodeClassifyNet5.build_ae(input_img)
        autoencoder.load_weights(autoencoder_model_path)
        #loading weights of encoder layers of autoencoder to the classifier network
        #and initially keep these layers of classifier nontrainable.
        for l1,l2 in zip(full_model.layers[:16],autoencoder.layers[0:16]):
            l1.set_weights(l2.get_weights())
        for layer in full_model.layers[0:16]:
            layer.trainable = False
        # return the constructed classification network architecture    
        return full_model
        
