#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
import seaborn as sb
import KeyFunctions as me
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sympy.discrete.transforms import fwht, ifwht


# In[ ]:


RSvec = np.random.randint(1, 500, 10)
n = 4
print(RSvec)


# In[ ]:


for RandState in RSvec:


    # In[ ]:


    #Import Full Triclosan Dataset
    df, labels = me.ConstructCombinedTriclosanDataset()

    [train, test] = train_test_split(df, random_state = RandState, shuffle = True, train_size = 0.9, stratify = df.index)

    y_tn = train.index
    y_tt = test.index
    X_tt = test.to_numpy()
    X_tn = train.to_numpy()

    #Augment Data to 2000 Spectra
    X_tnAu, y_tnAu = me.AugmentData(X_tn, y_tn, 2000, df.columns.to_numpy(), False)


    # In[ ]:


    #Set Training Parameters
    verbose = 0
    epochsvec = [5, 20, 50]
    batch_sizevec = [10, 50, 100]
    epochs = epochsvec[1]
    batch_size = batch_sizevec[1]


    # In[ ]:


    #Scale X-Data with Training Xs
    scaler = StandardScaler()
    scaler.fit(X_tnAu)
    X_tnS = scaler.transform(X_tnAu)
    X_ttS = scaler.transform(X_tt)

    #Encode y-Data with Training ys
    encoder = LabelEncoder()
    encoder.fit(y_tnAu)
    y_tn_e = encoder.transform(y_tnAu)
    y_tn_p = np_utils.to_categorical(y_tn_e, num_classes = len(labels))
    y_tt_e = encoder.transform(y_tt)
    y_tt_p = np_utils.to_categorical(y_tt_e, num_classes = len(labels))


    #Reshape All Data to a 3D Tensor of Shape [Number of Spectra, Number of Timesteps(1), Number of Wavelengths]
    X_tn_p = X_tnS.reshape(X_tnS.shape[0], X_tnS.shape[1], 1)
    X_tt_p = X_ttS.reshape(X_ttS.shape[0], X_ttS.shape[1], 1)

    y_tnT = tf.convert_to_tensor(y_tn_p)
    y_ttT = tf.convert_to_tensor(y_tt_p)
    X_tnT = tf.convert_to_tensor(X_tn_p)
    X_ttT = tf.convert_to_tensor(X_tt_p)

    ytruth = tf.argmax(input = y_ttT, axis = 1).numpy()
    ytruth = encoder.inverse_transform(ytruth)


    # In[ ]:


    #Multi-class Classification with Keras

    n_timesteps, n_features, n_outputs = X_tn_p.shape[1], X_tn_p.shape[2], y_tn_p.shape[1]

    #Define Sequential Model - 1 Convolution Layer, 1 Dropout Layer, 1 Flatten Layer, 2 Dense Layers
    model = Sequential()
    model.add(Conv1D(filters = 64, kernel_size = 3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters = 64, kernel_size = 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Implement EarlyStopping
    stopper = tf.keras.callbacks.EarlyStopping(monitor = "val_loss",patience = 2,verbose = 0, restore_best_weights = True)

    #Fit Model
    history_sc = model.fit(X_tnT, y_tnT, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split = 0.1, callbacks = stopper)

    #Evaluate Model
    SCloss, SCaccuracy = model.evaluate(X_ttT, y_ttT, batch_size=batch_size, verbose=verbose)


    # In[ ]:


    #Make Prediction
    SCypred = model.predict(X_ttT)
    SCypred = tf.argmax(input = SCypred, axis = 1).numpy()
    SCypred = encoder.inverse_transform(SCypred)


    # In[ ]:


    print('SCALED')
    print('Test Acc', 'Test Loss', 'Train Acc', 'Train Loss', 'Val_Acc', 'Val_Loss', 'Regression Error')
    print(SCaccuracy, SCloss, history_sc.history['accuracy'][-1], history_sc.history['loss'][-1], history_sc.history['val_accuracy'][-1], history_sc.history['val_loss'][-1], me.Scorer(ytruth, SCypred, labels))


    # In[ ]:


    #Apply Fourier Transform to Training and Testing Data
    X_tnf = np.apply_along_axis(np.fft.fft, axis=1, arr=X_tnAu)
    X_ttf = np.apply_along_axis(np.fft.fft, axis=1, arr=X_tt)

    #Combine Real and Imaginary Part of FT in form [real, imaginary]
    X_tnf = np.append(X_tnf.real, X_tnf.imag, axis = 1)
    X_ttf = np.append(X_ttf.real, X_ttf.imag, axis = 1)
    X_tnf= X_tnf.astype('float32')
    X_ttf= X_ttf.astype('float32')

    #Scale X-Data with Training Xs
    scaler = StandardScaler()
    scaler.fit(X_tnf)
    X_tnf = scaler.transform(X_tnf)
    X_ttf = scaler.transform(X_ttf)

    #Encode y-Data with Training ys
    encoder = LabelEncoder()
    encoder.fit(y_tnAu)
    y_tn_e = encoder.transform(y_tnAu)
    y_tn_p = np_utils.to_categorical(y_tn_e, num_classes = len(labels))
    y_tt_e = encoder.transform(y_tt)
    y_tt_p = np_utils.to_categorical(y_tt_e, num_classes = len(labels))

    #Reshape All Data to a 3D Tensor of Shape [Number of Spectra, Number of Timesteps(1), Number of Wavelengths]
    X_tn_p = X_tnf.reshape(X_tnf.shape[0], X_tnf.shape[1], 1)
    X_tt_p = X_ttf.reshape(X_ttf.shape[0], X_ttf.shape[1], 1)

    y_tnT = tf.convert_to_tensor(y_tn_p)
    y_ttT = tf.convert_to_tensor(y_tt_p)
    X_tnT = tf.convert_to_tensor(X_tn_p)
    X_ttT = tf.convert_to_tensor(X_tt_p)


    # In[ ]:


    #Multi-class Classification with Keras

    n_timesteps, n_features, n_outputs = X_tn_p.shape[1], X_tn_p.shape[2], y_tn_p.shape[1]

    #Define Sequential Model - 1 Convolution Layer, 1 Dropout Layer, 1 Flatten Layer, 2 Dense Layers
    model = Sequential()
    model.add(Conv1D(filters = 64, kernel_size = 3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters = 64, kernel_size = 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Implement EarlyStopping
    stopper = tf.keras.callbacks.EarlyStopping(monitor = "val_loss",patience = 2,verbose = 0, restore_best_weights = True)

    #Fit Model
    history_ft = model.fit(X_tnT, y_tnT, epochs=epochs, batch_size=batch_size, verbose=verbose,  validation_split=0.1, callbacks = stopper)

    #Evaluate Model
    FTloss, FTaccuracy = model.evaluate(X_ttT, y_ttT, batch_size=batch_size, verbose=verbose)


    # In[ ]:


    #Make Prediction
    FTypred = model.predict(X_ttT)
    FTypred = tf.argmax(input = FTypred, axis = 1).numpy()
    FTypred = encoder.inverse_transform(FTypred)


    # In[ ]:


    print('Fourier')
    print('Test Acc', 'Test Loss', 'Train Acc', 'Train Loss', 'Val_Acc', 'Val_Loss', 'Regression Error')
    print(FTaccuracy, FTloss, history_ft.history['accuracy'][-1], history_ft.history['loss'][-1], history_ft.history['val_accuracy'][-1], history_ft.history['val_loss'][-1], me.Scorer(ytruth, FTypred, labels))


    # In[ ]:


    #Apply Welsh-Hadamard Transform to Training and Testing Data

    X_tnh = np.apply_along_axis(fwht, axis=1, arr=X_tnS)
    X_tth = np.apply_along_axis(fwht, axis=1, arr=X_ttS)
    X_tnh = X_tnh.astype('float32')
    X_tth = X_tth.astype('float32')

    #Scale X-Data with Training Xs
    scaler = StandardScaler()
    scaler.fit(X_tnh)
    X_tnh = scaler.transform(X_tnh)
    X_tth = scaler.transform(X_tth)

    #Encode y-Data with Training ys
    encoder = LabelEncoder()
    encoder.fit(y_tnAu)
    y_tn_e = encoder.transform(y_tnAu)
    y_tn_p = np_utils.to_categorical(y_tn_e, num_classes = len(labels))
    y_tt_e = encoder.transform(y_tt)
    y_tt_p = np_utils.to_categorical(y_tt_e, num_classes = len(labels))

    #Reshape All Data to a 3D Tensor of Shape [Number of Spectra, Number of Timesteps(1), Number of Wavelengths]
    X_tn_p = X_tnh.reshape(X_tnh.shape[0], X_tnh.shape[1], 1)
    X_tt_p = X_tth.reshape(X_tth.shape[0], X_tth.shape[1], 1)

    y_tnT = tf.convert_to_tensor(y_tn_p)
    y_ttT = tf.convert_to_tensor(y_tt_p)
    X_tnT = tf.convert_to_tensor(X_tn_p)
    X_ttT = tf.convert_to_tensor(X_tt_p)


    # In[ ]:


    #Multi-class Classification with Keras

    n_timesteps, n_features, n_outputs = X_tn_p.shape[1], X_tn_p.shape[2], y_tn_p.shape[1]

    #Define Sequential Model - 1 Convolution Layer, 1 Dropout Layer, 1 Flatten Layer, 2 Dense Layers
    model = Sequential()
    model.add(Conv1D(filters = 64, kernel_size = 3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters = 64, kernel_size = 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #Implement EarlyStopping
    stopper = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", mode = 'min',                                           patience = 2, verbose = 1, restore_best_weights = True)

    #Fit Model
    history_ht = model.fit(X_tnT, y_tnT, epochs=epochs, batch_size=batch_size, verbose=verbose,  validation_split=0.2, callbacks = stopper)

    #Evaluate Model
    HTloss, HTaccuracy = model.evaluate(X_ttT, y_ttT, batch_size=batch_size, verbose=verbose)


    # In[ ]:


    #Make Prediction
    HTypred = model.predict(X_ttT)
    HTypred = tf.argmax(input = HTypred, axis = 1).numpy()
    HTypred = encoder.inverse_transform(HTypred)


    # In[ ]:


    print('HADAMARD')
    print('Test Acc', 'Test Loss', 'Train Acc', 'Train Loss', 'Val_Acc', 'Val_Loss', 'Regression Error')
    print(HTaccuracy, HTloss, history_ht.history['accuracy'][-1], history_ht.history['loss'][-1], history_ht.history['val_accuracy'][-1], history_ht.history['val_loss'][-1], me.Scorer(ytruth, HTypred, labels))


    # In[ ]:


    print('----------------------------')

