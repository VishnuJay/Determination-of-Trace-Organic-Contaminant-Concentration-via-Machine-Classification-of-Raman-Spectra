#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from scipy import stats as stats
import matplotlib.pyplot as plt
import seaborn as sb
import KeyFunctions as me
import tensorflow as tf
RandState = 117

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, MaxPooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

n = 4


# In[2]:


from sklearn.model_selection import train_test_split

#Import Full Triclosan Dataset
df, labels = me.ConstructCombinedTriclosanDataset()

[train, test] = train_test_split(df, random_state = RandState, shuffle = True, train_size = 0.9, stratify = df.index)

y_tn = train.index
y_tt = test.index
X_tt = test.to_numpy()
X_tn = train.to_numpy()

#Augment Data to 2000 Spectra
X_tnAu, y_tnAu = me.AugmentData(X_tn, y_tn, 2000, df.columns.to_numpy(), False)


# In[3]:


df.index.value_counts()


# In[4]:


#Set Training Parameters
verbose = 1
epochsvec = [5, 20, 50]
batch_sizevec = [10, 50, 100]
epochs = epochsvec[1]
batch_size = batch_sizevec[1]


# In[5]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

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

display(X_tnT.shape)
display(y_tnT.shape)
display(X_ttT.shape)
display(y_ttT.shape)

ytruth = tf.argmax(input = y_ttT, axis = 1).numpy()
ytruth = encoder.inverse_transform(ytruth)


# In[6]:


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
model.fit(X_tnT, y_tnT, epochs=epochs, batch_size=batch_size, verbose=verbose, validation_split = 0.1, callbacks = stopper)

#Evaluate Model
_, SCaccuracy = model.evaluate(X_ttT, y_ttT, batch_size=batch_size, verbose=verbose)
display(SCaccuracy)


# In[7]:


#Make Prediction
SCypred = model.predict(X_ttT)
SCypred = tf.argmax(input = SCypred, axis = 1).numpy()
SCypred = encoder.inverse_transform(SCypred)


# In[8]:


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


display(X_tnT.shape)
display(y_tnT.shape)
display(X_ttT.shape)
display(y_ttT.shape)


# In[9]:


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
model.fit(X_tnT, y_tnT, epochs=epochs, batch_size=batch_size, verbose=verbose,  validation_split=0.1, callbacks = stopper)

#Evaluate Model
_, FTaccuracy = model.evaluate(X_ttT, y_ttT, batch_size=batch_size, verbose=verbose)
display(FTaccuracy)


# In[10]:


#Make Prediction
FTypred = model.predict(X_ttT)
FTypred = tf.argmax(input = FTypred, axis = 1).numpy()
FTypred = encoder.inverse_transform(FTypred)


# In[11]:


#Apply Welsh-Hadamard Transform to Training and Testing Data
from sympy.discrete.transforms import fwht, ifwht


#Scale X-Data with Training Xs
scaler = StandardScaler()
scaler.fit(X_tnAu)
X_tnS = scaler.transform(X_tnAu)
X_ttS = scaler.transform(X_tt)


X_tnh = np.apply_along_axis(fwht, axis=1, arr=X_tnS)
X_tth = np.apply_along_axis(fwht, axis=1, arr=X_ttS)
X_tnh = X_tnh.astype('float32')
X_tth = X_tth.astype('float32')

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


display(X_tnT.shape)
display(y_tnT.shape)
display(X_ttT.shape)
display(y_ttT.shape)


# In[12]:


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
model.fit(X_tnT, y_tnT, epochs=epochs, batch_size=batch_size, verbose=verbose,  validation_split=0.2, callbacks = stopper)

#Evaluate Model
_, HTaccuracy = model.evaluate(X_ttT, y_ttT, batch_size=batch_size, verbose=verbose)
display(HTaccuracy)


# In[13]:


#Make Prediction
HTypred = model.predict(X_ttT)
HTypred = tf.argmax(input = HTypred, axis = 1).numpy()
HTypred = encoder.inverse_transform(HTypred)


# In[14]:


from sklearn.metrics import confusion_matrix
display(SCypred)
CMSC = confusion_matrix(ytruth, SCypred, labels = labels)
CMFT = confusion_matrix(ytruth, FTypred, labels = labels)
CMHT = confusion_matrix(ytruth, HTypred, labels = labels)
display(SCaccuracy, FTaccuracy, HTaccuracy)


# In[15]:


fig, axs = plt.subplots(1, 3, sharey = True, figsize = [15, 5])

plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

pcm = axs[0].pcolor(CMSC, edgecolors = 'k', cmap = 'gist_heat_r')
plt.gca().invert_yaxis()
axs[0].set_xticks(ticks = np.linspace(0.5, len(labels)-0.5, num = len(labels)), labels = labels)
axs[0].set_yticks(ticks = np.linspace(0.5, len(labels)-0.5, num = len(labels)), labels = labels)
axs[0].set_ylabel("Actual Condition")
axs[0].set_xlabel("Predicted Condition")
axs[0].xaxis.set_label_position('top') 
axs[0].set_title('Raw Data');

axs[1].pcolor(CMFT, edgecolors = 'k', cmap = 'gist_heat_r');
plt.gca().invert_yaxis()
axs[1].set_xticks(ticks = np.linspace(0.5, len(labels)-0.5, num = len(labels)), labels = labels)
axs[1].set_title('Fourier Transform');
axs[1].set_xlabel("Predicted Condition")
axs[1].xaxis.set_label_position('top')

axs[2].pcolor(CMHT, edgecolors = 'k', cmap = 'gist_heat_r')
plt.gca().invert_yaxis()
axs[2].set_xticks(ticks = np.linspace(0.5, len(labels)-0.5, num = len(labels)), labels = labels);
axs[2].set_title('Walsh Hadamard Transform');
axs[2].set_xlabel("Predicted Condition")
axs[2].xaxis.set_label_position('top')

fig.colorbar(pcm, ax = axs[:], location = 'bottom', label = 'Number of Assignments');


# In[ ]:




