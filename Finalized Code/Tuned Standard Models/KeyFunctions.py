#!/usr/bin/env python
# coding: utf-8

# In[1]:


def GetSpectra(Source, Chemical, Concentration, ProfileNumber):
    import pandas as pd
    filepath = "Raman Data MITACs/" + Source + "/" + Chemical + "/"+ Concentration +    "/" + Concentration + "(" + ProfileNumber + ").txt"
    df = pd.read_csv(filepath, delim_whitespace = True)
    return df


# In[2]:


def GetFFT(Series):
    import numpy as np
    import pandas as pd

    ft = np.fft.fft(Series)
    fdf = pd.DataFrame(Series)

    fdf["Fourier"] = ft
    fdf["Real"] = np.real(ft)
    fdf["Imaginary"] = np.imag(ft)
    fdf["Modulus"] = np.abs(ft)
    fdf["Argument"] = np.angle(ft)
    
    return fdf


# In[3]:


def GetFolderSpectra(Source, Chemical, concs):
    import glob
    import pandas as pd
    
    df = pd.DataFrame()

    for i in concs:
        l = [pd.read_csv(filename, delimiter = '\t', index_col = "#Wave") for filename             in glob.glob("Raman Data MITACs/"+Source+"/"+Chemical+"/" + i + "/*.txt")]
        temp = pd.concat(l, axis = 1)
        temp.drop('#Intensity', axis = 1, inplace = True)
        temp.rename({'Unnamed: 1': i}, axis = 1, inplace = True)
        df = pd.concat([df, temp.transpose()])
    return df


# In[4]:


def DetectPeaks(Spectra, prom, mx = None, plot = False):
    import pandas as pd
    import scipy.stats as stats
    import matplotlib.pyplot as plt
    import scipy.signal as sgn
    import numpy as np

    PeakPos = np.zeros(np.shape(Spectra)).astype(int)
    Peaks, vals = sgn.find_peaks(Spectra, prominence = prom, width = [None, 100], height = [None, mx])
    PeakPos[Peaks] = 1
    
    if plot:
        plt.figure(figsize = [10, 6])
        plt.plot(Spectra)
        plt.scatter(Peaks, Spectra[Peaks], color = 'k')
    return PeakPos


# In[5]:


def ConstructCombinedDataset(dropmode = 0):
    import pandas as pd
    import numpy as np
    from scipy import stats as stats
    from sympy.discrete.transforms import fwht, ifwht
    from scipy import signal

    #Get Evaporating Ouzo Data
    concs = ['10-5', '10-9', '10-11', '10-14', '10-16']
    dfO = GetFolderSpectra("CHIRANJEEVI", "r6g(Evap Ouzo)", concs)
    col_names = np.array(dfO.columns)
    
    #Get and Process Ag Rings Data
    Source = "CHIRANJEEVI"
    Chemical = "r6g(Ag Nano Rings)"

    dfA1 = pd.DataFrame()
    for j in [6,9]:
        Concentration = "10-" + str(j)
        for i in [1, 2, 3, 4, 5]:
            temp = GetSpectra(Source, Chemical, Concentration, str(i))
            temp.set_index("#Wave", inplace = True)
            targ_len = len(temp)
            temp = temp.transpose()
            temp = temp.rename({'#Intensity':Concentration}, axis = 0)
            temp.set_axis(np.array(range(targ_len)), axis =1, inplace = True)
            dfA1 = pd.concat([dfA1, temp], axis = 0)  

    for j in [5,7,8]:
        Concentration = "10-" + str(j)
        for i in [1, 2, 3, 4, 5]:
            temp = GetSpectra(Source, Chemical, Concentration, str(i))
            temp.set_index("#Wave", inplace = True)
            temp = np.flip(signal.resample(temp, targ_len))
            temp = np.reshape(temp, [targ_len])
            temp = pd.DataFrame(temp)
            temp = temp.transpose()
            temp = temp.rename({0:Concentration}, axis = 0)
            temp.set_axis(np.array(range(targ_len)), axis =1, inplace = True)
            dfA1 = pd.concat([dfA1, temp], axis = 0) 

    dfA1.set_axis(col_names, axis =1, inplace = True)

    #Get and Process Ag Rings Data Second Set
    Chemical = "r6g(Ag Nano Rings 2)"

    dfA2 = pd.DataFrame()
    for j in [6,9]:
        Concentration = "10-" + str(j)
        for i in [1, 2, 3, 4, 5]:
            temp = GetSpectra(Source, Chemical, Concentration, str(i))
            temp.set_index("#Wave", inplace = True)
            targ_len = len(temp)
            temp = temp.transpose()
            temp = temp.rename({'#Intensity':Concentration}, axis = 0)
            temp.set_axis(np.array(range(targ_len)), axis =1, inplace = True)
            dfA2 = pd.concat([dfA2, temp], axis = 0)  

    for j in [5,7,8]:
        Concentration = "10-" + str(j)
        for i in [1, 2, 3, 4, 5]:
            temp = GetSpectra(Source, Chemical, Concentration, str(i))
            temp.set_index("#Wave", inplace = True)
            temp = signal.resample(temp, targ_len)
            temp = np.reshape(temp, [targ_len])
            temp = pd.DataFrame(temp)
            temp = temp.transpose()
            temp = temp.rename({0:Concentration}, axis = 0)
            temp.set_axis(np.array(range(targ_len)), axis =1, inplace = True)
            dfA2 = pd.concat([dfA2, temp], axis = 0) 

    dfA2.set_axis(col_names, axis =1, inplace = True)

    #Combine Datasets
    if dropmode == 0:
        df = pd.concat([dfO, dfA1, dfA2], axis = 0)
        df = df.fillna(0)
    elif dropmode == 1:
        df = pd.concat([dfA1], axis = 0)
        df = df.fillna(0)
    elif dropmode == 2:
        df = pd.concat([dfA2], axis = 0)
        df = df.fillna(0)
    elif dropmode == 3:
        df = pd.concat([dfO], axis = 0)
        df = df.fillna(0)
    elif dropmode == 4:
        df = pd.concat([dfA1, dfA2], axis = 0)
        df = df.fillna(0)
        
    labels = df.index.unique().values.tolist()
    labels.sort(key = lambda x: int(x.split('-', 1)[-1]))
    
    return df, labels


# In[6]:


def ConstructCombinedTriclosanDataset(dropmode = 0):
    import pandas as pd
    import numpy as np
    from scipy import stats as stats
    from sympy.discrete.transforms import fwht, ifwht
    from scipy import signal
    
    #Get First Set Triclosan Data 
    Source = "TULSI"
    Chemical = "triclosan"
    concs = ['10-5', '10-7', '10-8', '10-9']
    dfT1 = GetFolderSpectra(Source, Chemical, concs)
    dfT1 = dfT1.fillna(0)
    dfT1 = dfT1.iloc[:, 120:]
    col_names = np.array(dfT1.columns)

    #Get and Process Ag Rings Data
    Source = "CHIRANJEEVI"
    Chemical = "Triclosan"
    concs = ['10-3', '10-4', '10-5', '5x10-4', '5x10-5']
    dfT2 = pd.DataFrame()
    for j in [0, 1, 2, 3, 4]:
        Concentration = concs[j]
        for i in [1, 2, 3, 4, 5]:
            temp = GetSpectra(Source, Chemical, Concentration, str(i))
            temp.set_index("#Wave", inplace = True)
            targ_len = len(temp)
            temp = temp.transpose()
            temp = temp.rename({'#Intensity':Concentration}, axis = 0)
            temp.set_axis(np.array(range(targ_len)), axis =1, inplace = True)
            temp = temp.iloc[:, :-120]
            dfT2 = pd.concat([dfT2, temp], axis = 0)  

    dfT2.set_axis(col_names, axis =1, inplace = True)
    
    #Combine Datasets
    if dropmode == 0:
        df = pd.concat([dfT1, dfT2], axis = 0)
        df = df.fillna(0)
        df = df.rename({'5x10-4':'10-3','5x10-5':'10-4'}, axis = 0)
        labels = df.index.unique().values.tolist()
        labels.sort(key = lambda x: int(x.split('-', 1)[-1]))
    elif dropmode == 1:
        df = pd.concat([dfT1], axis = 0)
        df = df.fillna(0)
        labels = df.index.unique().values.tolist()
        labels.sort(key = lambda x: int(x.split('-', 1)[-1]))
    elif dropmode == 2:
        df = pd.concat([dfT2], axis = 0)
        df = df.fillna(0)
        labels = df.index.unique().values.tolist()
        labels.sort(key = lambda x: int(x.split('-', 1)[-1]))
    
    
    return df, labels


# In[7]:


def ConstructCombinedChlorDataset(dropmode = 0):
    import pandas as pd
    import numpy as np
    from scipy import stats as stats
    from sympy.discrete.transforms import fwht, ifwht
    from scipy import signal
    
    #Get Chlorpyrifos Data
    Source = "CHIRANJEEVI"
    Chemical = "chlorpyrifos(Ag Nano Rings)"
    
    concs = ['10-3', '10-4', '10-5', '10-6', '10-7']
    dfC1 = GetFolderSpectra(Source, Chemical, concs)
    dfC1 = dfC1.iloc[:, 73:-1]
    col_names = np.array(dfC1.columns)
    targ_len = np.shape(dfC1)[1]
    
    Chemical = "chlorpyrifos(Ag Nano Rings 2)"
    dfC2 = pd.DataFrame()

    for j in [0, 1, 2, 3, 4]:
        Concentration = concs[j]
        for i in [1, 2, 3, 4, 5]:
            temp = GetSpectra(Source, Chemical, Concentration, str(i))
            temp.set_index("#Wave", inplace = True)
            temp = signal.resample(temp, targ_len)
            temp = np.reshape(temp, [targ_len])
            temp = pd.DataFrame(temp)
            temp = temp.transpose()
            temp = temp.rename({0:Concentration}, axis = 0)
            temp.set_axis(np.array(range(targ_len)), axis =1, inplace = True)
            dfC2 = pd.concat([dfC2, temp], axis = 0) 
    dfC2.set_axis(col_names, axis =1, inplace = True)
        
     #Combine Datasets
    if dropmode == 0:
        df = pd.concat([dfC1, dfC2], axis = 0)
        df = df.fillna(0)
        labels = df.index.unique().values.tolist()
        labels.sort(key = lambda x: int(x.split('-', 1)[-1]))
    elif dropmode == 1:
        df = pd.concat([dfC1], axis = 0)
        df = df.fillna(0)
        labels = df.index.unique().values.tolist()
        labels.sort(key = lambda x: int(x.split('-', 1)[-1]))
    elif dropmode == 2:
        df = pd.concat([dfC2], axis = 0)
        df = df.fillna(0)
        labels = df.index.unique().values.tolist()
        labels.sort(key = lambda x: int(x.split('-', 1)[-1]))
    
    return df, labels


# In[8]:


ConstructCombinedChlorDataset(dropmode = 0)


# In[9]:


def PeakDistribution(df, plot = True):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sb
    
    PeakDist = pd.Series(0, index = np.arange(np.shape(df)[1]), name = 'Number of Peaks')
    for ind, row in df.iterrows():
            mx = max(row)
            
            PeakDist = PeakDist + DetectPeaks(row, prom = 0.2*mx)
    
    if plot:
        plt.figure(figsize = [8, 5])
        ax = sb.barplot(x = PeakDist.index, y = PeakDist.values)
        ax.set_xlabel('Wavelength')
        ax.set_ylabel('Number of Peaks')
        ax.set_xticks(ticks = ax.get_xticks()[::50], labels = df.columns[::50].astype(int))
        plt.xticks(rotation = 25, rotation_mode = 'anchor', fontsize = 'medium');
    
    return PeakDist


# In[10]:


def CreateOffset(Xvec, X_tn):
    import random as rng
    import numpy as np
    
    dev = 0.1*(X_tn.min(axis = 1).std())
    NewXvec = Xvec + rng.uniform(0, dev)
    return NewXvec

def PeakStretch(Xvec, X_tn, prom):
    import random as rng
    import numpy as np
    
    NewXvec = np.zeros(np.shape(Xvec))
    peakpos = DetectPeaks(Xvec, prom = prom)
    stretch = rng.uniform(-0.25, 0.25)
    for i in range(len(Xvec)):
        if peakpos[i]:
            NewXvec[i] = Xvec[i] * (1 + stretch)
        else:
            NewXvec[i] = Xvec[i]
        
    return NewXvec

def PeakFlip(Xvec, X_tn, prom, mx):
    import random as rng
    import numpy as np
    
    NewXvec = Xvec
    
    peakpos = DetectPeaks(Xvec, prom = prom, mx = mx)
    stretch = rng.lognormvariate(0, 2)
    numpeaks = rng.randint(0, 5)
    options = np.where(peakpos)[0]
    if len(options) < numpeaks:
        return NewXvec
    
    selection = np.random.choice(options, (numpeaks, 1))
    
    for i in selection:
        NewXvec[i] = abs(NewXvec[i]) * (1 + stretch)
        
    return NewXvec


# In[11]:


def AugmentData(X_tn, y_tn, size, labels, plot = False, peakflip = False):
    import random as rng
    import pandas as pd
    import time as time
    import numpy as np
    import seaborn as sb
    import matplotlib.pyplot as plt

    start = time.time()
    X_tn_aug = np.zeros((size-np.shape(X_tn)[0], np.shape(X_tn)[1]))
    y_tn_aug = np.empty((size-np.shape(X_tn)[0]), object)
    
    for i in range(len(X_tn_aug)):
        RndInd = rng.randint(0, np.shape(X_tn)[0]-1)
        currspec = X_tn[RndInd]
        
        #Detect if baseline correction is present
        if currspec[0] > 500:
            currspec = CreateOffset(currspec, X_tn)
        
        mx = max(currspec)-min(currspec)
        X_tn_aug[i, :] = PeakStretch(currspec, X_tn, 0.2*mx)
        if peakflip:
            X_tn_aug[i, :] = PeakFlip(PeakStretch(currspec, X_tn, 0.2*mx), X_tn, prom = [0.05*mx, 0.18*mx], mx = 0.2*mx)
        y_tn_aug[i] = y_tn[RndInd]

    X_tn_aug = np.concatenate((X_tn_aug, X_tn), axis = 0)
    y_tn_aug = np.concatenate((y_tn_aug, y_tn), axis = 0)
    
    if plot:
        
        Dist1 = PeakDistribution(pd.DataFrame(X_tn), False)
        Dist1 = (Dist1-Dist1.min())/(Dist1.max()-Dist1.min())
        Dist2 = PeakDistribution(pd.DataFrame(X_tn_aug), False)
        Dist2 = (Dist2-Dist2.min())/(Dist2.max()-Dist2.min())
        
        fig, ax = plt.subplots(1, 2, sharex = True, sharey = True, figsize = [14, 6])
        
        sb.barplot(x = Dist1.index, y = Dist1.values, ax = ax[0])
        ax[0].set_title('Original Peak Distribution')
        ax[0].set_xlabel('Wavelength (γ)')
        ax[0].set_ylabel('Relative Number of Peaks')
        ax[0].set_xticks(ticks = ax[0].get_xticks()[::50], labels = labels[::50],                         rotation = 25, rotation_mode = 'anchor', fontsize = 'medium')
        
        sb.barplot(x = Dist2.index, y = Dist2.values, ax = ax[1])
        ax[1].set_title('Augmented Data Peak Distribution')
        ax[1].set_xlabel('Wavelength (γ)')
        ax[1].set_ylabel('Relative Number of Peaks')
        ax[1].set_xticks(ticks = ax[1].get_xticks()[::50], labels = labels[::50],                         rotation = 25, rotation_mode = 'anchor', fontsize = 'medium')

    return X_tn_aug, y_tn_aug


# In[12]:


def Scorer(ytrue, ypred, labels):
    from sklearn.preprocessing import LabelEncoder
    labels = [int(i.split('-', 1)[1]) for i in labels]
    ytrue = [int(i.split('-', 1)[1]) for i in ytrue]
    ypred = [int(i.split('-', 1)[1]) for i in ypred]
    
    Encode = LabelEncoder()
    labels = Encode.fit_transform(y = labels)
    ytrue = Encode.transform(ytrue)
    ypred = Encode.transform(ypred)
    
    
    dist = 0
    for i in range(len(ytrue)):
        if ytrue[i] != ypred[i]:
            dist += (ypred[i]-ytrue[i])**2/(len(labels)**2)
            
    return dist


# In[76]:


def BaselineCorrection(df):
    import pandas as pd
    import numpy as np
    import scipy.signal as sgn
    import scipy.integrate as aderiv
    import scipy as sp
    import matplotlib.pyplot as plt
    
    dfN = df.copy()
    dfN.reset_index(inplace = True)
    for ind, row in dfN.iterrows():
        con = row.iloc[0]
        row = row.iloc[1:]
        peaks, temp = sgn.find_peaks(-row, prominence = 0.1*max(-row), width = 1.5)

        xbase = np.insert(row.index[peaks].values, [0, -1], [row.index[0], row.index[-1]])
        ybase = np.insert(row.values[peaks], [0, -1], [row.values[0], row.values[-1]])



        bsfnc = sp.interpolate.interp1d(x = xbase, y = ybase)
        yinterp =  bsfnc(row.index.astype(float))
        
        dfN.iloc[ind, 1:] = (row-yinterp).values
    
    dfN.set_index(['index'], inplace = True, drop = True)
    return dfN


# In[ ]:




