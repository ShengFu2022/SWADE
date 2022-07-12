

import pywt
import numpy as np; 
import pandas as pd

# This code is adapted from one public code of https://uqer.datayes.com/v3/community/share/57175736228e5b82757f53e2
#wavelet
# package one function
def c_deno_wave_(index_list,data,wavefunc,lv,m,n):   

#   m,n are levels to calculate the threshold processing. 
   
    # decompose
    coeff = pywt.wavedec(index_list,wavefunc,mode='sym',level=lv)  
#index_list  Input data
#wavefunc  Wavelet to use
#mode   Signal extension mode
#level  Decomposition level (must be >= 0).

# decompose based on level. Calculate by pywt package.  cAn is . cDn is wavelet parameter.
# cAn:scale coefficient; cDn:Wavelet coefficient

    sgn = lambda x: 1 if x > 0 else -1 if x < 0 else 0 # sgn function

    # decompose
    for i in range(m,n+1):   #   select the levels from m to n. Nothing to deal with scale  coefficient.
        cD = coeff[i]
        for j in range(len(cD)):
            Tr = np.sqrt(2*np.log(len(cD)))  # calculate the threshold 
            if cD[j] >= Tr:
                coeff[i][j] = sgn(cD[j]) - Tr  # shrink to the zero
            else:
                coeff[i][j] = 0   # lower than the threshold and set to 0

    # reconstruction
    denoised_index = pywt.waverec(coeff,wavefunc)

    # add the results to the dataframe in order to plot
    data['denoised_NDVI']=pd.Series('x',index=data.index)
    for i in range(len(data)):
        data.iloc[i,2] = denoised_index[i]
    
    return data



