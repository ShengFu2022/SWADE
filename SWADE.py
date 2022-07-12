# 20210704


#%%

# pip install xlwt
# pip install xlrd
# plot three lines
# 1 Forest_NDVI  minus  Forest_fit 
# 2 Landslide_NDVI  minus  Landslide_fit 
# 3 cum_sum((Forest_NDVI  minus  Forest_fit)-(Landslide_NDVI  minus  Landslide_fit ))

## additional lines: actual vertical + wavelet line + detrend line + non-detrend line +landtrendr (area plot)
## x axis-actual time. and label 


## combine Axel's method and SWADE, showing break point and recovery periods.

# 1985-2017 include winter months.

# line: actual landslide occurrence time
# points: Seg1  Seg2  Non-detrended  Detrended  LandTrendr
#%%  by Sheng


from c_deno_wave import c_deno_wave_
from b_read_Act_Axel_Lr import b_read_Act_Axel_Lr_
from d_step_wise import d_step_wise_

#############################################################
from mpl_toolkits.axisartist.parasite_axes import HostAxes, ParasiteAxes

import matplotlib.ticker as mticker
from matplotlib.dates import DateFormatter
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from typing import NamedTuple
from scipy.signal import find_peaks
from datetime import datetime
from datetime import date
#############################################################
from scipy import optimize
import numpy, scipy, matplotlib
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
#############################################################
import datetime
from pywt import wavedec
import pywt
import csv
import time, datetime
from datetime import datetime
from numpy import *;
import numpy as np; 
import pandas as pd
import xlrd, xlwt 
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import math
import os



#%%    gee_Landslide #add forest files
'''import data'''
"""
8 various column
    #'fitted' 'Date' 'NDVI' 'MM_NDVI' 'NDVI_diff' 'MMNDVI_diff' 'CumNDVI_diff' 'CumMMNDVI_diff'
    fitted:list
    Date:list
    NDVI:list
    MM_NDVI:list
    NDVI_diff:list
    MMNDVI_diff:list
    CumNDVI_diff:list
    CumMMNDVI_diff:list
"""
'''
5 various variable
    FOREST_fNDVI
    FOREST_fitted
    FOREST_combined
    FOREST_filtered
    FOREST_DIFF_AVG
'''
# gee_Forest: from 12 to the average.

# Forest_all=(Forest1,Forest2,Forest3,Forest4,Forest5,Forest6,Forest7,Forest8,Forest9,Forest10,Forest11,Forest12)
# number=len(Forest_all)
# Forest_all_label=[None] * number
# for i in range(number):
#     Forest_all_label[i]='Forest'+str(i+1)
    
# for i in range(number):
#     LS_Slide = Forest_all[i]
#     label = Forest_all_label[i]
#     train = a_gee_down_landslide_raw_fit_(LS_Slide,label);   #### too too too slow slow slow slow slow slow

# Forest_all=(LS1,LS2,LS3,LS4,LS5,LS6,LS7,LS8,LS9,LS10,
#             LS11,LS12,LS13,LS14,LS15,LS16,LS17,LS18,LS19,LS20,
#             LS21,LS22,LS23,LS24,LS25,LS26,LS27,LS28,LS29,LS30,LS31,
#             LS32,LS33,
#             LS34,LS35,LS36,LS37,LS38,LS39,LS40,
#             LS41,LS42,LS43,LS44,LS45,LS46,LS47,LS48,LS49,LS50,
#             LS51,LS52,LS53,LS54,LS55,LS56,LS57,LS58,LS59,LS60,
#             LS61,LS62,LS63,LS64,LS65,LS66);
# number=len(Forest_all)
# # number = number+31
# Forest_all_label=[None] * number
# for i in range(number):
#     Forest_all_label[i]='LS'+str(i+1)
    
# for i in range(number):
#     LS_Slide = Forest_all[i]
#     label = Forest_all_label[i]
#     train = a_gee_down_landslide_raw_fit_(LS_Slide,label);   #### too too too slow slow slow slow slow slow


#%%
# LS_Slide = Forest1
# label = 'Forest1'
# train = pd.read_csv(my_local_file+'/LS2/'+label+'_backup.csv')


number = 12
Forest_all_label=[None] * number
for i in range(number):
    Forest_all_label[i]='Forest'+str(i+1)

my_local_file = os.path.dirname(os.path.abspath(__file__))

train = pd.read_csv(my_local_file+'/01_data/'+'Forest1'+'_backup.csv')



##########delete Janurary, Feburary, March,    October, November, December
train['month']=train['Date'].apply(lambda x: int(x[5:7]))
# train = train[train.month.isin(range(4, 10))]# 10 excluded   4, 10
train=train.reset_index(drop=True)
##########delete Janurary, Feburary, March,    October, November, December 
##########delete 1984 
train['year']=train['Date'].apply(lambda x: int(x[0:4]))
train = train[train.year.isin(range(1985, 2018))]# 10 excluded
train=train.reset_index(drop=True)
##########delete 1984 
for i in range(1,number):
    label = Forest_all_label[i]
    train1 = pd.read_csv(my_local_file+'/01_data/'+label+'_backup.csv')
    train1['month']=train1['Date'].apply(lambda x: int(x[5:7]))
    train1=train1.reset_index(drop=True)
    train1['year']=train1['Date'].apply(lambda x: int(x[0:4]))
    train1 = train1[train1.year.isin(range(1985, 2018))]
    train1 = train1.reset_index(drop=True)
    train['fitted'] = (train['fitted']+train1['fitted'])
    train['NDVI'] = (train['NDVI']+train1['NDVI'])
train['fitted'] = train['fitted']/number
train['NDVI'] = train['NDVI']/number
column_names = ['DD','fitted','Date_','Date','NDVI','MM_NDVI','NDVI_diff','MMNDVI_diff','CumNDVI_diff','CumMMNDVI_diff']
FOREST_combined = pd.DataFrame(columns = column_names) 
FOREST_combined['DD'] = train['DD']
FOREST_combined['Date_'] = train['date_struct']
FOREST_combined['fitted'] = train['fitted']
FOREST_combined['Date'] = train['Date']
FOREST_combined['NDVI'] = train['NDVI']

del Forest_all_label
del train
del train1
#####################
#%%
# decrease DD by excluding extra time. Jan\ Feb\ Mar.
FOREST_com = FOREST_combined.copy()
for dd_i in range(FOREST_com['DD'].size):
    FOREST_com['Date'][dd_i] = int(FOREST_com['Date'][dd_i][0:4]) 

FOREST_com['DD'][0]=0
YEAR = 1985
for dd_i in range(1,FOREST_com['DD'].size):
    if FOREST_com['Date'][dd_i] == YEAR:
        FOREST_com['DD'][dd_i] = FOREST_com['DD'][dd_i-1]+(pd.to_datetime(FOREST_com['Date_'][dd_i])-pd.to_datetime(FOREST_com['Date_'][dd_i-1])).days
    else:
        YEAR=YEAR+1
        FOREST_com['DD'][dd_i] = FOREST_com['DD'][dd_i-1]+(pd.to_datetime(FOREST_com['Date_'][dd_i])-pd.to_datetime(FOREST_com['Date_'][dd_i-1])).days# -  (date(YEAR, 4, 1 )-date(YEAR-1, 9, 30 )).days
        #datetime(YEAR, 4, 1, 00, 00, 00) -datetime(YEAR-1, 10, 1, 00, 00, 00)

FOREST_combined['DD'] = FOREST_com['DD'].tolist()
del FOREST_com
#%%
### add LS files

#8
#class Landslides_type(NamedTuple):
#    #'number' 'fitted' 'Date' 'NDVI' 'MM_NDVI' 'NDVI_diff' 'MMNDVI_diff' 'CumNDVI_diff' 'CumMMNDVI_diff'
#    fitted:list
#    Date:list
#    NDVI:list
#    MM_NDVI:list
#    NDVI_diff:list
#    MMNDVI_diff:list
#    CumNDVI_diff:list
#    CumMMNDVI_diff:list
#
#
#6
#    LS_fNDVI
#    LS_fitted
#    LS_combined
#    LS_filtered
#    LS_DIFF
#    LS_DIFF_detrend

# LS_Slide = LS1
# label = 'LS1'
#%%



# LS_all=(LS1,LS2,LS3,LS4,LS5,LS6,LS7,LS8,LS9,LS10,
#             LS11,LS12,LS13,LS14,LS15,LS16,LS17,LS18,LS19,LS20,
#             LS21,LS22,LS23,LS24,LS25,LS26,LS27,LS28,LS29,LS30,LS31,
#             LS32,LS33,LS34,LS35,LS36,LS37,LS38,LS39,LS40,
#             LS41,LS42,LS43,LS44,LS45,LS46,LS47,LS48,LS49,LS50,
#             LS51,LS52,LS53,LS54,LS55,LS56,LS57,LS58,LS59,LS60,
#             LS61,LS62,LS63,LS64,LS65,LS66);
# number=len(LS_all)
number = 66
LS_all_label=[None] * number
for i in range(number):
    LS_all_label[i]='LS'+str(i+1) 
####################################################################################################
aaaa=ceil(10) 
LS_i = 28
LS_i = LS_i-1 
seg_i = 3 
# LS_Slide = LS_all[LS_i]
label = LS_all_label[LS_i]
plt.rcParams["font.family"] = "Times New Roman"
#
train = pd.read_csv(my_local_file+'/01_data/'+label+'_backup.csv')
##########delete Janurary, Feburary, March,    October, November, December
train['month']=train['Date'].apply(lambda x: int(x[5:7]))
# train = train[train.month.isin(range(4, 10))]# 10 excluded
train=train.reset_index(drop=True)
##########delete Janurary, Feburary, March,    October, November, December
##########delete 1984 
train['year']=train['Date'].apply(lambda x: int(x[0:4]))
train = train[train.year.isin(range(1985, 2018))]# 1984 , 2018 are excluded
train=train.reset_index(drop=True)
##########delete 1984 

LS_combined=pd.DataFrame(columns = column_names) 
LS_combined['DD'] = train['DD']
LS_combined['Date_'] = train['date_struct']
LS_combined['fitted'] = train['fitted']
LS_combined['Date'] = train['Date']
LS_combined['NDVI'] = train['NDVI']
del train

####################################################################################################
####################################################################################################
#%%
# decrease DD by excluding extra time. Jan\ Feb\ Mar.
LS_com = LS_combined.copy()
for dd_i in range(LS_com['DD'].size):
    LS_com['Date'][dd_i] = int(LS_com['Date'][dd_i][0:4]) 
LS_com['DD'][0]=0
YEAR = 1985
for dd_i in range(1,LS_com['DD'].size):
    if LS_com['Date'][dd_i] == YEAR:
        LS_com['DD'][dd_i] = LS_com['DD'][dd_i-1]+(pd.to_datetime(LS_com['Date_'][dd_i])-pd.to_datetime(LS_com['Date_'][dd_i-1])).days
    else:
        YEAR=YEAR+1
        LS_com['DD'][dd_i] = LS_com['DD'][dd_i-1]+(pd.to_datetime(LS_com['Date_'][dd_i])-pd.to_datetime(LS_com['Date_'][dd_i-1])).days# -  (date(YEAR, 4, 1 )-date(YEAR-1, 9, 30 )).days
        #datetime(YEAR, 4, 1, 00, 00, 00) -datetime(YEAR-1, 10, 1, 00, 00, 00)

LS_combined['DD'] = LS_com['DD'].tolist()
del LS_com
#%%    landslide inventory
Landslide_id=label[2:]
Landslide_id = int(Landslide_id)

occur_ = b_read_Act_Axel_Lr_(Landslide_id)

occur_t = (occur_.copy())[0:2]  #['1997-08-04', '1997-08-20']
# transfer the date string to 10 digit timestamp.The default datestampe is the style of 2017-10-01 13:37:04
Table_label = pd.DataFrame(columns = ['real','date_stamp'])
Table_label['real'] = FOREST_combined['Date']
for i in range(0,FOREST_combined.iloc[:,0].size):
    time_array = time.strptime(Table_label.iloc[i,0], '%Y-%m-%d') #Table_label:str 1986-06-13; time_array::struct_time:1986-06-13.
    time_stamp = time.mktime(time_array)# time_array: time struct:1986-06-13. time_stamp:11，10 digit 
    Table_label.iloc[i,1] =time_stamp# No.1 colume: 11，10 digit
#
occur_t[0] = time.strptime(occur_t[0], '%Y-%m-%d')# result is construct_time: 1984-06-21
occur_t[0] = time.mktime(occur_t[0]) #The original is 5 digit timestamp, but now result is 11，10 digit，lol
occur_t[1] = time.strptime(occur_t[1], '%Y-%m-%d')
occur_t[1] = time.mktime(occur_t[1]) #The original is 5 digit timestamp, but now result is 11，10 digit，lol
#
Label_x = Table_label 
Label_x1 = Label_x.drop(index=(Label_x.loc[(Label_x['date_stamp']>=occur_t[0])].index))
Label_x1 = Label_x1.iloc[:,0].size
Label_x2 = Label_x.drop(index=(Label_x.loc[(Label_x['date_stamp']>=occur_t[1])].index))
Label_x2 = Label_x2.iloc[:,0].size 
Label_=[]
Label_.append(Label_x1)
Label_.append(Label_x2)
del Table_label
del time_array
del time_stamp
del Label_x 

####################################################################################################
####################################################################################################
#%%    read Axel's none detrended result
Landslide_id=label[2:]
Landslide_id = int(Landslide_id) 
occur2_ = occur_#b_read_Axel_Lr_(Landslide_id) 

occur_t = occur2_.copy() 
# transfer the date string to 10 digit timestamp.The default datestampe is the style of 2017-10-01 13:37:04
Table_label = pd.DataFrame(columns = ['real','date_stamp'])
Table_label['real'] = FOREST_combined['Date']
for i in range(0,FOREST_combined.iloc[:,0].size):
    time_array = time.strptime(Table_label.iloc[i,0], '%Y-%m-%d') #Table_label:str 1986-06-13; time_array::struct_time:1986-06-13.
    time_stamp = time.mktime(time_array)# time_array: time struct:1986-06-13. time_stamp:11，10 digit 
    Table_label.iloc[i,1] =time_stamp# No.1 colume: 11，10 digit
#
Label_x = Table_label 
if int(occur_t[2][0:4])>1983:
    occur_t[2] = time.strptime(occur_t[2], '%Y-%m-%d')# result is construct_time: 1984-06-21
    occur_t[2] = time.mktime(occur_t[2]) #The original is 5 digit timestamp, but now result is 11，10 digit，lol
    Label_x1 = Label_x.drop(index=(Label_x.loc[(Label_x['date_stamp']>=occur_t[2])].index))
    Label_x1 = Label_x1.iloc[:,0].size
else:
    Label_x1=-9999


if int(occur_t[3][0:4])>1983:
    occur_t[3] = time.strptime(occur_t[3], '%Y-%m-%d')
    occur_t[3] = time.mktime(occur_t[3]) #The original is 5 digit timestamp, but now result is 11，10 digit，lol
    Label_x2 = Label_x.drop(index=(Label_x.loc[(Label_x['date_stamp']>=occur_t[3])].index))
    Label_x2 = Label_x2.iloc[:,0].size 
else:
    Label_x2=-9999


Label_.append(Label_x1)
Label_.append(Label_x2)
del Table_label
del time_array
del time_stamp
del Label_x 

####################################################################################################
####################################################################################################
#%%    read Axel's detrended result
# transfer the date string to 10 digit timestamp.The default datestampe is the style of 2017-10-01 13:37:04
Table_label = pd.DataFrame(columns = ['real','date_stamp'])
Table_label['real'] = FOREST_combined['Date'] 
for i in range(0,FOREST_combined.iloc[:,0].size):
    time_array = time.strptime(Table_label.iloc[i,0], '%Y-%m-%d') #Table_label:str 1986-06-13; time_array::struct_time:1986-06-13.
    time_stamp = time.mktime(time_array)# time_array: time struct:1986-06-13. time_stamp:11，10 digit 
    Table_label.iloc[i,1] =time_stamp# No.1 colume: 11，10 digit
#
Label_x = Table_label 
if int(occur_t[4][0:4])>1983:
    occur_t[4] = time.strptime(occur_t[4], '%Y-%m-%d')# result is construct_time: 1984-06-21
    occur_t[4] = time.mktime(occur_t[4]) #The original is 5 digit timestamp, but now result is 11，10 digit，lol
    Label_x1 = Label_x.drop(index=(Label_x.loc[(Label_x['date_stamp']>=occur_t[4])].index))
    Label_x1 = Label_x1.iloc[:,0].size
else:
    Label_x1=-9999


if int(occur_t[5][0:4])>1983:
    occur_t[5] = time.strptime(occur_t[5], '%Y-%m-%d')
    occur_t[5] = time.mktime(occur_t[5]) #The original is 5 digit timestamp, but now result is 11，10 digit，lol
    Label_x2 = Label_x.drop(index=(Label_x.loc[(Label_x['date_stamp']>=occur_t[5])].index))
    Label_x2 = Label_x2.iloc[:,0].size 
else:
    Label_x1=-9999


Label_.append(Label_x1)
Label_.append(Label_x2)
del Table_label
del time_array
del time_stamp
del Label_x 

#%% read LandTrendr result.

# transfer the date string to 10 digit timestamp.The default datestampe is the style of 2017-10-01 13:37:04
Table_label = pd.DataFrame(columns = ['real','date_stamp'])
Table_label['real'] = FOREST_combined['Date'] 
for i in range(0,FOREST_combined.iloc[:,0].size):
    time_array = time.strptime(Table_label.iloc[i,0], '%Y-%m-%d') #Table_label:str 1986-06-13; time_array::struct_time:1986-06-13.
    time_stamp = time.mktime(time_array)# time_array: time struct:1986-06-13. time_stamp:11，10位 
    Table_label.iloc[i,1] =time_stamp# No.1 colume: 11，10 digit
#
Label_x = Table_label 
if int(occur_t[6][0:4])>1983:
    occur_t[6] = time.strptime(occur_t[6], '%Y-%m-%d')# result is construct_time: 1984-06-21
    occur_t[6] = time.mktime(occur_t[6]) #The original is 5 digit timestamp, but now result is 11，10 digit，lol
    Label_x1 = Label_x.drop(index=(Label_x.loc[(Label_x['date_stamp']>=occur_t[6])].index))
    Label_x1 = Label_x1.iloc[:,0].size
else:
    Label_x1=-9999


if int(occur_t[7][0:4])>1983:
    occur_t[7] = time.strptime(occur_t[7], '%Y-%m-%d')
    occur_t[7] = time.mktime(occur_t[7]) #The original is 5 digit timestamp, but now result is 11，10 digit，lol
    Label_x2 = Label_x.drop(index=(Label_x.loc[(Label_x['date_stamp']>=occur_t[7])].index))
    Label_x2 = Label_x2.iloc[:,0].size-1
else:
    Label_x1=-9999
    
    
Label_.append(Label_x1)
Label_.append(Label_x2)


del Table_label
del time_array
del time_stamp
del Label_x 
####################################################################################################
####################################################################################################

#%%    movemean
N_ = FOREST_combined['Date'].size
window_size = 7
window_size_half = int(ceil(window_size/2))
window_size_h = int(window_size_half-1)
FOREST_mm=pd.DataFrame(index=FOREST_combined.index.tolist(), columns = ['raw','mm']) 
FOREST_mm['raw'] = FOREST_combined['NDVI'].copy() 

for i in range(N_): 
    if i>= window_size_h and i<=(N_-window_size_h) :
        FOREST_mm['mm'][i] = sum(FOREST_mm['raw'][i-window_size_h:i+window_size_h+1])/FOREST_mm['raw'][i-window_size_h:i+window_size_h+1].count()
    elif i< window_size_h:
        FOREST_mm['mm'][i] = sum(FOREST_mm['raw'][0:i+window_size_h+1])/FOREST_mm['raw'][0:i+window_size_h+1].count()# 0 :0+3
    else:
        FOREST_mm['mm'][i] = sum(FOREST_mm['raw'][i-window_size_h:N_])/FOREST_mm['raw'][i-window_size_h:N_].count()

### find the nan, and then movemean it.
Forest_nan = FOREST_combined[FOREST_combined['MM_NDVI'].isnull()].index.tolist()
for j in range(len(Forest_nan)):
    i = Forest_nan[j]
    if i>= window_size_h and i<=(N_-window_size_h) :
        FOREST_mm['mm'][i] = sum(FOREST_mm['mm'][i-window_size_h:i+window_size_h+1])/FOREST_mm['mm'][i-window_size_h:i+window_size_h+1].count()
    elif i< window_size_h:
        FOREST_mm['mm'][i] = sum(FOREST_mm['mm'][0:i+window_size_h+1])/FOREST_mm['mm'][0:i+window_size_h+1].count()# 0 :0+3
    else:
        FOREST_mm['mm'][i] = sum(FOREST_mm['mm'][i-window_size_h:N_])/FOREST_mm['mm'][i-window_size_h:N_].count()


FOREST_combined['MM_NDVI']=FOREST_mm['mm'].copy() 
del FOREST_mm


#%%    movemean
LS_mm=pd.DataFrame(index=FOREST_combined.index.tolist(), columns = ['raw','mm']) 
LS_mm['raw'] = LS_combined['NDVI'].copy() 

for i in range(N_): 
    if i>= window_size_h and i<=(N_-window_size_h) :
        LS_mm['mm'][i] = sum(LS_mm['raw'][i-window_size_h:i+window_size_h+1])/LS_mm['raw'][i-window_size_h:i+window_size_h+1].count()
    elif i< window_size_h:
        LS_mm['mm'][i] = sum(LS_mm['raw'][0:i+window_size_h+1])/LS_mm['raw'][0:i+window_size_h+1].count()# 0 :0+3
    else:
        LS_mm['mm'][i] = sum(LS_mm['raw'][i-window_size_h:N_])/LS_mm['raw'][i-window_size_h:N_].count()

### find the nan, and then movemean it.
Forest_nan = LS_combined[LS_combined['MM_NDVI'].isnull()].index.tolist()
for j in range(len(Forest_nan)):
    i = Forest_nan[j]
    if i>= window_size_h and i<=(N_-window_size_h) :
        LS_mm['mm'][i] = sum(LS_mm['mm'][i-window_size_h:i+window_size_h+1])/LS_mm['mm'][i-window_size_h:i+window_size_h+1].count()
    elif i< window_size_h:
        LS_mm['mm'][i] = sum(LS_mm['mm'][0:i+window_size_h+1])/LS_mm['mm'][0:i+window_size_h+1].count()# 0 :0+3
    else:
        LS_mm['mm'][i] = sum(LS_mm['mm'][i-window_size_h:N_])/LS_mm['mm'][i-window_size_h:N_].count()


LS_combined['MM_NDVI']=LS_mm['mm'].copy() 

del LS_mm

del i,j
del window_size
del window_size_half
# LS_combined['MM_NDVI']=LS_combined.NDVI.rolling(window=window_size,min_periods=2).mean()
# FOREST_combined['MM_NDVI']=FOREST_combined.NDVI.rolling(window=window_size,min_periods=2).mean()
#%%
###Calculate cumulative NDVI difference from harmonic  _ LANDSLIDES
#From NDVI
LS_combined['NDVI_diff'] = -LS_combined['NDVI'] + LS_combined['fitted']
LS_combined['NDVI_diff'] = LS_combined['NDVI_diff'].replace(np.nan, 0)
#From movmean NDVI
LS_combined['MMNDVI_diff'] = LS_combined['MM_NDVI']-LS_combined['fitted']
LS_combined['MMNDVI_diff'] = LS_combined['MMNDVI_diff'].replace(np.nan, 0)
#
# cumulative NDVI difference
s = LS_combined['NDVI_diff']
LS_combined['CumNDVI_diff']=s.cumsum()
#
# Movmean cumulative NDVI difference 
s = LS_combined['MMNDVI_diff'] 
LS_combined['CumMMNDVI_diff']=s.cumsum() 
###Calculate cumulative NDVI difference from harmonic  _ FORESTS 
#From NDVI
FOREST_combined['NDVI_diff'] = FOREST_combined['NDVI']-FOREST_combined['fitted']
FOREST_combined['NDVI_diff'] = FOREST_combined['NDVI_diff'].replace(np.nan, 0)
#From movmean NDVI
FOREST_combined['MMNDVI_diff'] = FOREST_combined['MM_NDVI']-FOREST_combined['fitted']
FOREST_combined['MMNDVI_diff'] = FOREST_combined['MMNDVI_diff'].replace(np.nan, 0)
# cumulative NDVI difference
s = FOREST_combined['NDVI_diff']
FOREST_combined['CumNDVI_diff']=s.cumsum()
# Movmean cumulative NDVI difference
s = FOREST_combined['MMNDVI_diff']
FOREST_combined['CumMMNDVI_diff']=s.cumsum()
#
###AVERAGE FOR CUM DIFFERENCE NDVI FROM FITTED HARMONIC for several forest
LS_DIFF_detrend = pd.DataFrame(columns = column_names) 
LS_DIFF_detrend['CumNDVI_diff'] = FOREST_combined['CumNDVI_diff'] - LS_combined['CumNDVI_diff']
#
LS_DIFF_detrend['CumMMNDVI_diff'] = FOREST_combined['CumMMNDVI_diff'] - LS_combined['CumMMNDVI_diff']
LS_DIFF_detrend=LS_DIFF_detrend.drop(columns=['fitted', 'Date','NDVI','MM_NDVI','NDVI_diff','MMNDVI_diff'])
LS_DIFF_detrend = LS_DIFF_detrend.dropna(axis=0,how='any')#delete the rows which includes any NaN in the table
#

# %%    denoising method for FOREST_combined['MM_NDVI']
#  call function of wt. Calculate NDVI
data = pd.DataFrame(columns=['tDate','rawNDVI']) 
data.iloc[:,1] = FOREST_combined['MM_NDVI'].tolist()
data.iloc[:,0] = range(FOREST_combined['Date'].size)
index_list = np.array(data['rawNDVI'])
c_deno_wave_(index_list,data,'db34',4,1,4)  # Wavelet object or name string(db4), decomposition level:4. Select wavelet level from 1 to 4.
data=data.set_index('tDate')
FOREST_combined['MM_NDVI'] = data['denoised_NDVI'].tolist() 

del data 

#%%    denoising method for LS_combined['MM_NDVI']
#  call function of wt. Calculate NDVI
data = pd.DataFrame(columns=['tDate','rawNDVI']) 
data.iloc[:,1] = LS_combined['MM_NDVI'].tolist()
data.iloc[:,0] = range(FOREST_combined['Date'].size)
index_list = np.array(data['rawNDVI'])
c_deno_wave_(index_list,data,'db34',4,1,4)  # Wavelet object or name string(db4), decomposition level:4. Select wavelet level from 1 to 4.
data=data.set_index('tDate')
LS_combined['MM_NDVI'] = data['denoised_NDVI'].tolist()
del data 

#%%    denoising method for (forest-landslide)
#Table_4   index is date--tDate, the 0 column is cdNDVI.
Table_4 = pd.DataFrame(columns = ['cdNDVI_raw','number','cdNDVI']) 
#Table_4['cdNDVI']=pd.Series(0,index=LS_DIFF_detrend.index)
Table_4['cdNDVI_raw'] = (FOREST_combined['MM_NDVI'] - LS_combined['MM_NDVI'] ).cumsum()
Table_4['number'] = range(len(Table_4['cdNDVI_raw']))


# Table_4 = Table_4.dropna(how='any')
# 调用函数wt From NDVI
data = pd.DataFrame(columns=['tDate','rawNDVI'])
data.iloc[:,1] = Table_4['cdNDVI_raw'].tolist()
data.iloc[:,0] = range(FOREST_combined['Date'].size)
index_list = np.array(data['rawNDVI'])
c_deno_wave_(index_list,data,'db34',4,1,4)  # Wavelet object or name string(db4), decomposition level:4. Select wavelet level from 1 to 4.
data=data.set_index('tDate')
Table_4['cdNDVI'] = data['denoised_NDVI'].tolist()
del data 
# Table_4['cdNDVI'] = Table_4['cdNDVI_raw']
#%%
X_label = pd.DataFrame(index=FOREST_combined['DD'].tolist(), columns = ['DD','ID','Date','R2','label'])
##index--DD
X_label['DD'] = LS_combined['DD'].tolist()
X_label['ID'] = range(0,LS_combined['DD'].size)

X_fDD = pd.DataFrame(index=range(0,LS_combined.iloc[:,0].size), columns = ['DD','ID','Date','R2','label'])
##index--ID
X_fDD['DD'] = LS_combined['DD'].tolist()
X_fDD['ID'] = range(0,LS_combined['DD'].size)
X_fDD['Date'] = FOREST_combined['Date'].tolist()


#%%    stepwise 
###step wise linear fitting

##find several steps through step_wise function###############################################
xData = np.arange(0.0,Table_4.shape[0],1.0) 
yData = numpy.array(Table_4['cdNDVI'].tolist()) 
xData_seg = pd.DataFrame(columns=['tDate','diff'])
xData_seg.iloc[:,0] = FOREST_combined['Date'].tolist() 
xData_seg['tDate'] = pd.to_datetime(xData_seg['tDate']) 


seg_i=2
seg_stop=8 
while seg_i<=(seg_stop-1):
    seg_i = seg_i+1 # 3 4 5 6 7 8
    [step_matrix,step_matrix_possible_compare]=d_step_wise_(xData,yData,xData_seg,seg_i)
    if (not step_matrix_possible_compare.empty) and step_matrix_possible_compare['score'].size>=2: 
        seg_i2=seg_i
        seg_i = seg_stop



step_matrix_best_id_head2 = step_matrix_possible_compare.sort_values('score',ascending = False).head(2).index.tolist()
# step_matrix_best_id = step_matrix_possible_compare
#

#%%
#Label of x axis 
#prepare 2 things: one is lable, one is ID. 
Label_xtick = pd.DataFrame(index=FOREST_combined.index, columns=['tick','DD','ID','Date','year','index'])
Label_xtick_1 = np.linspace(1990,2020,num=7) 
Label_xtick_1 = list(map(int, Label_xtick_1))##show these years
### from yyyy-mm get 1985-01,1990-04,1995-04,2000-04,2005-04,2010-04,2015-04
Label_xtick['Date'] = FOREST_combined['Date'] 
Label_xtick['year'] = Label_xtick['Date'].apply(lambda x: int(x[0:4])) 
Label_xtick['index'] = FOREST_combined.index.tolist() 

year_1_i = 0 

for i in range(FOREST_combined['Date'].size ):
    year_1 = Label_xtick_1[year_1_i] 
    if Label_xtick['year'][i] == year_1:
        Label_xtick['tick'][i]=Label_xtick['Date'][i][0:4]# + Label_xtick['Date'][i][4:7]
        Label_xtick['DD'][i]=FOREST_combined['DD'][i]
        Label_xtick['ID'][i]=Label_xtick['index'][i]
        year_1_i=year_1_i+1



#add first one and last one
# Label_xtick['tick'][0] = FOREST_combined['Date'][0][0:7] occur2_[0]
Label_xtick['tick'][0] = FOREST_combined['Date'][0][0:4]#+FOREST_combined['Date'][0][4:7]
Label_xtick['DD'][0] = 0
Label_xtick['ID'][0] = 0
# Label_xtick['tick'][FOREST_combined['Date'].size-1] = FOREST_combined['Date'][FOREST_combined['Date'].size-1][0:5] +FOREST_combined['Date'][FOREST_combined['Date'].size-1][6:7]
# Label_xtick['DD'][FOREST_combined['Date'].size-1] = FOREST_combined['DD'].size-1
# Label_xtick['ID'][FOREST_combined['Date'].size-1] = FOREST_combined['Date'].size-1

fig2= plt.figure(1) 
ax_cof= HostAxes(fig2, [0.16, 0.1, 0.7, 0.7]) #use[left, bottom, weight, height]to define axes，0 <= l,b,w,h <= 1#parasite addtional axes, share x
#customize x-axis
ax_cof.set_xticklabels([i for i in Label_xtick['tick']], rotation=0, fontsize=4)#fontproperties=myfont
#set x-axis and set x label is special x tick
# ax0.set_xticks([i for i in range(0,198,30)]) 
ax_cof.set_xticks([i for i in Label_xtick['DD']]) 
ax_temp= ParasiteAxes(ax_cof, sharex=ax_cof) 
# ax_load= ParasiteAxes(ax_cof, sharex=ax_cof)
ax_cp= ParasiteAxes(ax_cof, sharex=ax_cof) 

#append axes 
ax_cof.parasites.append(ax_temp)
# ax_cof.parasites.append(ax_load)
ax_cof.parasites.append(ax_cp)
# hfont = {'fontname':'Helvetica'}
#invisible right axis of ax_cof
ax_cof.axis['right'].set_visible(False)
ax_cof.axis['top'].set_visible(False)
ax_temp.axis['left'].set_visible(True)
ax_temp.axis['left'].major_ticklabels.set_visible(True)
ax_temp.axis['left'].label.set_visible(True)

#set label foraxis
ax_cof.set_ylabel('Forest NDVI')
ax_cof.set_xlabel('Date ('+'The '+label+' experiment)')
ax_temp.set_ylabel('Landslide NDVI')
# ax_load.set_ylabel('CDNDVI_h')
ax_cp.set_ylabel('Cumulative NDVI difference (CDNDVI_w)')
temp_axisline=ax_temp.get_grid_helper().new_fixed_axis
# load_axisline=ax_temp.get_grid_helper().new_fixed_axis
cp_axisline=ax_cp.get_grid_helper().new_fixed_axis
ax_temp.axis['left'] = temp_axisline(loc='left', axes=ax_temp, offset=(-40,0))
# ax_load.axis['right2'] = load_axisline(loc='right', axes=ax_load, offset=(0,0))
# ax_cp.axis['right3'] = cp_axisline(loc='right', axes=ax_cp, offset=(40,0))
ax_cp.axis['right3'] = cp_axisline(loc='right', axes=ax_cp, offset=(0,0))

fig2.add_axes(ax_cof)
################171,219,227   172, 167, 252
color_1 = (0/255,0/255,0/255)#'yellow'  ##NDVI_L //161/255,160/255,176/255
color_2 = (117/255,0/255,32/255)   ##NDVI_F 36/255,165/255,224/255 //76/255,76/255,100/255
color_3 = (11/255,66/255,156/255)#'green'       ##CDNDVI_w (21/255,76/255,121/255)


Table_4_x=LS_combined['DD'].tolist() 
#"NDVI_L"
ax0_y =numpy.array(LS_combined['MM_NDVI'].values.tolist())
curve_temp,= ax_temp.plot(Table_4_x, ax0_y,'.', label='_nolegend_', color=color_1,markersize=1,zorder=3)

#"NDVI_F"
ax0_y =numpy.array(FOREST_combined['MM_NDVI'].values.tolist())
curve_cof,= ax_cof.plot(Table_4_x, ax0_y,'.', label='_nolegend_', color=color_2,markersize=1,zorder=3)


Table_4_y = Table_4['cdNDVI'].values.tolist() 
curve_cp,= ax_cp.plot(Table_4_x, Table_4_y, label="_nolegend_", color=color_3,zorder=6)
# CDNDVI_w
###############################
#%% add 1+3  actual results and three results
###################################
ax_cof.set_ylim(0,0.8)
ax_temp.set_ylim(0,0.8)
# ax_load.set_ylim(0,4)
# ax_cp.set_ylim(0,50)
y_min, y_max = ax_cof.get_ylim()
y_heitht = y_max-y_min
#####
column_names = ['x', 'y','Date']
re_wave = pd.DataFrame(np.zeros((1, 3)),columns = column_names) 
re_Actual = pd.DataFrame(np.zeros((2, 3)),columns = column_names)# pre  post 
re_Axel_1 = pd.DataFrame(np.zeros((2, 3)),columns = column_names)
re_Axel_2 = pd.DataFrame(np.zeros((2, 3)),columns = column_names)
re_L = re_wave.copy()
####actual landslide occurrance time 
re_Actual['x'][0] = X_fDD['DD'][Label_[0]]
re_Actual['x'][1] = X_fDD['DD'][Label_[1]]
re_Actual['Date'][0] = occur2_[0]

x=re_Actual['x'][0] 
y=y_min+y_heitht*0.97
plt.plot([x], [y], marker='o', markersize=5, color='black',zorder=8, label= re_Actual['Date'][0])
plt.axvline(x=re_Actual['x'][0], color='black',    zorder=8)#, alpha=0.7
# ax_cof.text(x+100, y,re_Actual['Date'][0],color='black',     transform=ax_cof.transData,zorder=8,fontsize=8) # write date
#####wavelet date
## x axis
j=0
if len(step_matrix_best_id_head2) ==1:
    Result1 = FOREST_combined['Date'][step_matrix['x1'][step_matrix_best_id_head2[0]]-1]
    Result1post = FOREST_combined['Date'][step_matrix['x1'][step_matrix_best_id_head2[0]]]
elif len(step_matrix_best_id_head2) ==0:
    Result1=0
    Result1post=0
if len(step_matrix_best_id_head2) ==2:
    Result1 = FOREST_combined['Date'][step_matrix['x1'][step_matrix_best_id_head2[0]]-1]
    Result1post = FOREST_combined['Date'][step_matrix['x1'][step_matrix_best_id_head2[0]]]
    Result2 = FOREST_combined['Date'][step_matrix['x1'][step_matrix_best_id_head2[1]]-1]
    Result2post = FOREST_combined['Date'][step_matrix['x1'][step_matrix_best_id_head2[1]]]
else:
    Result2=0
    Result2post=0
marker=(5, 1)

while j<len(step_matrix_best_id_head2):
    step_matrix_best_id=step_matrix_best_id_head2[j]
    i = step_matrix_best_id
    x_1=step_matrix['x1'][i]-1
    x_2=step_matrix['x2'][i]-1
    t = step_matrix['x1'][i]-1
    re_wave['x'][0] = X_fDD['DD'][step_matrix['x1'][i]]   # re_wave['y'][0] = step_matrix['y1'][i]
    re_wave['x'][1] = X_fDD['DD'][step_matrix['x2'][i]] 
    re_wave['Date'][0] = FOREST_combined['Date'][t]
    x=re_wave['x'][0] 
    # y=y_min+y_heitht*0.8+0.08*j
    y=step_matrix['y1'][i]
    if j == 0:#(0/255,255/255,0/255,255/255) 
        plt.scatter([x], [y], marker='^', s=30, color=(102/255,178/255,255/255),transform=ax_cp.transData,zorder=10, label=re_wave['Date'][0])
        # ax_cof.text(x+100, y,re_wave['Date'][0],color='green',transform=ax_cp.transData,zorder=10,fontsize=8) # write date
    if j == 1:#(0/255,128/255,0/255,255/255) 
        plt.scatter([x], [y], marker='v', s=30, color=(0/255,102/255,204/255),transform=ax_cp.transData,zorder=10, label=re_wave['Date'][0])
        # ax_cof.text(x+100, y,re_wave['Date'][0],color='green',transform=ax_cp.transData,zorder=10,fontsize=8) # write date
    # plt.axvline(x=x, color='goldenrod',        zorder=6, label='W', alpha=0.8)
    # ax_cof.text(x+100, y,re_wave['Date'][0],color='green',           transform=ax_cp.transData,zorder=10,fontsize=8) # write date
    # width = re_wave['x'][1] - re_wave['x'][0]
    # ax_cof.add_patch(
    #      patches.Rectangle(
    #         (re_wave['x'][0], y_min),#point of origin
    #         width, #
    #         y_max-y_min,#height
    #         # edgecolor = 'purple',
    #         facecolor = 'firebrick',
    #         fill=True,
    #         zorder=6, 
    #         # label = 'Pre - Post', 
    #         alpha=0.3
    #      ) )
    j=j+1
    
#####none detrend date
if Label_[2]>=0:
    re_Axel_1['x'][0] = X_fDD['DD'][Label_[2]]
    re_Axel_1['x'][1] = X_fDD['DD'][Label_[3]]
    re_Axel_1['Date'][0] = occur2_[2]
    x=re_Axel_1['x'][0]
    y=y_min+y_heitht*0.91
    # plt.axvline(x=x, color='green',zorder=7, label='H')
    plt.scatter([x], [y], marker='<', s=40,    color=(1.0,0.7,0.4,1),zorder=10, label=re_Axel_1['Date'][0])
    # ax_cof.text(x+100, y,re_Axel_1['Date'][0], color=(1.0,0.7,0.4,1),transform=ax_cof.transData,zorder=10,fontsize=8) # write date


#####detrend date
if Label_[4]>=0:
    re_Axel_2['x'][0] = X_fDD['DD'][Label_[4]]
    re_Axel_2['x'][1] = X_fDD['DD'][Label_[5]]
    re_Axel_2['Date'][0] = occur2_[4]
    x=re_Axel_2['x'][0]
    y=y_min+y_heitht*0.85
    # plt.axvline(x=x, color='blue',zorder=7, label='dH')
    plt.scatter([x], [y], marker='>', s=40,    color=(0.8,0.4,0.0,1),zorder=10, label=re_Axel_2['Date'][0])
    # ax_cof.text(x+100, y,re_Axel_2['Date'][0], color=(0.8,0.4,0.0,1), transform=ax_cof.transData,zorder=10,fontsize=8) # write date


###LandTrendr
if Label_[6]>=0:
    re_L['x'][0] = X_fDD['DD'][Label_[6]]
    re_L['x'][1] = X_fDD['DD'][Label_[7]]
    re_L['Date'][0] = occur2_[6]
    x=re_L['x'][0] 
    y=y_min+y_heitht*0.79
    # plt.axvline(x=x, color='purple',   zorder=7, label='L')
    plt.scatter([x], [y], marker='d', s=40,    color=(0.4,0.4,0.0),zorder=10, label='Year '+re_L['Date'][0][0:4])
    # ax_cof.text(x+100, y,re_L['Date'][0][0:4], color=(0.4,0.4,0.0),    transform=ax_cof.transData,zorder=10,fontsize=8) # write date
    width=re_L['x'][1]-re_L['x'][0] 
    ax_cof.add_patch(
         patches.Rectangle(
            (re_L['x'][0], y_min),#point of origin
            width, #
            y_max-y_min,#height
            # edgecolor = 'purple',
            facecolor = (0.4,0.4,0.0),
            fill=True,
            zorder=6, 
            # label = 'Pre - Post', 
            alpha=0.3
         ) )
# ax_cof.legend(loc='lower right', bbox_to_anchor=(0.3, 0.1), prop={'size': 6})
if int (re_Actual.loc[0]['Date'][0:4] )<2000:
    ax_cof.legend(loc='lower right', prop={'size': 6})   #lower   upper   right  left

if int (re_Actual.loc[0]['Date'][0:4] )>=2000:
    ax_cof.legend(loc='upper left', prop={'size': 6})   #lower   upper   right  left

# #axis label, tick color
ax_cof.axis['left'].label.set_color(color_2)
ax_temp.axis['left'].label.set_color(color_1)
# ax_load.axis['right2'].label.set_color('green')
ax_cp.axis['right3'].label.set_color(color_3)

ax_cof.axis['left'].major_ticks.set_color(color_2)
ax_temp.axis['left'].major_ticks.set_color(color_1)
# ax_load.axis['right2'].major_ticks.set_color('green')
ax_cp.axis['right3'].major_ticks.set_color(color_3)

ax_cof.axis['left'].major_ticklabels.set_color(color_2)
ax_temp.axis['left'].major_ticklabels.set_color(color_1)
# ax_load.axis['right2'].major_ticklabels.set_color('green')
ax_cp.axis['right3'].major_ticklabels.set_color(color_3)

ax_cof.axis['left'].line.set_color(color_2)
ax_temp.axis['left'].line.set_color(color_1)
# ax_load.axis['right2'].line.set_color('green')
ax_cp.axis['right3'].line.set_color(color_3)

plt.show() 


fig2.savefig(my_local_file+'/LS2/'+ label+'.jpg',dpi=700,bbox_inches = 'tight')
# return Result1,Result2

