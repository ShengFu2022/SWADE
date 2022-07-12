# %%
'''
function: insert the occurance time of landslide
          and Axel's none detrended time
          and Axel's detrended time 
          and LandTrendr time from excel
Input: the label
       
Output: two time: one is pre time, the other one is post time.
'''
#%%
from scipy import optimize
from scipy.optimize import curve_fit
#############################################################
import numpy, scipy, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings
#############################################################

from pywt import wavedec
import pywt
import csv
import matplotlib.pyplot as plt
# import time, datetime
from datetime import datetime
from numpy import *;  
import numpy as np; 
import pandas as pd
import xlrd, xlwt   
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
#%%
# label_number=10
def b_read_Act_Axel_Lr_(label_number):
# Reading the content in the xlsx file is to get the pre- and post- landslide date to construct a date range.
    Table_ori = []   #create an empty list to store rows
    my_local_file = os.path.dirname(os.path.abspath(__file__))
    table1 = pd.read_excel(my_local_file+'/ReadFile/03_Axel_Lr.xlsx')
    # book = xlrd.open_workbook((my_local_file+'/ReadFile/03_Axel_Lr.xlsx'))  #open the Excel spreadsheet as workbook
    # sheet = book.sheet_by_index(0)    #get the first sheet
    # for user in range(1, sheet.nrows):  #iterate 1 to maxrows
        # Table_ori.append(list(sheet.row_values(user, 0, sheet.ncols)))  #iterate through the sheet and get data from rows in list
    #define the function of a timestamp conversion 
    
    #extract the date stamp, 5 digit date
    # for user in range(1, sheet.nrows):  #iterate 1 to maxrows
        # Table_occurT.append(list(sheet.row_values(user, 0, 9)))  #iterate through the sheet and get data from rows in list
    #list to dataframe
    Table_occurT = pd.DataFrame(index=np.arange(len(table1.index)), columns=['ID', 'time1', 'time2','Axel1_time1','Axel1_time2','Axel2_time1','Axel2_time2','Lr_time','Lr_time2']) 
    
    for i in range(0,len(table1.index)):
        Table_occurT.ID[i] = i
        Table_occurT.time1[i] = str(table1['time1'][i])[0:10]
        Table_occurT.time2[i] = str(table1['time2'][i])[0:10]
        
        Table_occurT.Axel1_time1[i] = str(table1['Axel1_time1'][i])[0:10]
        Table_occurT.Axel1_time2[i] = str(table1['Axel1_time2'][i])[0:10]
        
        Table_occurT.Axel2_time1[i] = str(table1['Axel2_time1'][i])[0:10]
        Table_occurT.Axel2_time2[i] = str(table1['Axel2_time2'][i])[0:10]
        
        Table_occurT.Lr_time[i] = str(table1['Lr_time'][i])[0:10]
        Table_occurT.Lr_time2[i] = str(table1['Lr_time2'][i])[0:10]
    
    
    
    
    time1 = Table_occurT['time1'][label_number-1]
    time2 = Table_occurT['time2'][label_number-1]
    time3 = Table_occurT['Axel1_time1'][label_number-1]
    time4 = Table_occurT['Axel1_time2'][label_number-1]
    time5 = Table_occurT['Axel2_time1'][label_number-1]
    time6 = Table_occurT['Axel2_time2'][label_number-1]
    time7 = Table_occurT['Lr_time'][label_number-1]
    time8 = Table_occurT['Lr_time2'][label_number-1]
    time_two = []
    time_two = [time1, time2,time3,time4,time5,time6,time7,time8]
    for i in range(0,len(time_two)):
        if time_two[i] == 'NaT':
            time_two[i] = '1900-01-01'

    return time_two
