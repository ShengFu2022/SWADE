# the first version, just calculate the slope of different part of CDNDVI curve
# and compare the slope directly.

import numpy as np
import matplotlib.pylab as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy
import warnings
# This code is adapted from Markus Dutschke's answer (edited May 20, 2021 at 5:11) (https://stackoverflow.com/questions/29382903/how-to-apply-piecewise-linear-fit-in-python).
def d_step_wise_(xs,ys,date,n_seg):
# segmented linear regression parameters
   # n_seg = 3
    
    # np.random.seed(0)
    # # parameters for setup
    # n_data = 20
    
    # xs = np.linspace(-1, 1, 20)
    # ys = np.random.rand(n_data) * .3 + np.tanh(3*xs)
    ################
    dys = np.gradient(ys, xs)#calculate slope at each point
    # Fit regression model
    rgr = DecisionTreeRegressor(max_leaf_nodes=n_seg)
    rgr.fit(xs.reshape(-1, 1), dys.reshape(-1, 1))
    # Predict
    dys_dt = rgr.predict(xs.reshape(-1, 1)).flatten()

    ys_sl = np.ones(len(xs)) * np.nan
    # Plot the results
    indexes=np.unique(dys_dt, return_index = True)[1]
    dys_dt_sec = [dys_dt[index] for index in sorted(indexes)] 

    # N_rows=len(np.unique(dys_dt))
    # N_cols=13
    # data_zeros = numpy.zeros(shape=(N_rows,N_cols)) data_zeros.copy()
    step = pd.DataFrame(index = range(n_seg),columns=['orderNO','d1','d2','dc','x1','x2','y1','y2','y1p','y2p','dys_dt_sec'])
    #first step: first x, last x, first y, last y, first y prediction, last y prediction, average slope
    

    for size_i in range(n_seg):
        y = dys_dt_sec[size_i]
        msk = dys_dt == y
        lin_reg = LinearRegression()
        lin_reg.fit(xs[msk].reshape(-1, 1), ys[msk].reshape(-1, 1)) 
        ys_sl[msk] = lin_reg.predict(xs[msk].reshape(-1, 1)).flatten() 
        step.iloc[size_i]['orderNO'] = size_i
        step.iloc[size_i]['x1'] = xs[msk][0]
        step.iloc[size_i]['x2'] = xs[msk][-1]
        a = int(step['x1'][size_i])
        b = int(step['x2'][size_i])
        step.iloc[size_i]['d1'] = date['tDate'][a]
        step.iloc[size_i]['d2'] = date['tDate'][b]
        step.iloc[size_i]['dc'] = (step.iloc[size_i]['d2']-step.iloc[size_i]['d1']).days
        step.iloc[size_i]['y1'] = ys[msk][0]
        step.iloc[size_i]['y2'] = ys[msk][-1]
        step.iloc[size_i]['y1p'] = ys_sl[msk][0]
        step.iloc[size_i]['y2p'] = ys_sl[msk][-1]
        step.iloc[size_i]['dys_dt_sec'] = y
        
        



    step_matrix =  step.copy()   
    step_matrix_possible = pd.DataFrame(index= step_matrix.index,  columns = step_matrix.columns) 
    for i in range(n_seg):# i>=1 and((i==0 and step_matrix['dys_dt_sec'][i]>0) or 
        if (i>=1 and step_matrix.iloc[i]['dys_dt_sec']>step_matrix.iloc[i-1]['dys_dt_sec']) and step_matrix.iloc[i]['dys_dt_sec']>0 and step_matrix.iloc[i]['dc']>365: 
            step_matrix_possible.iloc[i,:] = step_matrix.iloc[i,:]
            
    step_matrix_possible = step_matrix_possible.dropna(how='all')
    #compare all the points 
    step_matrix_possible_num = len(step_matrix_possible.index)
    step_matrix_possible_compare = pd.DataFrame(0, index=step_matrix_possible.index, columns = ['score','score2','dd_con']) 
    for i in range(step_matrix_possible_num):
        j = step_matrix_possible.iloc[i]['orderNO']
        j = int(j)
        step_matrix_possible_compare.loc[j,'score'] = step_matrix['dys_dt_sec'][j]#/step_matrix['dc'][j]#-step_matrix['dys_dt_sec'][j-1] 

    
    return step_matrix,step_matrix_possible_compare
