#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 23:38:29 2017

@author: shugo
"""

"""
optimization tool for tanegashima rocket contest
"""

import numpy as np
import pandas as pd
from Rocket_simu import Rocket_simu
from scipy.optimize import minimize, differential_evolution
import time


"""
objective function definition
"""
def objective(x):
    # -----------------------------------
    #  objective function 
    #
    # input: x = [launcher_elev_angle[deg], para_diameter[m] ]
    # -----------------------------------
    
    # update state variable parameters in input dataframe
    #params_df.loc[params_df.parameter == 'elev_angle', 'value'] = x[0]
    #params_df.loc[params_df.parameter == 'S_para', 'value'] = np.pi * x[1]**2. / 4
    
    params_df.loc[params_df.parameter == 'elev_angle', 'value'] = x[0]
    params_df.loc[params_df.parameter == 'S_para', 'value'] = np.pi * x[1]**2. / 4
    
    # call main function: trajectory simulation
    myrocket.run(params_df)
    
    # get landing location
    x_loc, y_loc, flight_time = myrocket.postprocess('maxval')
    
    # distance of landing point from launching point
    dist = np.linalg.norm(np.array([x_loc, y_loc]))
    
    # objective function to be minimized
    obj_value = - (flight_time - dist + 100)
    
    """
    print('*************************************')
    print('*************************************')
    print('x,y,time,obj', x_loc, y_loc, flight_time, obj_value)
    print('*************************************')
    print('*************************************')
    """
    
    return obj_value
    
    
"""
run optimization problem
"""
# input rocket parameters
#
# params_df = pd.read_csv('test.csv',comment='$') # '$' denotes commet out
csv_filename = 'Parameters_csv/newmitei_parameters_C63.csv'
params_df = pd.read_csv(csv_filename, comment='$', names=('parameter', 'value') ) # '$' denotes commet out

# create instance
myrocket = Rocket_simu()

# state variable definition
# x = [launcher_elev_angle[deg], para_diameter[m]]

# set initial value
x0 = np.array([88., 0.3])
# x0 = 85.

# initial objective value
obj_inital = objective(x0)

# set variable boundary
# bnds = ((75, 90))
bnds = ((75, 90), (0.3, 0.6))

# set option
options={'eps': 1.e-03}
# ---------------------------
# solve optimization problem
# ---------------------------
start = time.time() 
# res = minimize(objective, x0, args=(), method='SLSQP', bounds=bnds, options={'eps': 1.e-03})
res = differential_evolution(objective, bnds)
elapsed_time = time.time() - start

# ---------------------------
# output result
# ---------------------------
print('initial elev_angle[deg], para_diameter[m]', x0)
print('initial score', obj_inital)

# output optimal x
print(res.message)
print('optimal elev_angle[deg], para_diameter[m]', res.x[0], res.x[1])
# optimzal objective value
print('optimal score', res.fun)
print('compuration time[s]:  ',elapsed_time)
    
    
    

