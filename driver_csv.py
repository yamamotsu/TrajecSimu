#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 01:05:10 2017

@author: shugo
"""

import numpy as np
import pandas as pd
from Rocket_simu import Rocket

# -------------------------
#  simulation purpose selection
# -------------------------
# 'detail': show detailed for a single trajectory
# 'loop'; loop over azimuth and wind speed
simu_type = 'loopp'

# -------------------------
# read csv file that contains parameters
# params_df = pd.read_csv('test.csv',comment='$') # '$' denotes commet out
params_df = pd.read_csv('felix_parameter.csv', comment='$', names=('parameter', 'value') ) # '$' denotes commet out

# get simulation type
#try:
#    simu_type = params_df[0,'simu_type']
    

# get integration type


# convert parameters into float
# params_df.value = params_df.value.convert_objects(convert_numeric=True)

# create instance
myrocket = Rocket()

if simu_type == 'loop':
    # -----------------------
    #   landing point prediction 
    # -----------------------
    # define wind angle array
    wind_angle_array = np.linspace(0.,360.,9)
    #wind_angle_array = np.array([90,45,0,315,270,225,180,135,90])
    
    # define wind speed array
    wind_speed_array = np.linspace(1.,9.,9)
    
    # array for landing location
    loc_bal = np.zeros((len(wind_speed_array), len(wind_angle_array), 2) )
    loc_para = np.zeros((len(wind_speed_array), len(wind_angle_array), 2) )
    
    # loop count initializatino
    i_speed = 0
    
    
    # loop over wind speed
    for wind_speed in wind_speed_array:
        # overwrite wind speed
        params_df.loc[params_df.parameter == 'wind_speed', 'value'] = wind_speed
        
        
        # loop over wind angle
        i_angle = 0
        for wind_angle in wind_angle_array[:-1]:
            # overwrite wind speed
            params_df.loc[params_df.parameter == 'wind_direction', 'value'] = wind_angle
            
            # -----------------------------------
            #  landing point for ballistic fall  
            # -----------------------------------
            # overwrite parachute opening delay time to inf.
            params_df.loc[params_df.parameter == 't_para_delay', 'value'] = 100000.
            # params_df.loc[params_df.parameter == 't_deploy', 'value'] = 100000.
            
            # set all parameters
            myrocket.set_parameters(params_df)
            # run main computation
            myrocket.run()
            # post-process
            loc_bal[i_speed,i_angle,:] = myrocket.postprocess('location')
            
            
            # ---------------------------------
            # landing point for parachute fall
            # ---------------------------------
            # overwrite parachute opening delay time to 1s.
            params_df.loc[params_df.parameter == 't_para_delay', 'value'] = 1.
            # params_df.loc[params_df.parameter == 't_deploy', 'value'] = 28.
            
            # set all parameters
            myrocket.set_parameters(params_df)
            # run main computation
            myrocket.run()
            # post-process
            loc_para[i_speed,i_angle,:] = myrocket.postprocess('location')
           
            # loop count
            i_angle += 1
        #END FOR
        
        # loop count
        i_speed += 1
    # END FOR
    
    # close wind direction loop
    loc_para[:,-1,:] = loc_para[:,0,:]
    loc_bal[:,-1,:] = loc_bal[:,0,:]
    
else:
    # -------------------------
    #  Single trajectory simulation for detail
    # -------------------------
    # set all parameters
    myrocket.set_parameters(params_df)
    
    # run main computation
    myrocket.run()
    
    # post-process
    myrocket.postprocess('all')
#END IF
    

    