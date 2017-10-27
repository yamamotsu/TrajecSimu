#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 01:44:37 2017

@author: shugo
"""

import numpy as np
from Rocket_simu import Rocket


# define rocket parameters

# fin configuration
Lroot = 0.17   # root length of fin
Ltip = 0.096   # tip length of fin
h = 0.15     # height of fin
ang = 39.5     # back angle
alpha_fins = 0.0 * np.pi/180.
#alpha_fins = 0.

rocket_diameter = 0.17

dis = 3  # how many to discretize
y = np.linspace(h/dis/2. ,h-h/dis/2. ,dis)
xLE = 1/np.tan((90.-ang)*np.pi/180.) * y   # leading edge locations
fin_len = (Ltip*y + (h-y)*Lroot) / h    # length of fin
xFP = xLE + fin_len/4   # forcing points
r_arm = y + rocket_diameter/2. # arm length for rotation
dy_fin = h/dis  # y step


rocket_params = {
                 # geometric configuration
                 'rocket_height'   : 3.,                # height of rocket
                 'rocket_diameter' : rocket_diameter,                # diameter of rocket
                 'X_area'          : np.pi*rocket_diameter**2. /4., # cross-sectional area
                 
                 # mass/inertia properties
                 'm_dry'   : 36.,   # dry weight of rocket i.e. exclude fuel
                 'm_fuel'  : 14.,       # fule weight // NOTE: total weight at lift off = m_dry + m_fuel
                 'CG_dry'  : 1.767,    # CG of dried body (nose tip = 0)
                 'CG_fuel' : 1.6535,   # CG of fuel (assume CG_fuel = const.)
                 
                 'MOI_dry' : 3*np.array([0.1007,33.888,33.888]),  # dry MOI (moment of inertia) wrt CG_dry
                 'MOI_fuel': 3*np.array([0.01688,0.3788,0.3788]), # fuel MOI wrt CG_fuel
                 
                 # aerodynamic properties
                 'CP_body' : 0.55,   # CP location without fins (budy tube+nose) (nose tip = 0)
                 # 'CP_fins' : 2.444,  # CP location of fins (nose tip = 0)
                 'Cd0'     : 0.55,    # total 0 angle-of-attack drag coefficient
                 
                 # fin configuration
                 'alpha_fins' : alpha_fins,  # fin attachment angle
                 'fin_len'    : fin_len,     # length of fin
                 'xFP'        : xFP,         # forcing points
                 'r_arm'      : r_arm,       # arm length for rotational moment
                 'dy_fin'     : dy_fin,      # y step of fin discretization
                 'LE_fins'    : 2.7,       # leading edge location of fin
                 }

numerical_params = {
                    'dt' : 0.001,
                    # 'vode' or 'zvode' or 'lsoda' or 'dopri5' or 'dop853' 
                    # if 'lsoda' is chosen, "odeint" will be used 
                    'integ' : 'lsoda_odeint',
                    't_max'    : 60,    
                    }
                    
launch_params = {'elev_angle'     : 85.,     # angle of elevation [deg]
                 'azimuth'        : 0.,    # north=0, east=90, south=180, west=270 [deg]
                 'wind_direction' : 135.,    # azimuth where wind is blowing from
                 'rail_length'    : 4., 
                 }  

# engine parameters
engine_params = {'t_MECO' : 10.,   # Main Engine Cut Off time
                 'thrust' : 2000.,  # thrust (const)
                } 
                        
                        
# create instance
myrocket = Rocket()

# set rockett parameters
myrocket.set_parameters(rocket_params,'rocket')

# set numerical parameters
myrocket.set_parameters(numerical_params,'execute')

# set launch condition parameters
myrocket.set_parameters(launch_params,'launch')

# set launch condition parameters
myrocket.set_parameters(engine_params,'engine')
  
# run main computation
myrocket.run()

# post-process
myrocket.postprocess('all')



