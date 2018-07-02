#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 17:33:18 2017

@author: shugo
"""


# sample driver script for trajectory simulation

import numpy as np
from Scripts.interface import TrajecSimu_UI

# define path and filename of raa csv file
config_filename = 'Config_sample/sample_config.csv'
# config_filename = 'Parameters_csv/2018noshiro/felix_ver0525.csv'


# create an instance
mysim = TrajecSimu_UI(config_filename, 'noshiro_sea')

# ------------------------------------
# run a single trajectory computation 
# ------------------------------------
#mysim.run_single()

# ------------------------------------
# run a loop for landing point distribution
# ------------------------------------
# format: run_loop(n_winddirec, max_windspeed, windspeed_step)
#         n_winddirec: number of wind directions 
#         max_windspeed: max. wind speed [m/s]
#         windspeed_step: wind speed step [m/s]
mysim.run_loop(8, 7, 1)

# ------------------------------------
# run an optimization problem
# ------------------------------------

"""
m_dry = 30.
obj_type= 'Mach'
obj_value = 1.2
mysimu.run_rapid_design(m_dry, obj_type, obj_value)
"""