#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 17:33:18 2017

@author: shugo
"""

# sample driver script for trajectory simulation

import numpy as np
from UI_landingdist import TrajecSimu_UI

# define path and filename of a csv file
csv_filename = 'Parameters_csv/2018izu/relenza_ntk.csv'
# csv_filename = 'Parameters_csv/newmitei_parameters_C63.csv'
# csv_filename = 'Parameters_csv/2018noshiro/felix_2018noshiro.csv'
# csv_filename = 'Parameters_csv/2016izu/2016M_parameters.csv'
# csv_filename = 'Parameters_csv/reference/hyend_parameters.csv'
# csv_filename = 'Parameters_csv/2018noshiro/felix_2018noshiro_ver122517.csv'

# create an instance
mysim = TrajecSimu_UI(csv_filename)

# ------------------------------------
# run a single trajectory computation 
# ------------------------------------
mysim.run_single()

# ------------------------------------
# run a loop for landing point distribution
# ------------------------------------
# run_loop(n_winddirec, max_windspeed, windspeed_step)
# n_winddirec: number of wind directions 
# max_windspeed: max. wind speed [m/s]
# windspeed_step: wind speed step [m/s]
# mysim.run_loop(8, 7, 2)


# ------------------------------------
# run an optimization problem
# ------------------------------------

"""
m_dry = 30.
obj_type= 'Mach'
obj_value = 1.2
mysimu.run_rapid_design(m_dry, obj_type, obj_value)
"""