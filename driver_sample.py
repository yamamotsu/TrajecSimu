#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 17:33:18 2017

@author: shugo
"""

# sample driver script for trajectory simulation

import numpy as np
from Rocket_simu import Rocket_simu

# path and filename to a csv file
csv_filename = 'Parameters_csv/2016m_parameters.csv'

# create an instance
myrocket = Rocket_simu()

# ------------------------------------
# run a single trajectory computation 
# ------------------------------------
myrocket.run_single(csv_filename)

# ------------------------------------
# run a loop for landing point distribution
# ------------------------------------
"""
wind_direction_array = np.linspace(0.,360.,9)  # wind direction array
wind_speed_array = np.linspace(1.,7., 7)       # wind speed array
myrocket.run_loop(csv_filename, wind_direction_array, wind_speed_array)
"""







