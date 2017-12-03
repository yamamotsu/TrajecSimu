#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 22:39:52 2017

@author: shugo
"""

import pandas as pd
import csv

# read csv file 
params = pd.read_csv('test.csv',comment='$') # '$' denote commet out


params_all = dict(params.as_matrix())
    

