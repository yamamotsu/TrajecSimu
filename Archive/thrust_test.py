#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 16:06:39 2017

@author: shugo
"""

import numpy as np
from scipy import fftpack, integrate
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
#  read thrust curve data
# ----------------------
# csv file name
# thrust_filename = 'Thrust_curve_csv/thrust_HyperTEK_K_per0_001s.csv'
thrust_filename = 'Thrust_curve_csv/C6-3.csv'

# thrust_filename = 'Thrust_curve_csvdaq_LC.csv'

# sampling period 
sample_dt = 0.001   # [s]

# read csv file into dataframe -> convert to numpy array
# thrust = np.array( pd.read_csv(thrust_filename), names=('T') )  
thrust = np.array( pd.read_csv(thrust_filename) ) 
# print('original len', len(thrust))

# calibration
# thrust = ( (thrust-0.1137) * 75.239 * 3.66) * 4.448 * 3/5

# maximum thrust
T_max = np.max(thrust)

# cut off info where thrust is less that 1% of T_max
thrust = thrust[ thrust >= 0.01*T_max ]
# overwrite time array
time = np.arange(0., len(thrust)* sample_dt, sample_dt)


# total impulse
I_total = integrate.trapz(thrust, time)

# averaged thrust
T_avg = I_total / time[-1]

print('--------------------')
print(' THRUST DATA ECHO')
print(' total impulse: ', I_total, '[N.s]')
print(' burn time: ', time[-1], '[s]')
print(' max. thrust: ', T_max, '[N]')
print(' average thrust: ', T_avg, '[N]')
print('--------------------')

plt.figure()
plt.plot(time, thrust)


#"""
# ------------------
# noise cancellation
# ------------------
# FFT (fast fourier transformation)
tf = fftpack.fft(thrust)
freq = fftpack.fftfreq(len(thrust), sample_dt)

# filtering 
fs = 3.                         # cut off frequency [Hz]
tf2 = np.copy(tf)
tf2[(freq > fs)] = 0

# inverse FFT
thrust = np.real(fftpack.ifft(tf2))

# plot filtered thrust curve
plt.plot(time, thrust, color='red')
plt.grid()


#"""



"""
# FFT (fast fourier transformation)
n = len(time)
tf = fftpack.fft(thrust)/(n/2)
freq = fftpack.fftfreq(n, sample_dt)

# filtering 
fs = 3.                         # cut off frequency [Hz]
tf2 = np.copy(tf)
tf2[(freq > fs)] = 0
tf2[(freq < 0)] = 0

# inverse FFT
thrust = np.real(fftpack.ifft(tf2)*n)

# plot filtered thrust curve
plt.plot(time, thrust, color='red')
"""
