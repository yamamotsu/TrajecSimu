#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 20:47:34 2017

@author: shugo
"""

import numpy as np
import matplotlib.pyplot as plt

# plot landing spot distribution

n_windspeed, n_winddir, _ = loc_bal.shape

# ----------------------------
# Izu umi zone: D = 5km
# ----------------------------
R = 2500.
center = np.array([1768., -1768.])
theta = np.linspace(0,2*np.pi,100)
x_circle = R * np.cos(theta) + 1768.
y_circle = R * np.sin(theta) - 1768.

# ----------------------------
#    ballistic
# ----------------------------
plt.figure()
# loop over wind speed
for i in range(n_windspeed):
    # plot ballistic landing distribution
    legend =  str(wind_speed_array[i]) + 'm/s'
    
    plt.plot(loc_bal[i,:,0] ,loc_bal[i,:,1] ,label=legend )
    plt.plot(loc_bal[i,0,0] ,loc_bal[i,0,1] , color='b', marker='x', markersize=4)  # wind: 0 deg
    plt.plot(loc_bal[i,1,0] ,loc_bal[i,1,1] , color='b', marker='x', markersize=4)  # wind: 0+ deg
#END IF


plt.grid()
plt.legend()
plt.title('ballistic landing distribution')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.plot(0,0,color='r',marker='*',markersize=12)
plt.plot(x_circle, y_circle,color='r',lw=5)
plt.show


# ----------------------------
#    parachute
# ----------------------------
plt.figure()
# loop over wind speed
for i in range(n_windspeed):
    # plot parachute landing distribution
    legend =  str(wind_speed_array[i]) + 'm/s'
    
    plt.plot(loc_para[i,:,0] ,loc_para[i,:,1] ,label=legend )
    plt.plot(loc_para[i,0,0],loc_para[i,0,1] , color='b', marker='x', markersize=4) # wind: 0 deg
    plt.plot(loc_para[i,1,0],loc_para[i,1,1] , color='b', marker='x', markersize=4) # wind: 0+ deg
#END IF

plt.grid()
plt.legend()
plt.title('parachute landing distribution')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.plot(0,0,color='r',marker='*',markersize=12)
plt.plot(x_circle, y_circle,color='r',lw=5)
plt.show
#plot landing location

