#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 18:21:48 2017

@author: shugo
"""

import numpy as np
import matplotlib.pyplot as plt

def ode_func(u):
    # main function for 1st order ODE
    A = np.array([[0.,1.,0.,0.], [0.,0.,0.,0.], [0.,0.,0.,1.], [0.,0.,0.,0.]])
    b = np.array([0.,0.,0.,-9.81])
    
    dudt = np.dot(A,u.T) + b.T

    return dudt
    
    
def f_euler(u1,dt):
    
    u2 = u1 + dt*ode_func(u1)
    return u2
    
    
def f_RK4(u1,dt):
    
    k1 = ode_func(u1)
    k2 = ode_func(u1 + dt/2.*k1)
    k3 = ode_func(u1 + dt/2.*k2)
    k4 = ode_func(u1 + dt*k3)
    
    u2 = u1 + dt/6. * (k1 + 2.*k2 + 2.*k3 + k4)
    
    return u2
    
    
def run(dt,theta,v0,solver):
    
    # initialization
    dxdt0 = v0 * np.cos(theta)
    dydt0 = v0 * np.sin(theta)
    
    t = 0
    
    u = np.array([0.,dxdt0,0.,dydt0])
    t_max = 60
    
    while u[2] >= 0.:
        t = t+dt
        
        if solver == 'RK4':
            u = f_RK4(u,dt)
        else:
            u = f_euler(u,dt)

        if t>t_max:
            break
        # END
    # END WHILE
    
    print('landing point = ',u[0])
    print('at t = ',t)
    
    return u[0]
    
    
def main(dt,solver):
    """
    set parameters
    """
    theta = 45.*np.pi/180
    v0 = 10
    
    # solve numerically
    L_euler = run(dt,theta,v0,solver)
      
    # compare with numerical solution  
    L_analytical = 2.*v0**2./9.81 * np.cos(theta) * np.sin(theta)
    print('analytical solution',L_analytical )     
    
    error = abs(L_euler - L_analytical) / L_analytical
    print('error in log scale',np.log10(error))
    
    return error
        
def order_of_accuracy(solver='feuler'):
    
    dt = np.array([0.1,0.01,0.001,0.0001])
    errors = np.zeros(4)
    
    for i in range(4):
        print('i',i)
        print('dt',dt[i])
        # error for the size of time step
        errors[i] = main(dt[i],solver)
    #END IF
    
    # plot
    
    print ('solver=',solver)
    plt.figure()
    plt.loglog(dt,errors,lw=1.5)
    plt.title('time step vs error')
    plt.xlabel('dt [s]')
    plt.ylabel('error')
    plt.grid()
 
    
    
   
"""
run functions
"""
order_of_accuracy('RK4')



# main()