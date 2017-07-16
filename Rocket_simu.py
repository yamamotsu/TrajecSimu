#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 22:23:48 2017

@author: shugo
"""

import numpy as np
from trajectory import simu_main
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# class
class Rocket:
    
    """
    ----------------------------------------------------
    ----     Methods for initilization            ------
    ----------------------------------------------------
    """
    
    def __init__(self):
        # =============================================
        # this method is called when instance is created
        # =============================================
        
        # initialize (set default) parameters
        self.execute   = self.get_default('execute')
        self.launch    = self.get_default('launch')
        self.engine    = self.get_default('engine')
        self.parachute = self.get_default('parachute')
        self.rocket    = {}        
        return
    
        
    def get_default(self,dict_type=None):
        # =============================================
        # returns default parameters
        # =============================================
        if dict_type == 'execute':
            # executive (numerical) parameters
            defaults = {'dt'       : 0.05,   # time step
                        't_max'    : 20*60,  # maximum time
                        'N_record' : 20,     # record history every 20 iteration
                        'integ' : 'dopri5',   # time integration scheme
                        }         
                        
        elif dict_type == 'launch':
            # launch condition parameters            
            defaults = {
                        # launcher configuration
                        'elev_angle'     : 85.,     # angle of elevation [deg]
                        'azimuth'        : 0.,      # north=0, east=90, south=180, west=270 [deg]
                        'rail_length'    : 5.,      # length of launcher rail
                        # air property
                        'T0'             : 288.,    # temperature [K] at 10m alt.
                        'p0'             : 1.013e5, # pressure [Pa] at 10km alt.
                        'wind_direction' : 0.,      # azimuth where wind is blowing from 
                        'wind_speed'     : 3.,      # wind speed at 10m alt. [m/s]
                        'Cwind'          : 0.143,   # wind coefficient
                        }  
                        
        elif dict_type == 'engine':
            # engine parameters
            defaults = {'t_MECO' : 9.3,   # Main Engine Cut Off time
                        'thrust' : 900.,  # thrust (const)
                        } 
                        
        elif dict_type == 'parachute':
            # parachute parameters
            defaults = {'t_deploy' : 1000., # parachute deployment time
                        } 
        
        return defaults
    
    
    def set_parameters(self, params={},dict_type=None):
        # =============================================
        # update parameters of simulation
        #
        # INPUT: params = user-defined dict containing parameters
        #        dict_type = type of parameters
        # =============================================
        if dict_type == 'execute':
            # update executive parameters
            self.execute.update(params)
            
        elif dict_type == 'launch':
            # update launch condition parameters
            self.launch.update(params)
            
        elif dict_type == 'rocket':
            # update rocket parameters
            self.rocket.update(params)   
            
        elif dict_type == 'engine':
            # update engine parameters
            self.engine.update(params)
            
        elif dict_type == 'parachute':
            # update launch condition parameters
            self.parachute.update(params)
        #END IF
        return

    """
    ----------------------------------------------------
    ----     Method for main computation          ------
    ----------------------------------------------------
    """
    
    def run(self,postprocess=None):
        # =============================================
        # actually run ODE integration
        #
        # INPUT: postprocess = type of postprocess
        # =============================================
        
        # "dict of dict" containing all parameters
        params_all = {
                      'exec'     : self.execute,
                      'launch'   : self.launch,
                      'rocket'   : self.rocket,
                      'engine'   : self.engine,
                      'parachute': self.parachute,
                       }
        # create instance of trajectry simulation
        trajectory = simu_main(params_all)
        
        print '----------------------------'
        print '  Completed Parameter difinition'
        print '----------------------------'
        print ' '
        
        # run ODE integration
        trajectory.ODE_main()
        
        # post processing
        if postprocess == 'plot_all':
            print '----------------------------'
            print '  Post-processing'
            print '----------------------------'
            print ' '
            
            self.visualize_trajectory(trajectory.history)
            self.plot_loc(trajectory.history)
            self.plot_velocity(trajectory.history)
            self.plot_omega(trajectory.history)
        #END IF
            
        # quit
        print '----------------------------'
        print '  Quit simulation'
        print '----------------------------'
        print ' '
        return
        
        
        
    """
    ----------------------------------------------------
    ----     Methods for post_processing          ------
    ----------------------------------------------------
    """
        
    def visualize_trajectory(self,history):
        # visualize 3D trajectory
        flag = history[:,1]
        xloc = history[:,2]
        yloc = history[:,3]
        zloc = history[:,4]
    
        # sort index
        ids1 = np.where(flag==1)  # on launch rail
        ids2 = np.where(flag==2)  # thrusted flight
        ids3 = np.where(flag==3)  # inertial flight
        ids4 = np.where(flag==4)  # parachute deploy
        
        
        fig = plt.figure()
        ax = Axes3D(fig)
        # lanuch rail
        try:
            ax.plot(xloc[ids1], yloc[ids1], zloc[ids1],lw=3,label='On rail')
        except:
            pass
        # thrusted
        try:
            ax.plot(xloc[ids2], yloc[ids2], zloc[ids2],lw=3,label='Thrusted')
        except:
            pass
        # inertial
        try:
            ax.plot(xloc[ids3], yloc[ids3], zloc[ids3],lw=3,label='Inertial')
        except:
            pass
        # parachute
        try:
            ax.plot(xloc[ids3], yloc[ids4], zloc[ids4],lw=3,label='Parachute')
        except:
            pass
            
    
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Trajectory')
        ax.legend()
        plt.show()
        
        return
    
    
    def plot_loc(self,history):
        # plot x,y,z location as a function of time
        time = history[:,0]
        xloc = history[:,2]
        yloc = history[:,3]
        zloc = history[:,4]
    
        plt.figure()
        # x
        plt.plot(time,xloc,lw=1.5,label='x')
        # y
        plt.plot(time,yloc,lw=1.5,label='y')
        # z
        plt.plot(time,zloc,lw=4,label='z')
        plt.legend()
        plt.title('XYZ history')
        plt.xlabel('t [s]')
        plt.ylabel('xyz [m]')
        plt.show   
        
        return
        
    def plot_velocity(self,history):
        # plot u,v,w velocity and speed as a function of time
        time = history[:,0]
        u = history[:,5]
        v = history[:,6]
        w = history[:,7]
        speed = np.linalg.norm(history[:,5:8],axis=1)
    
        plt.figure()
        # u
        plt.plot(time,u,lw=1.5,label='Vx')
        # v
        plt.plot(time,v,lw=1.5,label='Vy')
        # w
        plt.plot(time,w,lw=1.5,label='Vz')
        # speed
        plt.plot(time,speed,lw=5,label='Speed')
        plt.legend()
        plt.title('Velocity history')
        plt.xlabel('t [s]')
        plt.ylabel('v [m/s]')
        plt.show  
        
        return 
        
    def plot_omega(self,history):
        # plot omega_x,y,z velocity and speed as a function of time
        history.shape
        time = history[:,0]
        p = history[:,12]  # around x
        q = history[:,13]  # around y
        r = history[:,14]  # around z
    
        plt.figure()
        # p
        plt.plot(time,p,lw=1.5,label='omega_x')
        # q
        plt.plot(time,q,lw=1.5,label='omega_y')
        # r
        plt.plot(time,r,lw=1.5,label='omega_z')
    
        plt.legend()
        plt.title('Angular velocity history')
        plt.xlabel('t [s]')
        plt.ylabel('omega [rad/s]')
        plt.show 
        
        return
            
            
        
        
        
        
         
     
         
    
    

        
        
        
   
        
        
        
    
    