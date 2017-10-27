
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
    ====================================================
    This class is.... coming soon!
    
    
    
    ====================================================
    """
    
    """
    ----------------------------------------------------
                Methods for initilization            
    ----------------------------------------------------
    """
    
    def __init__(self):
        # =============================================
        # this method is called when an instance is created
        # =============================================
        
        # initialize parameters: set default values
        self.execute   = self.get_default('execute')    # numerical executive parameters
        self.launch    = self.get_default('launch')     # launch condition parameters
        self.engine    = self.get_default('engine')     # rocket engine parameters
        self.parachute = self.get_default('parachute')  # parachute parameters
        self.rocket    = {}        
        return
    
        
    def get_default(self,dict_type=None):
        # =============================================
        # this method returns default parameters
        # =============================================
        if dict_type == 'execute':
            # numerical executive parameters
            defaults = {'dt'       : 0.05,          # time step
                        't_max'    : 20*60,         # maximum time
                        'N_record' : 500,           # record history every N_record iteration
                        'integ' : 'lsoda_odeint',   # time integration scheme
                        }         
                        
        elif dict_type == 'launch':
            # launch condition parameters            
            defaults = {
                        # launcher configuration
                        'elev_angle'     : 89.,     # angle of elevation [deg]
                        'azimuth'        : 0.,      # north=0, east=90, south=180, west=270 [deg]
                        'rail_length'    : 5.,      # length of launcher rail
                        # air property
                        'T0'             : 288.,    # temperature [K] at 10m altitude
                        'p0'             : 1.013e5, # pressure [Pa] at 10m alt.
                        'wind_direction' : 0.,      # azimuth where wind is blowing from 
                        'wind_speed'     : 2.,      # wind speed [m/s] at 10m alt. 
                        'Cwind'          : 1./6.5,  # wind model power coefficient 
                        }  
                        
        elif dict_type == 'engine':
            # rocket engine parameters
            defaults = {'t_MECO' : 9.3,   # Main Engine Cut Off (MECO) time
                        'thrust' : 900.,  # thrust (const.)
                        } 
                        
        elif dict_type == 'parachute':
            # parachute parameters
            defaults = {'t_deploy' : 1000., # parachute deployment time
                        } 
        
        return defaults
    
    
    def set_parameters(self, params={},dict_type=None):
        # =============================================
        # this method updates default parameters with user-defined parameters. 
        #
        # INPUT: params = user-defined dict containing parameters
        #        dict_type = type of parameters
        #                    'execute'   : numerical executive parameters
        #                    'launch'    : launch condition parameters
        #                    'rocket'    : rocket configuration parameters
        #                    'engine'    : engine parameters
        #                    'parachute' : parachute deployment parameters
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
                Methods for main computation            
    ----------------------------------------------------
    """
    
    def run(self):
        # =============================================
        # this method runs ODE integration
        # =============================================
        
        # create a "dict of dict" that contains all the parameters
        params_all = {
                      'exec'     : self.execute,
                      'launch'   : self.launch,
                      'rocket'   : self.rocket,
                      'engine'   : self.engine,
                      'parachute': self.parachute,
                       }
                       
        # create an instance for a trajectry simulation
        self.trajectory = simu_main(params_all)   # providing parameters here
        
        print('----------------------------')
        print('  Completed Parameters Setup')
        print('----------------------------')
        print(' ')
        
        # run ODE integration
        self.trajectory.ODE_main()
                    
        # quit
        print('----------------------------')
        print('  Quit simulation')
        print('----------------------------')
        print(' ')
        return
        
        
        
    """
    ----------------------------------------------------
    ----     Methods for post_processing          ------
    ----------------------------------------------------
    """
    def postprocess(self,process_type='all'):
        # =============================================
        # this method controls post-processing. 
        #
        # INPUT: process_type = post-processing type. default = 'all'
        #        dict_type = type of parameters
        #                    'location'   : returns only landing location. 
        #                    'maxval'     : returns max values of interest along with the landing location
        #                    'all'        : plot all variable histories along with max values and landing location.
        # =============================================
        print('----------------------------')
        print('  Post-processing')
        print('----------------------------')
        print(' ')
       
        if process_type == 'location':
            # *** return landing location ***
            self.show_landing_location()
           
        elif process_type == 'maxval':
            # *** return max M, q, speed, altitude, flight time, and landing location  ***
            
            # creat time array to find out the max values 
            time = self.trajectory.t    # extract time array
            dt = self.execute['dt']     # time step
            land_time_id = int(np.ceil(self.trajectory.landing_time/dt)) # id of time array when the rocket landed
            time = time[0:land_time_id] # array before landing: cut off useless after-landing part
            
            # returns landing location
            self.show_landing_location() 
            # returns max values
            self.show_max_values(time)
            
        elif process_type == 'all':
            # *** plot all variable histories along with max values and landing location ***
            
            # creat time array to plot
            time = self.trajectory.t     # extract time array
            dt = self.execute['dt']      # time step
            land_time_id = int(np.ceil(self.trajectory.landing_time/dt)) # id of time array when the rocket landed
            time = time[0:land_time_id] # array before landing: cut off useless after-landing part
            # cut off useless info out of ODE solution array
            self.trajectory.solution = self.trajectory.solution[0:len(time),:]
            
            # *** plot and show all results ***
            # returns landing location
            self.show_landing_location()
            # returns max values
            self.show_max_values(time)
            
            # plot trajectory
            self.visualize_trajectory(time)
            # plot xyz history
            self.plot_loc(time)
            # plot velocity/speed history
            self.plot_velocity(time)
            # plot angular velocity history
            self.plot_omega(time)
            # plot Mach number history
            
            # plot dynamic pressure history
            
            
                
                
        else:
            # if input variable "process_type" is incorrect, show error message
            print('error: process_type must be "location" or "max" or "all". ')
        #END IF
        
    def show_landing_location(self):
        # =============================================
        # this method shows the location that the rocket has landed
        # =============================================
        
        # landing point coordinate is is stored at the end of array "history"
        xloc = self.trajectory.solution[-1,0]
        yloc = self.trajectory.solution[-1,1]
        zloc = self.trajectory.solution[-1,2]
        print('----------------------------')
        print('landing location:')
        print('[x,y,z] = ', xloc, yloc, zloc)
        print('----------------------------')
        
        
    def show_max_values(self,time):
        # =============================================
        # this method shows max values of M, q, speed, altitude
        # =============================================
        
        # array of rho, a histories: use standard air
        rho,a = self.trajectory.standard_air(self.trajectory.solution[:,2])  # provide altitude=u[2]
        
        # array of speed history
        speed = np.linalg.norm(self.trajectory.solution[:,3:6],axis=1) # provide velocity=u[3:6]
        
        # index of max. Mach number
        M_max = np.argmax(speed / a)
        
        # index of max Q
        Q_max = np.argmax(0.5 * rho * speed**2.)
        
        # index of max. speed
        v_max = np.argmax(speed)
        
        # index of max. altitude: max. of z
        h_max = np.argmax(self.trajectory.solution[:,2])
        
        # flight time: the last value of time array
        print('----------------------------')
        print('Max. Mach number: ',"{0:.3f}".format(speed[M_max]/a[M_max]),' at t=',"{0:.2f}".format(time[M_max]),'[s]')
        print('Max. Q: ', "{0:6.2e}".format(0.5*rho[Q_max]*speed[Q_max]**2.), '[Pa] at t=',"{0:.2f}".format(time[Q_max]),'[s]')
        print('Max. speed: ', "{0:.1f}".format(speed[v_max]),'[m/s] at t=',"{0:.2f}".format(time[v_max]),'[s]')
        print('Max. altitude: ', "{0:.1f}".format(self.trajectory.solution[h_max,2]), '[m] at t=',"{0:.2f}".format(time[h_max]),'[s]')
        print('total flight time: ', "{0:.2f}".format(self.trajectory.landing_time),'[s]')
        print('----------------------------')
        
        
    def visualize_trajectory(self,time):
        # =============================================
        # this method visualizes 3D trajectory
        # =============================================

        # xyz location history
        xloc = self.trajectory.solution[:,0]
        yloc = self.trajectory.solution[:,1]
        zloc = self.trajectory.solution[:,2]

        """
        # adjust array length
        if len(time) < len(xloc):
            # adjust the length of time array 
            xloc = xloc[0:len(time)]
            xloc = xloc[0:len(time)]
            xloc = xloc[0:len(time)]
        # END IF
        """
        
        # split arrays for each flight mode
        t_MECO = self.engine['t_MECO']
        dt = self.execute['dt']
        
        # ***_t: thrusted flight (before MECO)
        # ***_i: inertial flight
        time_t, time_i = np.split(time,[int(np.ceil(t_MECO/dt))])
        xloc_t, xloc_i = np.split(xloc,[int(np.ceil(t_MECO/dt))])
        yloc_t, yloc_i = np.split(yloc,[int(np.ceil(t_MECO/dt))])
        zloc_t, zloc_i = np.split(zloc,[int(np.ceil(t_MECO/dt))])
        
        # create plot
        fig = plt.figure()
        ax = Axes3D(fig)
        
        # plot thrusted trajectory
        ax.plot(xloc_t, yloc_t, zloc_t,lw=3,label='Thrusted')

        # plot inertial trajectory
        try:
            ax.plot(xloc_i, yloc_i, zloc_i,lw=3,label='Inertial')
        except:
            pass
            
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title('Trajectory')
        ax.legend()
        plt.show()
        
        return
    
    
    def plot_loc(self,time):
        # =============================================
        # this method plots x,y,z location as a function of time
        # =============================================
        
        # xyz location history
        xloc = self.trajectory.solution[:,0]
        yloc = self.trajectory.solution[:,1]
        zloc = self.trajectory.solution[:,2]
    
        """
        # time array
        time = self.trajectory.t
        if len(time) > len(xloc):
            # adjust the length of time array 
            time = time[0:len(xloc)]
        # END IF
        """
        
        # create plot
        plt.figure()
        # plot x history
        plt.plot(time,xloc,lw=1.5,label='x')
        # plot y history
        plt.plot(time,yloc,lw=1.5,label='y')
        # plot z history
        plt.plot(time,zloc,lw=4,label='z')
        plt.legend()
        plt.title('XYZ vs. time')
        plt.xlabel('t [s]')
        plt.ylabel('xyz [m]')
        plt.show   
        
        return
        
    def plot_velocity(self,time):
        # =============================================
        # this method plots u,v,w (velocity) as a function of time
        # =============================================
        
        # velocity = [u,v,w] history
        u = self.trajectory.solution[:,3]
        v = self.trajectory.solution[:,4]
        w = self.trajectory.solution[:,5]
        #speed = np.linalg.norm(history[:,3:6],axis=1)
        
        """
        # time array
        time = self.trajectory.t
        if len(time) > len(u):
            # adjust the length of time array 
            time = time[0:len(u)]
        # END IF
        """
        
        plt.figure()
        # u history
        plt.plot(time,u,lw=1.5,label='Vx')
        # v history
        plt.plot(time,v,lw=1.5,label='Vy')
        # w history
        plt.plot(time,w,lw=1.5,label='Vz')
        # speed history
        #plt.plot(time,speed,lw=5,label='Speed')
        plt.legend()
        plt.title('Velocity vs. time')
        plt.xlabel('t [s]')
        plt.ylabel('v [m/s]')
        plt.show  
        
        return 
        
    def plot_omega(self,time):
        # =============================================
        # this method plots omega_x,y,z velocity and speed as a function of time
        # =============================================
        
        # omega = [p,q,r] history
        p = self.trajectory.solution[:,10]  # angular velocity around x
        q = self.trajectory.solution[:,11]  # angular velocity y
        r = self.trajectory.solution[:,12]  # angular velocity z
        
        """
        # time array
        time = self.trajectory.t
        if len(time) > len(p):
            # adjust the length of time array 
            time = time[0:len(p)]
        # END IF
        """
        
        plt.figure()
        # p history 
        plt.plot(time,p,lw=1.5,label='omega_x')
        # q history
        plt.plot(time,q,lw=1.5,label='omega_y')
        # r history
        plt.plot(time,r,lw=1.5,label='omega_z')
    
        plt.legend()
        plt.title('Angular velocity vs. time')
        plt.xlabel('t [s]')
        plt.ylabel('omega [rad/s]')
        plt.show 
        
        return
          
        
    """          
    def standard_air(self,h):
        # ==============================================
        # this method returns air property given an altitude 
        # INPUT: h = altitude [m]
        # ==============================================
        
        # temperature goes down 0.0065K/m until it reaches -56.5C (216.5K)
        #                                       it is approximately 11km
        T0 = self.launch['T0']
        p0 = self.launch['p0']
        T = T0 - 0.0065*h # [K]
        
        # temperature is const at 216.5 K for alt. < 20km
        T[T<216.5] = 216.5
            
        # pressure
        p = p0 * (T/T0)**5.256  #[Pa]

        # density
        rho = p/(287.15*T) #[kg/m^3]
        
        # acoustic speed
        a = np.sqrt(1.4*287.15*T) # [m/s]
        
        return rho,a 
        
    """
        
        
        
         
     
         
    
    

        
        
        
   
        
        
        
    
    