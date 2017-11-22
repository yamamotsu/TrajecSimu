
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
        self.rocket    = self.get_default('rocket')     # rocket engine parameters
        self.engine    = self.get_default('engine')     # rocket engine parameters
        self.parachute = self.get_default('parachute')  # parachute parameters       
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
                        'wind_speed'     : 4.,      # wind speed [m/s] at 10m alt. 
                        'Cwind'          : 1./7.4,  # wind model power coefficient 
                        }  
                        
        elif dict_type == 'rocket':
            # rocket airframe parameters
            defaults = {
                        'aero_fin_mode'    : 'integ',  # 'indiv' for individual fin computation, 'integ' for compute fin-body at once
                        }
            
        elif dict_type == 'engine':
            # rocket engine parameters
            defaults = {'t_MECO' : 9.3,   # Main Engine Cut Off (MECO) time
                        'thrust' : 800.,  # thrust (const.)
                        } 
                        
        elif dict_type == 'parachute':
            # parachute parameters
            defaults = {'t_deploy' : 100000., # parachute deployment time from ignition
                        't_para_delay': 1., # parachute deployment time from apogee detection
                        'Cd_para': 1.,      # parachute drag coefficient
                        'S_para': 0.64,     # parachute area [m^2]
                        } 
        
        return defaults
    
    
    def set_parameters(self, params={}, params_type=None, data_type='dataframe'):
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

        if data_type == 'dict':
            # input var type: dict 
            if params_type == 'execute':
                # update executive parameters
                self.execute.update(params)
                
            elif params_type == 'launch':
                # update launch condition parameters
                self.launch.update(params)
                
            elif params_type == 'rocket':
                # update rocket parameters
                self.rocket.update(params)   
                
            elif params_type == 'engine':
                # update engine parameters
                self.engine.update(params)
                
            elif params_type == 'parachute':
                # update launch condition parameters
                self.parachute.update(params)
            #END IF

        # integrate all parameters into a dict:
        self.params_all = {}
        self.params_all.update(self.execute)
        self.params_all.update(self.launch)
        self.params_all.update(self.rocket)
        self.params_all.update(self.engine)
        self.params_all.update(self.parachute)

        if data_type == 'dataframe':
            # input var type: pandas dataframe
            # note: all variables are in the same file
            
            # update (convert Dataframe -> array -> dict)
            self.params_all.update( dict( params.as_matrix() ) )
        #END IF   
            
        # print(self.params_all.items(),'\n')
        
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
                       
        # create an instance for a trajectry simulation
        self.trajectory = simu_main(self.params_all)   # providing parameters here
        
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
        
        x_loc = 0.
        y_loc = 0.
       
        if process_type == 'location':
            # *** return landing location ***
            x_loc,y_loc = self.show_landing_location()
           
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
            dt = self.params_all['dt']      # time step
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
        
        return x_loc, y_loc
        
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
        
        return xloc, yloc
        
        
    def show_max_values(self,time):
        # =============================================
        # this method shows max values of M, q, speed, altitude
        # =============================================
        
        # array of rho, a histories: use standard air
        n = len(self.trajectory.solution[:,2])
        T = np.zeros(n)
        p = np.zeros(n)
        rho = np.zeros(n)
        a = np.zeros(n)
        
        for i in range(n):
            T[i],p[i],rho[i],a[i] = self.trajectory.standard_air(self.trajectory.solution[i,2])  # provide altitude=u[2]
        #END IF
        
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
        
        # get wind speed at Max_Q
        wind_vec = self.trajectory.wind(self.trajectory.solution[Q_max,2]*4.)  # provide altitude=u[2]
        wind_speed = np.linalg.norm(wind_vec)

        # flight time: the last value of time array
        print('----------------------------')
        print(' Max. Mach number: ',"{0:.3f}".format(speed[M_max]/a[M_max]),' at t=',"{0:.2f}".format(time[M_max]),'[s]')
        print(' Max. Q: ', "{0:6.2e}".format(0.5*rho[Q_max]*speed[Q_max]**2.), '[Pa] at t=',"{0:.2f}".format(time[Q_max]),'[s]')
        print(' Max. speed: ', "{0:.1f}".format(speed[v_max]),'[m/s] at t=',"{0:.2f}".format(time[v_max]),'[s]')
        print(' Max. altitude: ', "{0:.1f}".format(self.trajectory.solution[h_max,2]), '[m] at t=',"{0:.2f}".format(time[h_max]),'[s]')
        print(' total flight time: ', "{0:.2f}".format(self.trajectory.landing_time),'[s]')
        print('----------------------------')
        
        # output flight condition at Max.Q
        print(' Flight conditions at Max-Q.')
        print(' free-stream pressure: ', "{0:6.2e}".format(p[Q_max]) ,'[Pa]')
        print(' free-stream temperature: ', "{0:.1f}".format(T[Q_max]) ,'[T]')
        print(' free-stream Mach: ', "{0:.3f}".format(speed[Q_max]/a[Q_max]) )
        print(' Wind speed: ',  "{0:.2f}".format(wind_speed),'[m/s]')
        print(' Angle of attack for gust rate 2: ', "{0:.1f}".format(np.arctan( wind_speed/speed[Q_max])*180./np.pi ),'[deg]')
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
        t_MECO = self.params_all['t_MECO']
        t_deploy = self.trajectory.t_deploy
        dt = self.params_all['dt']
        
        # ***_t: thrusted flight (before MECO)
        # ***_c: inertial flight
        # ***_p: parachute fall
        try:
            time_t, time_c, time_p = np.split(time,[ int(np.ceil(t_MECO/dt)), int(np.ceil(t_deploy/dt)) ] )
            xloc_t, xloc_c, xloc_p = np.split(xloc,[ int(np.ceil(t_MECO/dt)), int(np.ceil(t_deploy/dt)) ] )
            yloc_t, yloc_c, yloc_p = np.split(yloc,[ int(np.ceil(t_MECO/dt)), int(np.ceil(t_deploy/dt)) ] )
            zloc_t, zloc_c, zloc_p = np.split(zloc,[ int(np.ceil(t_MECO/dt)), int(np.ceil(t_deploy/dt)) ] )
        except:
            time_t, time_c = np.split(time,[int(np.ceil(t_MECO/dt))])
            xloc_t, xloc_c = np.split(xloc,[int(np.ceil(t_MECO/dt))])
            yloc_t, yloc_c = np.split(yloc,[int(np.ceil(t_MECO/dt))])
            zloc_t, zloc_c = np.split(zloc,[int(np.ceil(t_MECO/dt))])
            
            # create plot
        fig = plt.figure()
        ax = Axes3D(fig)
        
        # plot powered-phase trajectory
        ax.plot(xloc_t, yloc_t, zloc_t,lw=3,label='Powered')

        # plot coast-phase trajectory
        try:
            ax.plot(xloc_c, yloc_c, zloc_c,lw=3,label='Coast')
        except:
            pass
        
        # plot parachute descent-phase trajectory
        try:
            ax.plot(xloc_p, yloc_p, zloc_p,lw=3,label='Parachute')
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
        plt.grid()
        plt.show   
        
        return
        
    def plot_velocity(self,time):
        # =============================================
        # this method plots u,v,w (velocity wrt earth) as a function of time
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
        plt.grid()
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
        plt.grid()
        plt.show 
        
        return
          
        