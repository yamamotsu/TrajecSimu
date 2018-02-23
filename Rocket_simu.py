
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 22:23:48 2017

@author: shugo
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import fftpack, interpolate, integrate, optimize
import pandas as pd
import subprocess

# class
class Rocket():
    """
    ====================================================
    This is a class for simulation parameter setting, execution of main computation, and post-processing.
    An instance of this class should be created at first.
    Main computation will be executed by a sub-class "trajec_main"
    ====================================================
    """
    
    """
    ----------------------------------------------------
        Methods for initilization and parameter setting            
    ----------------------------------------------------
    """
    
    def __init__(self, csv_filename):
        # =============================================
        # This method is called when an instance is created
        # =============================================
        
        # read csv file that contains initial parameters
        self.params_df = pd.read_csv(csv_filename, comment='$', names=('parameter', 'value') ) # '$' denotes commet out  
        
        # initialize parameters by setting default values
        self.get_default()   
        self.overwrite_parameters(self.params_df)
        
        # initialize a dict for result
        self.res = {}

        return None      
    
    def overwrite_dataframe(self, params):   
        # =============================================
        # this method overwrites pd.dataframe parameter values.
        # used for loop / optimization 
        #
        # input: params : n*2 numpy array that contains parameters to be updated
        #                 [ ['param_name1', value], ['param_name2', value] , ...]
        #
        # =============================================
        
        # loop over parameters in params
        for i in range(params.shape[0]):
            self.params_df.loc[self.params_df.parameter == params[i,0], 'value'] = params[i,1]
        # END FOR
        
        return None
        
    def get_default(self):
        # =============================================
        # This method defines default parameters
        # =============================================
        self.params_dict = {
                            # -----------------------------
                            # numerical executive parameters
                            # -----------------------------
                            'dt'       : 0.05,          # time step [s]
                            't_max'    : 100*60,         # maximum time [s]
                            'N_record' : 500,           # record history every N_record iteration
                            'integ' : 'lsoda_odeint',   # time integration scheme       
                        
                            # -----------------------------
                            # launch condition parameters          
                            # -----------------------------  
                            # launcher configuration
                            'elev_angle'     : 89.,     # angle of elevation [deg]
                            'azimuth'        : 0.,      # north=0, east=90, south=180, west=270 [deg]
                            'rail_length'    : 5.,      # length of launcher rail
                            # atmosphere property
                            'T0'             : 288.,    # temperature at 10m altitude [K] 
                            'p0'             : 1.013e5, # pressure  at 10m alt. [Pa]
                            # wind property
                            'wind_direction'   : 0.,      # azimuth where wind is blowing from 
                            'wind_speed'       : 4.,      # wind speed at 'wind_alt_std' alt. [m/s] 
                            'wind_power_coeff' : 7.,
                            'wind_alt_std'     : 10.,      # alt. at which the wind speed is given [m]
    
                            # -----------------------------
                            # rocket aerodynamic parameters
                            # ----------------------------- 
                            'aero_fin_mode'    : 'integ',  # 'indiv' for individual fin computation, 'integ' for compute fin-body at once
                            'Cd0'              : 0.6,      # drag coefficient at Mach 0.1, AoA = 0deg
                            'Cmp'              : -0.,      # stability derivative of rolling moment coefficient (aerodynamic damping)
                            'Cmq'              : -3.,     # stability derivative of pitching/yawing moment coefficient (aerodynamic damping)
                            'Cl_alpha'         : 10.,      # lift slope for small AoA

                            # -----------------------------
                            # rocket engine parameters
                            # ----------------------------- 
                            #'t_MECO' : 9.3,   # Main Engine Cut Off (MECO) time
                            #'thrust' : 800.,  # thrust (const.)
                            'thrust_input_type' : 'curve_const_t',   # thrust input csv file type
                            'curve_fitting'     : True,              # True if curvefitting
                            'fitting_order'     : 15,                # order of polynomial
                            'thrust_mag_factor' : 1.,                # thrust magnification factor
                            'time_mag_factor'   : 1.,                # burn time magnification factor
    
                            # -----------------------------       
                            # parachute parameters
                            # -----------------------------
                            't_deploy' : 1000.,   # parachute deployment time from ignition
                            't_para_delay': 1000.,   # parachute deployment time from apogee detection
                            'Cd_para': 1.,       # parachute drag coefficient
                            'S_para': 0.5,      # parachute area [m^2]
                            } 

        return None
            
    
    def overwrite_parameters(self, params_df_userdef={}):
        # =============================================
        # This method updates default parameters with user-defined parameters. 
        # User-defined parameters should be provided in pandas dataframe
        #
        # INPUT: params_df = dataframe containing parameters
        # =============================================

        # overwrite param_dict (convert Dataframe -> array -> dict)
        self.params_dict.update( dict( params_df_userdef.as_matrix() ) ) 

        # set instance variables from params_dict
        # -----------------------------
        # numerical executive 
        # -----------------------------
        try:
            self.dt = float( self.params_dict['dt'] )            # time step
            self.t_max = float(self.params_dict['t_max'] )       # maximum time
            self.N_record = float(self.params_dict['N_record'] ) # record history every ** iteration
            self.integ = self.params_dict['integ'].strip()       # time integration scheme
        except:
            # display error message
            print('Error in executive control parameters')
            sys.exit()
        
        # -----------------------------
        # launch condition
        # -----------------------------
        try:
            # launcher property
            rail_length = float( self.params_dict['rail_length'] )            # length of launch rail 
            self.elev_angle = float( self.params_dict['elev_angle'] )         # angle of elevation [deg]
            self.azimuth = float( self.params_dict['azimuth'] )               # north=0, east=90, south=180, west=270 [deg]
            self.rail_height = rail_length * np.sin(self.elev_angle * np.pi/180.) # height of launch rail in fixed coord.
    
            # atmosphere property
            self.T0 = float( self.params_dict['T0'] )  # temperature [K] at 10m alt.
            self.p0 = float( self.params_dict['p0'] )  # pressure [Pa] at 10m alt.
            
            # wind property
            wind_direction = float( self.params_dict['wind_direction'] )  # azimuth where wind is blowing from [deg]
            angle_wind = (-wind_direction + 90.) * np.pi/180.    # angle to which wind goes (x orients east, y orients north)
            self.wind_unitvec = -np.array([np.cos(angle_wind), np.sin(angle_wind) ,0.])
            self.wind_speed = float( self.params_dict['wind_speed'] )         # speed of wind [m/s] at 10m alt.
            self.Cwind = 1./float( self.params_dict['wind_power_coeff'] )   # wind power coefficient
            self.wind_alt_std = float( self.params_dict['wind_alt_std'])
            # earth gravity
            self.grav = np.array([0.,0.,-9.81])    # in fixed coordinate
            
        except:
            # display error message
            print('Error in launch condition parameters')
            sys.exit()
        
        # -----------------------------
        # mass/inertia properties
        # -----------------------------
        try:
            self.m_dry = float( self.params_dict['m_dry'] )       # dry weight of rocket i.e. exclude fuel
            self.m_prop = float( self.params_dict['m_prop'] )     # propellant weight // NOTE: total weight at lift off = m_dry + m_prop
            self.CG_dry = float( self.params_dict['CG_dry'] )     # CG location of dried body (nose tip = 0)
            self.CG_prop =float(  self.params_dict['CG_prop'] )   # CG location of prop (assume CG_fuel = const.)
            self.MOI_dry = np.array([float( self.params_dict['MOI_dry_x'] ), float( self.params_dict['MOI_dry_y'] ), float( self.params_dict['MOI_dry_z']) ])    # dry MOI (moment of inertia)
            self.MOI_prop = np.array([float( self.params_dict['MOI_prop_x']), float( self.params_dict['MOI_prop_y']), float( self.params_dict['MOI_prop_z']) ])  # dry MOI (moment of inertia)
            
        except:
            # display error message
            print('Error in mass property parameters')
            sys.exit()

        # -----------------------------
        # aerodynamic properties
        # -----------------------------
        try:
            # rocket dimension
            rocket_height = float(self.params_dict['rocket_height'] ) # height of rocket
            rocket_diameter = float(self.params_dict['rocket_diameter'] ) # height of rocket
            
            try:
                self.CP_body = float( self.params_dict['CP_body'] )  # CP location without fins (budy tube+nose) (nose tip = 0)
            except:
                # default CP: 15%Fst
                self.CP_body = self.CG_dry + 0.15*rocket_height
                
            self.Cd0 = float( self.params_dict['Cd0'] )          # total 0 angle-of-attack drag coefficient
            self.X_area = np.pi*rocket_diameter**2. /4.  # cross-sectional area
            self.Cl_alpha = float( self.params_dict['Cl_alpha'] )   
            
            self.aero_fin_mode = self.params_dict['aero_fin_mode'].strip()  # 'indiv' for individual fin computation, 'integ' for compute fin-body at once
            
            # aerodynamic damping moment coefficient]
            # non_dimensional
            Cm_omega = np.array([ float(self.params_dict['Cmp']), float(self.params_dict['Cmq']), float(self.params_dict['Cmq']) ]) 
            # Cm_omega_bar = Cm_omega * l^2 * S
            self.Cm_omega_bar = Cm_omega * np.array([rocket_diameter, rocket_height, rocket_height])**2. * self.X_area # multiply with length^2. no longer non-dimansional
            
            if self.aero_fin_mode == 'indiv':
                # for individual fin computation, define fin parameters here
                self.LE_fins = float( self.params_dict['LE_fins'] )  # root leading edge of fin location (nose tip = 0)
                self.alpha_fins = float( self.params_dict['fin_alpha'] ) # fin attachment angle
                fin_h = float( self.params_dict['fin_h'] )
                fin_dis = float( self.params_dict['fin_dis'] )
                fin_y = np.linspace(fin_h/fin_dis/2. ,fin_h-fin_h/fin_dis/2. ,fin_dis)
                fin_xLE = 1/np.tan((90.-float( self.params_dict['fin_ang'])*np.pi/180.)) * fin_y   # leading edge locations
                self.fin_len = ( float(self.params_dict['fin_Ltip'])*fin_y + (fin_h-fin_y)*float(self.params_dict['fin_Lroot']) )/ fin_h    # length of fin
                self.xFP = fin_xLE + self.fin_len/4.   # forcing points  # array to store forcing-point locations (0=leading edge*root)
                self.r_arm = fin_y + float( self.params_dict['rocket_diameter'] )/2. # array of arm length for roll-spin
                self.dy_fin = fin_h / fin_dis  # y step of fin discretization
            # END IF
        except:
            # display error message
            print('Error in aerodynamic property parameters')
            sys.exit()
        
        # -----------------------------
        # engine property
        # -----------------------------
        try:
            self.thrust_input_type = self.params_dict['thrust_input_type'].strip() 
            self.thrust_mag_factor = float(self.params_dict['thrust_mag_factor'] )
            self.time_mag_factor = float(self.params_dict['time_mag_factor'] )
            
            if self.thrust_input_type == 'rectangle':
                # rectangle thrust input (constant thrust * burn time)
                self.t_MECO = float( self.params_dict['t_MECO'] )
                self.thrustforce = float( self.params_dict['thrust'] )
                
            else:
                if self.thrust_input_type == 'curve_const_t':
                    # thrust curve with constant time step (csv of 1 raw)
                    self.thrust_dt = float( self.params_dict['thrust_dt'] )
                    self.thrust_filename = self.params_dict['thrust_filename'].strip()
                
                elif self.thrust_input_type == 'time_curve':
                    # time and thrust log is given in csv.
                    self.thrust_filename = self.params_dict['thrust_filename'].strip()
                # END IF
                self.curve_fitting = self.params_dict['curve_fitting']
                self.fitting_order = self.params_dict['fitting_order']
            # END IF

            # setup thrust 
            self.setup_thrust( self.thrust_mag_factor,  self.time_mag_factor)  # magnification
            
        except:
            # display error message
            print('Error in engine property parameters')
            sys.exit()
        
        # -----------------------------  
        # parachute property
        # -----------------------------
        try:
            self.t_deploy = float( self.params_dict['t_deploy'] )         # parachute deployment time from ignition
            self.t_para_delay = float( self.params_dict['t_para_delay'] ) # parachute deployment time from apogee detection
            self.apogee_count = 0                                         # apogee count
            self.Cd_para = float( self.params_dict['Cd_para'] )           # parachute drag coefficient
            self.S_para = float( self.params_dict['S_para'] )             # parachute area [m^2]
        except:
            # display error message
            print('Error in parachute property parameters')  
            sys.exit()
        # print(self.params_all.items(),'\n')  

        return None   
    
    
        
        
        
        
    """
    ----------------------------------------------------
        Method for thrust properties setup          
    ----------------------------------------------------
    """
    
    def setup_thrust(self, thrust_factor = 1., time_factor = 1.):
        # =============================================
        # this method sets up thrust curve from dataframe input or CSV thrust curve
        #
        # input: thrust_factor : thrust magnification factor
        #        time_factor   : burn time magnification factor
        #
        # =============================================
        
        thrust_factor = 1.
        time_factor = 1.
        
        if self.thrust_input_type == 'rectangle':
            # rectangle thrust input (constant thrust * burn time)
            
            # setup interp1d function
            self.time_array = np.array([0, self.t_MECO]) * time_factor
            self.thrust_array = np.ones(2) * self.thrustforce * thrust_factor
                
        else:
            # input thrust curve from csv file
            
            # path to csv (hardcoded)
            PATH_to_csv = 'Thrust_curve_csv'
            # filename with path
            filename = PATH_to_csv + '/' + self.thrust_filename
            # read csv file
            input_raw = np.array( pd.read_csv(filename, header=None) ) 
            
            if self.thrust_input_type == 'curve_const_t': 
                # raw thrust array
                thrust_raw = input_raw 
                
                # cut off info where thrust is less that 1% of T_max
                self.thrust_array = thrust_raw[ thrust_raw >= 0.01*np.max(thrust_raw) ] * thrust_factor
                # time array
                self.time_array = np.arange(0., len(self.thrust_array)*self.thrust_dt, self.thrust_dt) * time_factor
                
            elif self.thrust_input_type == 'time_curve': 
                # time array
                self.time_array = input_raw[:,0]
                # thrust array
                self.thrust_array = input_raw[:,1] 
                
                # cut-off and magnification
                self.time_array = self.time_array[ self.thrust_array >= 0.01*np.max(self.thrust_array)] * time_factor
                self.thrust_array = self.thrust_array[ self.thrust_array >= 0.01*np.max(self.thrust_array)] * thrust_factor
            # END IF
              
        # -----------------------
        #   get raw engine property
        # -----------------------
        # maximum thrust
        self.Thrust_max = np.max(self.thrust_array)
        # total impulse
        self.Impulse_total = integrate.trapz(self.thrust_array, self.time_array)
        # averaged thrust
        self.Thrust_avg = self.Impulse_total / self.time_array[-1]
        # MECO time
        self.t_MECO = self.time_array[-1]
        
        
        if self.thrust_input_type == 'rectangle':
            # set 1d interpolation function for rectangle thrust
            self.thrust_function = interpolate.interp1d(self.time_array, self.thrust_array, fill_value='extrapolate')
            
        else:
            # ------------------
            # noise cancellation
            # ------------------
            if self.thrust_input_type == 'curve_const_t': 
                
                # FFT (fast fourier transformation)
                tf = fftpack.fft(self.thrust_array)
                freq = fftpack.fftfreq(len(self.thrust_array), self.thrust_dt)
                
                # filtering 
                fs = 5.                         # cut off frequency [Hz]
                tf2 = np.copy(tf)
                tf2[(freq > fs)] = 0
                
                # inverse FFT
                self.thrust_array = np.real(fftpack.ifft(tf2))
            # END IF
            
            
            # -------------------
            #   represent raw thrust carve by "curve fitting" or "1d interpolation"
            # -------------------
            if self.curve_fitting: 
                # curve fitting 
                n_fit = self.fitting_order  # order of fitting
                a_fit = np.polyfit(self.time_array, self.thrust_array, n_fit)
            
                # define polynomial that returns thrust for a given time
                self.thrust_function = np.poly1d(a_fit)
                
                # total impulse for fitting function
                thrust_poly = self.thrust_function(self.time_array)   # polynomially fitted thrust curve
                thrust_poly[thrust_poly<0.] = 0.                      # overwrite negative value with 0
                Impulse_total_poly = integrate.trapz(thrust_poly, self.time_array)
                # error of total impulse [%]
                self.It_poly_error = abs(Impulse_total_poly - self.Impulse_total) / self.Impulse_total * 100.
                
            else:
                # 1d interpolation
                self.thrust_function = interpolate.interp1d(self.time_array, self.thrust_array, fill_value='extrapolate')
            
        # END IF
            
        
        # set interp1d function
        
        
        
    """
    ----------------------------------------------------
        Methods for main computation            
    ----------------------------------------------------
    """
    
    def run(self):
        # =============================================
        # A base method for a single trajectory simulation
        #
        # INPUT: params_df = dataframe containing user-defined parameters
        # =============================================
                      
        # import sub-class
        from trajectory import trajec_main
        
        # create an instance for a trajectry simulation
        self.trajectory = trajec_main(self.params_df)  # provide parameters for sub-class 
        
        #"""
        print('============================')
        print('  Completed Parameters Setup')
        print('============================')
        print(' ')
        #"""
        
        # run ODE integration
        self.trajectory.ODE_main()
                    
        # quit
        #"""
        print('============================')
        print('  Quit ODE_main')
        print('============================')
        print(' ')
        #"""

        return None
    
    
    
    
    """
    ----------------------------------------------------
        Methods for post_processing      
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
       
        if process_type == 'location':
            # *** get landing location  ***
            self.get_landing_location()
           
        elif process_type == 'maxval':
            # *** return max M, q, speed, altitude, flight time, and landing location  ***
            
            # create time array to find out the max values 
            time = self.trajectory.t    # extract time array
            dt = self.trajectory.dt                # time step
            land_time_id = int(np.ceil(self.trajectory.landing_time/dt)) # id of time array when the rocket landed
            time = time[0:land_time_id] # array before landing: cut off useless after-landing part
            # landing_time = time[-1]
            
            # get thrust deta
            self.echo_thrust()
            # get landing location
            self.get_landing_location() 
            # get max values
            self.get_flight_detail(time)
            
        elif process_type == 'all':
            # *** plot all variable histories along with max values and landing location ***
            print('============================')
            print('  Post-processing')
            print('============================')
            print(' ')
        
            # creat time array to plot
            time = self.trajectory.t     # extract time array
            dt = self.trajectory.dt                 # time step
            land_time_id = int(np.ceil(self.trajectory.landing_time/dt)) # id of time array when the rocket landed
            time = time[0:land_time_id] # array before landing: cut off useless after-landing part
            # cut off useless info out of ODE solution array
            self.trajectory.solution = self.trajectory.solution[0:len(time),:]
            
            # *** plot and show all results ***
            # thrust data echo
            self.echo_thrust(True)
            
            # get landing location
            self.get_landing_location(True)
            # get flight detail
            self.get_flight_detail(time,True)
            
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
            
            # show all plots
            plt.show()
                
        else:
            # if input variable "process_type" is incorrect, show error message
            print('error: process_type must be "location" or "max" or "all". ')
        #END IF
        
        return None
        
    def echo_thrust(self,show=False):
        # =============================================
        # this method plots thrust curve along with engine properties
        # =============================================
        
        # engine properties
            
        # record engine property
        tmp_dict = {'total_impulse  ': self.trajectory.Impulse_total, 
                    'max_thrust'     : self.trajectory.Thrust_max,
                    'average_thrust' : self.trajectory.Thrust_avg,
                    'burn_time'      : self.trajectory.t_MECO,
                }
        self.res.update({'engine' : tmp_dict})

        if show:
            print('--------------------')
            print(' THRUST DATA ECHO')
            print(' total impulse (raw): ', self.trajectory.Impulse_total, '[N.s]')
            try:
                if self.trajectory.curve_fitting:
                    print('       error due to poly. fit: ', self.trajectory.It_poly_error, ' [%]')    
                # END IF
            except:
                pass
            print(' burn time: ', self.trajectory.t_MECO, '[s]')
            print(' max. thrust: ', self.trajectory.Thrust_max, '[N]')
            print(' average thrust: ', self.trajectory.Thrust_avg, '[N]')
            print('--------------------')
            # plot filtered thrust curve
            fig = plt.figure(1)
            plt.plot(self.trajectory.time_array, self.trajectory.thrust_array, color='red', label='raw')
            try:
                if self.trajectory.curve_fitting:
                    thrust_used = self.trajectory.thrust_function(self.trajectory.time_array)
                    thrust_used[thrust_used<0.] = 0.
                    plt.plot(self.trajectory.time_array, thrust_used, lw = 3, label='fitted')
                # END IF
            except:
                pass
            plt.title('Thrust curve')
            plt.xlabel('t [s]')
            plt.ylabel('thrust [N]')
            plt.grid()
            plt.legend()
            #plt.show () 
        # END IF
        
        return None
        
    def get_landing_location(self, show=False):
        # =============================================
        # this method gets the location that the rocket has landed
        # =============================================
        
        # landing point coordinate is is stored at the end of array "history"
        xloc = self.trajectory.solution[-1,0]
        yloc = self.trajectory.solution[-1,1]
        zloc = self.trajectory.solution[-1,2]
        
        # record landing location in dict
        self.res.update( { 'landing_loc' : np.array([xloc, yloc]) } )
        
        if show:
            # show result
            print('----------------------------')
            print('landing location:')
            print('[x,y,z] = ', xloc, yloc, zloc)
            print('----------------------------')
        # END IF
        
        return None
        
        
    def get_flight_detail(self,time, show=False):
        # =============================================
        # this method gets flight detail (max values of M, q, speed, altitude)
        # =============================================
        
        # array of rho, a histories: use standard air
        n = len(self.trajectory.solution[:,2])
        T = np.zeros(n)
        p = np.zeros(n)
        rho = np.zeros(n)
        a = np.zeros(n)
        # time array
        time = self.trajectory.t 
        
        
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
        
        # ------------------------------
        # record flight detail
        # ------------------------------
        maxq_dict = {'max_Q' : 0.5*rho[Q_max]*speed[Q_max]**2.,  # max. dynamic pressure
                     'time'  : time[Q_max],                      # time
                     'free_stream_pressure': p[Q_max],
                     'free_stream_temperature' : T[Q_max], 
                     'free_stream_Mach: ' : speed[Q_max]/a[Q_max],
                     'wind_speed: ' : wind_speed,
                     'AoA_for_gustrate2' : np.arctan( wind_speed/speed[Q_max])*180./np.pi,
                     }
        tmp_dict = {'max_Mach' : np.array([speed[M_max]/a[M_max], time[M_max] ])  ,          # [max. Maxh number, time]
                    'max_speed': np.array([speed[v_max], time[v_max]]),                      # max. speed
                    'max_altitude'  : np.array([self.trajectory.solution[h_max,2], time[h_max]]), # max. altitude
                    'total_flight_time': self.trajectory.landing_time,                       # total flight time
                    'max_Q_all'  : maxq_dict,
                    }
        tmp_dict.update( self.trajectory.res_trajec_main )  # add results from trajectory
        self.res.update({'flight_detail' : tmp_dict})
        
        if show:
            # show results
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

        return None  
        
        
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
        t_MECO = self.trajectory.t_MECO
        t_deploy = self.trajectory.t_deploy
        dt = self.trajectory.dt
        
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
        fig = plt.figure(2)
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
        # ax.set_aspect('equal')
        #.show()
        
        return None
    
    
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
        plt.figure(3)
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
        #plt.show   
        
        return None
        
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
        
        plt.figure(4)
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
        #plt.show  
        
        return None
        
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
        
        
        plt.figure(5)
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
        #plt.show 
        
        return None
    
    """
    
    def plot_dist(self, wind_speed_array):
        # plot landing spot distribution

        n_windspeed, n_winddir, _ = self.loc_bal.shape
        # ----------------------------
        # Izu umi zone: D = 5km
        # ----------------------------
        R = 2500.
        center = np.array([1768., -1768.])
        theta = np.linspace(0,2*np.pi,100)
        x_circle = R * np.cos(theta) + center[0]
        y_circle = R * np.sin(theta) + center[1]
        
        # ----------------------------
        #    ballistic
        # ----------------------------
        plt.figure(6)
        # loop over wind speed
        for i in range(n_windspeed):
            # plot ballistic landing distribution
            legend =  str(wind_speed_array[i]) + 'm/s'
            
            plt.plot(self.loc_bal[i,:,0] ,self.loc_bal[i,:,1] ,label=legend )
            plt.plot(self.loc_bal[i,0,0] ,self.loc_bal[i,0,1] , color='b', marker='x', markersize=4)  # wind: 0 deg
            plt.plot(self.loc_bal[i,1,0] ,self.loc_bal[i,1,1] , color='b', marker='x', markersize=4)  # wind: 0+ deg
        #END IF
        
        
        plt.grid()
        plt.legend()
        plt.title('ballistic landing distribution')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.plot(0,0,color='r',marker='*',markersize=12)
        plt.plot(x_circle, y_circle,color='r',lw=5)
        #plt.show()
        
        
        # ----------------------------
        #    parachute
        # ----------------------------
        plt.figure(7)
        # loop over wind speed
        for i in range(n_windspeed):
            # plot parachute landing distribution
            legend =  str(wind_speed_array[i]) + 'm/s'
            
            plt.plot(self.loc_para[i,:,0] ,self.loc_para[i,:,1] ,label=legend )
            plt.plot(self.loc_para[i,0,0],self.loc_para[i,0,1] , color='b', marker='x', markersize=4) # wind: 0 deg
            plt.plot(self.loc_para[i,1,0],self.loc_para[i,1,1] , color='b', marker='x', markersize=4) # wind: 0+ deg
        #END IF
        
        plt.grid()
        plt.legend()
        plt.title('parachute landing distribution')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.axis('equal')
        plt.plot(0,0,color='r',marker='*',markersize=12)
        plt.plot(x_circle, y_circle,color='r',lw=5)
        # plt.show
        
        return None
        
    """
          
        