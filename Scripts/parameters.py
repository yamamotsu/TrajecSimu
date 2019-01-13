#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 18:22:15 2018

@author: shugo
"""

import numpy as np
import pandas as pd
from Scripts.errors import *
from scipy import fftpack, interpolate, integrate
from distutils.util import strtobool
import json
from Scripts.statistics_wind import *

# class for parameters
class Parameters():
    """
    ====================================================
    This is a class for rocket parameters setup
    ====================================================
    """


    def __init__(self, csv_filename):
        # =============================================
        # This method is called when an instance is created
        #
        # INPUT: csv_filename = path and filename of rocket parameters file
        # =============================================

        # read csv file that contains initial parameters
        try:
            self.params_df = pd.read_csv(csv_filename, comment='$', names=('parameter', 'value') ) # '$' denotes commet out
        except:
            self.params_df = pd.read_csv(csv_filename, comment='$', names=('parameter', 'value', ' ') ) # '$' denotes commet out
        self.params_df['value'] = self.params_df['value'].str.strip()

        # initialize parameters by setting default values
        self.get_default()
        self.overwrite_parameters(self.params_df)

        # initialize a dict for results
        # self.res = {}

        return None

    def overwrite(self, params):
        # wrapper method for overwrite parameters
        self.overwrite_dataframe(params)
        self.overwrite_parameters(self.params_df)

    def overwrite_dataframe(self, params):
        # =============================================
        # this method overwrites pd.dataframe parameter values.
        # used for loop / optimization
        #
        # input: params = n*2 numpy array that contains parameters to be updated
        #                 [ ['param_name1', value], ['param_name2', value] , ...]
        #
        # =============================================

        # loop over parameters in params
        for i in range(len(params)):
            self.params_df.loc[self.params_df.parameter == params[i][0], 'value'] = params[i][1]
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
                            # launch point configuration
                            'latitude'       : 35.,      # latitude of launch point [deg]

                            # atmosphere property
                            'T0'             : 298.,    # temperature at 10m altitude [K]
                            'p0'             : 1.013e5, # pressure  at 10m alt. [Pa]
                            # wind property
                            'wind_direction'   : 0.,      # azimuth where wind is blowing FROM
                            'wind_speed'       : 4.,      # wind speed at 'wind_alt_std' alt. [m/s]
                            'wind_power_coeff' : 7.,
                            'wind_alt_std'     : 10.,      # alt. at which the wind speed is given [m]

                            # wind model
                            'wind_model'       : 'power',  # 'power for Wind Power Method, 'power-forecast-hydrid' for power-forecast hybrid'


                            # -----------------------------
                            # rocket aerodynamic parameters
                            # -----------------------------
                            'aero_fin_mode'     : 'integ',  # 'indiv' for individual fin computation, 'integ' for compute fin-body at once
                            'Cd0'               : 0.6,      # drag coefficient at Mach 0.1, AoA = 0deg
                            'Cmp'               : -0.,      # stability derivative of rolling moment coefficient (aerodynamic damping)
                            'Cmq'               : -4.,      # stability derivative of pitching/yawing moment coefficient (aerodynamic damping)
                            'Cl_alpha'          : 12.,      # lift coefficient slope for small AoA [1/rad]
                            'Mach_AOA_dep'      : True,     # True if aerodynamic parameter depends on Mach/AOA, False if ignore


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
                            't_deploy' : 1000.,      # parachute deployment time from ignition
                            't_para_delay': 1000.,   # 1st parachute deployment time from apogee detection
                            'Cd_para': 1.,           # drag coefficient of 1st parachute
                            'S_para': 0.5,           # parachute area of 1st parachute[m^2]
                            'second_para': False,    # True if two stage parachute deployment
                            't_deploy_2': 20000,     # 2nd parachute deployment time from apogee detection
                            'Cd_para_2': 1,          # drag coefficient of 2nd parachute
                            'S_para_2': 6.082,       # parachute area of 2nd parachute[m^2]
                            'alt_para_2': -100,      # altitude at which 2nd parachute will be deployed
                            }

        return None


    def overwrite_parameters(self, params_df_userdef={}):
        # =============================================
        # This method updates default parameters with user-defined parameters.
        # User-defined parameters should be provided in pandas dataframe
        #
        # INPUT: params_df = dataframe containing parameter names/values to be updated.
        # =============================================

        # overwrite param_dict (convert Dataframe -> array -> dict)
        self.params_dict.update( dict( params_df_userdef.as_matrix() ) )

        # set instance variables from params_dict
        # -----------------------------
        # numerical executive
        # -----------------------------
        try:
            self.dt       = float( self.params_dict['dt'] )            # time step
            self.t_max    = float(self.params_dict['t_max'] )          # maximum time
            self.N_record = float(self.params_dict['N_record'] )       # record history every ** iteration
            self.integ    = self.params_dict['integ'].strip()          # time integration scheme
        except:
            raise ParameterDefineError('executive control')

        # -----------------------------
        # launch condition
        # -----------------------------
        try:
            # launcher property
            rail_length      = float( self.params_dict['rail_length'] )            # length of launch rail
            self.elev_angle  = float( self.params_dict['elev_angle'] )             # angle of elevation [deg]
            self.azimuth     = float( self.params_dict['azimuth'] )                # north=0, east=90, south=180, west=270 [deg]
            self.rail_height = rail_length * np.sin(np.deg2rad(self.elev_angle))   # height of launch rail in fixed coord.
            # launch point property
            self.omega_earth = np.array([0., 0., -7.29e-5*np.sin( np.deg2rad( float(self.params_dict['latitude']) ) ) ])  # Earth rotation velocity

            # atmosphere property
            self.T0 = float( self.params_dict['T0'] )  # temperature [K] at 10m alt.
            self.p0 = float( self.params_dict['p0'] )  # pressure [Pa] at 10m alt.

            # wind property
            self.wind_direction    = float( self.params_dict['wind_direction'] )               # azimuth where wind is blowing from [deg]
            angle_wind        = np.deg2rad( (-self.wind_direction + 90.) )                     # angle to which wind goes (x orients east, y orients north)
            self.wind_unitvec = -np.array([np.cos(angle_wind), np.sin(angle_wind) ,0.])   # wind unitvector (blowing TO)
            self.wind_speed   = float( self.params_dict['wind_speed'] )                   # speed of wind [m/s] at 10m alt.
            self.Cwind        = 1./float( self.params_dict['wind_power_coeff'] )          # wind power coefficient
            self.wind_alt_std = float( self.params_dict['wind_alt_std'])
            self.wind_model   = self.params_dict['wind_model']

            #if self.wind_model == 'power-forecast-hydrid':
            #    self.wind_forecast_csvname = self.params_dict['forecast_csvname']        # csv file name of wind forecast model
            # if self.params_dict['forecast_csvname'] is not None:
            if 'forecast_csvname' in self.params_dict:
                self.wind_forecast_csvname\
                    = self.params_dict['forecast_csvname']
                self.setup_forcast()

            if 'statistics_filename' in self.params_dict:
                self.wind_statistics_filename\
                    = self.params_dict['statistics_filename']
                self.setup_statistics()
            # earth gravity
            self.grav = np.array([0., 0., -9.81])    # in fixed coordinate

        except:
            raise ParameterDefineError('launch condition')

        # -----------------------------
        # mass/inertia properties
        # -----------------------------
        try:
            self.m_dry    = float( self.params_dict['m_dry'] )       # dry weight of rocket i.e. exclude fuel
            self.m_prop   = float( self.params_dict['m_prop'] )      # propellant weight // NOTE: total weight at lift off = m_dry + m_prop
            self.CG_dry   = float( self.params_dict['CG_dry'] )      # CG location of dried body (nose tip = 0)
            self.CG_prop  =float(  self.params_dict['CG_prop'] )     # CG location of prop (assume CG_fuel = const.)
            self.MOI_dry  = np.array([float( self.params_dict['MOI_dry_x'] ), float( self.params_dict['MOI_dry_y'] ), float( self.params_dict['MOI_dry_z']) ])    # dry MOI (moment of inertia)
            self.MOI_prop = np.array([float( self.params_dict['MOI_prop_x']), float( self.params_dict['MOI_prop_y']), float( self.params_dict['MOI_prop_z']) ])   # dry MOI (moment of inertia)

        except:
            raise ParameterDefineError('mass property')

        # -----------------------------
        # aerodynamic properties
        # -----------------------------
        try:
            # rocket dimension
            rocket_height   = float(self.params_dict['rocket_height'] )   # height of rocket
            rocket_diameter = float(self.params_dict['rocket_diameter'] ) # height of rocket
            self.X_area     = np.pi*rocket_diameter**2. /4.               # cross-sectional area

            # Aerodynamic parameters
            try:
                self.CP_body = float( self.params_dict['CP_body'] )         # CP location(nose tip = 0)
            except:
                self.CP_body = self.CG_dry + 0.15*rocket_height                        # default CP: 15%Fst
            self.Cd0      = float( self.params_dict['Cd0'] )                           # total drag coefficient at AoA=0deg
            self.Cl_alpha = float( self.params_dict['Cl_alpha'] )                      # lift slope [1/rad]
            self.Mach_AOA_dependent = self.params_dict['Mach_AOA_dep']     # flag whether aerodynamic parameters are dependent to Mach/AoA
            if type(self.Mach_AOA_dependent) == str:
                self.Mach_AOA_dependent = strtobool(self.Mach_AOA_dependent)
            self.aero_fin_mode = self.params_dict['aero_fin_mode'].strip()             # 'indiv' for individual fin computation, 'integ' for compute fin-body at once
            Cm_omega = np.array([ float(self.params_dict['Cmp']), float(self.params_dict['Cmq']), float(self.params_dict['Cmq']) ])  # aerodynamic damping moment coefficient
            # Dimensional coeff: Cm_omega_bar = Cm_omega * l^2 * S
            self.Cm_omega_bar = Cm_omega * np.array([rocket_diameter, rocket_height, rocket_height])**2. * self.X_area # multiply with length^2. no longer non-dimansional

            # For fin-body separated computation (currently not supported.)
            if self.aero_fin_mode == 'indiv':
                raise NotImplementedError('fin indivisual computation is currently not implemented.')
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
            raise ParameterDefineError('aerodynamic property')

        # -----------------------------
        # Tip-Off properties
        # -----------------------------
        # location of 1st(front) and 2nd(rear) lug (from nose tip)
        try:
            lug_1st = float( self.params_dict['lug_1st'] )
            self.lug_2nd = float( self.params_dict['lug_2nd'] )
        except:
            # set default values
            lug_1st = 0.3 * rocket_height
            self.lug_2nd = 0.8 * rocket_height

        # initial CG
        CG_init = ( self.m_dry*self.CG_dry + self.m_prop*self.CG_prop) / (self.m_dry+self.m_prop)
        # when CG point is above "height_1stlug_off", 1st lug is off the rail
        self.height_1stlug_off = (rail_length - (CG_init - lug_1st) ) * np.sin(np.deg2rad(self.elev_angle))
        # when CG point is above "height_2ndlug_off", 2nd lug is off the rail
        self.height_2ndlug_off = (rail_length + (self.lug_2nd - CG_init) ) * np.sin(np.deg2rad(self.elev_angle))
        #
        self.height_nozzle_off = (rail_length + (rocket_height - CG_init) ) * np.sin(np.deg2rad(self.elev_angle))

        # -----------------------------
        # Engine properties
        # -----------------------------
        try:
            self.thrust_input_type = self.params_dict['thrust_input_type'].strip()   # engine thrust input type
            self.thrust_mag_factor = float(self.params_dict['thrust_mag_factor'] )   # thrust magnification factor
            self.time_mag_factor   = float(self.params_dict['time_mag_factor'] )     # burn time magnification factor

            if self.thrust_input_type == 'rectangle':
                # rectangle thrust input (constant thrust * burn time)
                self.t_MECO      = float( self.params_dict['t_MECO'] )
                self.thrustforce = float( self.params_dict['thrust'] )

            else:
                if self.thrust_input_type == 'curve_const_t':
                    # thrust curve with constant time step (csv of 1 raw)
                    self.thrust_dt       = float( self.params_dict['thrust_dt'] )
                    self.thrust_filename = self.params_dict['thrust_filename'].strip()

                elif self.thrust_input_type == 'time_curve':
                    # time and thrust log is given in csv.
                    self.thrust_filename = self.params_dict['thrust_filename'].strip()
                else:
                    raise ParameterDefineError(' engine property')
                # END IF
                self.curve_fitting = self.params_dict['curve_fitting']
                if type(self.curve_fitting) == str:
                    self.curve_fitting = strtobool(self.curve_fitting)
                self.fitting_order = int(self.params_dict['fitting_order'])
            # END IF

            # setup thrust fitting curve
            self.setup_thrust( self.thrust_mag_factor,  self.time_mag_factor)  # magnification

        except:
            raise ParameterDefineError('engine property')


        # -----------------------------
        # parachute properties
        # -----------------------------
        try:
            self.t_deploy     = float( self.params_dict['t_deploy'] )         # parachute deployment time from ignition
            self.t_para_delay = float( self.params_dict['t_para_delay'] )     # parachute deployment time from apogee detection
            self.Cd_para      = float( self.params_dict['Cd_para'] )          # drag coefficient of 1st parachute
            self.S_para       = float( self.params_dict['S_para'] )           # area of 1st prarachute [m^2]
            self.flag_2ndpara = self.params_dict['second_para']               # True if two stage separation
            if type(self.flag_2ndpara) == str:
                    self.flag_2ndpara = strtobool(self.flag_2ndpara)
            if self.flag_2ndpara:
                # if two stage deployment is true, define 2nd stage parachute properties
                self.t_deploy_2 = float( self.params_dict['t_deploy_2'] )     # 2nd parachute deployment time from ignition# self.t_para_delay = float( self.params_dict['t_para_delay'] )   # 2nd parachute deployment time from apogee detection
                self.Cd_para_2  = float( self.params_dict['Cd_para_2'] )      # drag coefficient of 2nd parachute
                self.S_para_2   = float( self.params_dict['S_para_2'] )       # net area of 2nd prarachute [m^2]
                self.alt_para_2 = float( self.params_dict['alt_para_2'] )     # altitude of 2nd parachute deployment
            # END IF
        except:
            raise ParameterDefineError('parachute property')


        return None


    """
    ----------------------------------------------------
        Method for thrust curve setup
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

        if self.thrust_input_type == 'rectangle':
            # rectangle thrust input (constant thrust * burn time)

            # setup interp1d function
            self.time_array = np.array([0, self.t_MECO]) * time_factor
            self.thrust_array = np.ones(2) * self.thrustforce * thrust_factor

        else:
            # input thrust curve from csv file

            # path to csv (hardcoded)
            #PATH_to_csv = 'Thrust_curve_csv'
            # filename with path
            #filename = PATH_to_csv + '/' + self.thrust_filename
            filename = self.thrust_filename
            # read csv file
            input_raw = np.array( pd.read_csv(filename, header=None) )

            if self.thrust_input_type == 'curve_const_t':
                # raw thrust array
                thrust_raw = input_raw[:, 1]

                time_raw = input_raw[:, 0]
                # thrust array
                #self.time_array = time_raw * time_factor
                #self.thrust_array = thrust_raw * thrust_factor

                # cut off info where thrust is less that 1% of T_max
                self.thrust_array = thrust_raw[thrust_raw >= 0.01*np.max(thrust_raw)]*thrust_factor

                # time array
                # self.time_array = np.arange(0., len(self.thrust_array)*self.thrust_dt, self.thrust_dt) * time_factor
                self.time_array = time_raw[thrust_raw >= 0.01*np.max(thrust_raw)]*time_factor

            elif self.thrust_input_type == 'time_curve':
                # time array
                self.time_array = input_raw[:,0]
                # thrust array
                self.thrust_array = input_raw[:,1]

                # cut-off and magnification
                self.time_array = self.time_array[ self.thrust_array >= 0.01*np.max(self.thrust_array)] * time_factor
                self.thrust_array = self.thrust_array[ self.thrust_array >= 0.01*np.max(self.thrust_array)] * thrust_factor
            else:
                raise ParameterDefineError('Thrust input type definition is wrong.')
            # END IF

        # -----------------------
        #   get raw engine property
        # -----------------------
        self.Thrust_max    = np.max(self.thrust_array) # maximum thrust
        self.Impulse_total = integrate.trapz(self.thrust_array, self.time_array) # total impulse
        self.Thrust_avg    = self.Impulse_total / self.time_array[-1]  # averaged thrust
        self.t_MECO        = self.time_array[-1] # MECO time

        self.It_poly_error = 0.

        if self.thrust_input_type == 'rectangle':
            # set 1d interpolation function for rectangle thrust
            self.thrust_function = interpolate.interp1d(self.time_array, self.thrust_array, fill_value='extrapolate')

        else:
            # ------------------
            # noise cancellation
            # ------------------
            if self.thrust_input_type == 'curve_const_t':
                # FFT (fast fourier transformation)
                tf   = fftpack.fft(self.thrust_array)
                freq = fftpack.fftfreq(len(self.thrust_array), self.thrust_dt)
                # filtering
                fs = 10.                         # cut off frequency [Hz]
                tf2 = np.copy(tf)
                tf2[np.abs(freq) > fs] = 0
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
                # define polynomial that returns thrust for a given time (fitted thrust curve)
                self.thrust_function = np.poly1d(a_fit)
                # total impulse for fitting function
                time_for_poly = np.linspace(self.time_array[0], self.time_array[-1], 1e4)
                thrust_poly = self.thrust_function(time_for_poly)   # polynomially fitted thrust curve
                thrust_poly[thrust_poly<0.] = 0.                      # overwrite negative value with 0
                Impulse_total_poly = integrate.trapz(thrust_poly, time_for_poly)
                # error of total impulse [%]
                self.It_poly_error = abs(Impulse_total_poly - self.Impulse_total) / self.Impulse_total * 100.
            else:
                # 1d interpolation
                self.thrust_function = interpolate.interp1d(self.time_array, self.thrust_array, fill_value='extrapolate')
            # END IF
        # END IF

        return None

    """
    ----------------------------------------------------
        Method for aerodynamic properties (Mach, AoA independent) setup
    ----------------------------------------------------
    """

    def setup_aero_coeffs(self):
        # ==============================================
        # set up aerodynamic coefficients and C.P. location interpolations (default)
        # Put following .dat files in /bin
        #   Cd0.dat:     Mach numbers vs. Cd0
        #   Clalpha.dat: Mach numbers vs. Clalpha
        #
        # if self.Mach_AOA_dependent = False, set up simple function that returns constant Cd0, Cl_alpha, and CP location.
        # ==============================================

        if self.Mach_AOA_dependent:
            # When true, setup interpolation curves to represent
            # aero coeff dependency on Mach / AOA
            # --------------------------
            # drag coeff. at 0 AOA
            # --------------------------
            # setup Cd0 (Cd at AOA=0) curve as a function of Mach number
            # Cd0 = 0.6 here, but actural Cd in the program is scaled by the config input Cd0.

            # read dat file that contains Mach vs Cd0 relations.
            # n*2 ndarray. 1st column: Mach numbers, 2nd column: Cd0s.
            try:
                data = np.loadtxt('bin/Cd0.dat',delimiter=",", skiprows=1)
            except:
                raise FileNotFoundError(' bin/Cd0.dat not found')
            Mach_array = data[:,0]
            Cd0_array = data[:,1] * ( self.Cd0 / data[0,1] ) # scaling

            # create 2d interpolation (Mach vs Cd0)
            self.f_cd0 = interpolate.interp1d(Mach_array,Cd0_array,kind='linear')

            # --------------------------
            # lift coeff. slope
            # --------------------------
            # setup Cl_alpha (Cl slope wrt AOA) curve as a function of Mach number
            # Cl_alpha = 10.0 (M=0.3) here, but actural Cl_alpha in the program is scaled by the config input Cl_alpha.

            # read dat file that contains Mach vs Cl_alpha relations.
            # n*2 ndarray. 1st column: Mach numbers, 2nd column: Cl_alpha
            try:
                data = np.loadtxt('bin/Clalpha.dat',delimiter=",", skiprows=1)
            except:
                raise FileNotFoundError(' bin/Clalpha.dat not found')
            Mach_array = data[:,0]
            Clalpha_array = data[:,1] * ( self.Cl_alpha / data[0,1] ) # scaling

            # create 2d interpolation (Mach vs Cl_alpha
            self.f_cl_alpha = interpolate.interp1d(Mach_array,Clalpha_array,kind='linear')

            # --------------------------
            # C.P. location
            # --------------------------
            # import C.P. location data
            try:
                df = pd.read_csv("bin/CPloc.csv", header=None, na_values='Mach/AOA')
            except:
                raise FileNotFoundError(' bin/CPloc.csv not found')
            Mach_array = np.array(df.iloc[1:,0])  # mach array
            AOA_array = np.array(df.iloc[0,1:]) * np.pi/180.  # AOA array (convert from deg to rad)
            # create grid
            Mach2, AOA2 = np.meshgrid(Mach_array, AOA_array)

            # CPlocation 2D array (rows: Mach, columns: AOA)
            CPloc_array = np.array(df.iloc[1:,1:])
            # convert from % to len, and scaling. Use CPloc at M=0.3, AOA=2deg as standard
            # tmpfunc = interpolate.interp2d(Mach2, AOA2, CPloc_array.T, kind='linear')
            tmpfunc = interpolate.RectBivariateSpline(Mach_array, AOA_array, CPloc_array)
            CPloc_array *= self.CP_body /tmpfunc(0.3, np.deg2rad(2.))

            # create 2d interpolation curve
            self.f_CPloc =interpolate.RectBivariateSpline(Mach_array, AOA_array, CPloc_array)

        else:
            def f_cd0(a):
                 # constant Cd0
                return self.Cd0

            def f_cl_alpha(a):
                # constant Cl_alpha
                return self.Cl_alpha

            def f_CPloc(a,b):
                # constant CPlocation
                return np.array([self.CP_body])

            self.f_cd0 = f_cd0
            self.f_cl_alpha = f_cl_alpha
            self.f_CPloc = f_CPloc
        # END IF

        return None

    """
    ----------------------------------------------------
        Method for wind model setup
    ----------------------------------------------------
    """

    def setup_wind(self):
        # ==============================================
        # set up wind, i.e. wind speed/direction as a function of altitude.
        #   default: power law
        # ==============================================

        if self.wind_model == 'power':
            # -------------------
            # use power law only
            # -------------------
            self.wind = self.wind_power

        elif self.wind_model == 'log':
            self.wind = self.wind_log

        elif self.wind_model == 'forecast':
            self.wind = self.wind_forecast
        elif self.wind_model == 'statistics':
            self.wind = self.wind_statistics
        elif self.wind_model == 'power-forecast-hybrid':
            # -------------------
            # power law and forecast hybrid model
            # -------------------
            # definition of wind power-forecast hybrid method
            def wind_power_forecast(h):
                if h < 0.:
                    h = 0.

                #boundary_alt = 100.
                #transition = 20.

                # NOTE: 2018/10/08: changed parameters for Izu-Riku Nov 2018
                boundary_alt = 200.
                transition = 100.

                if h <= boundary_alt - transition:
                    # use power law only
                    return self.wind_power(h)
                elif h <= boundary_alt + transition:
                    # use both
                    weight = (h - (boundary_alt-transition) ) / (2*transition)
                    return weight*self.wind_forecast(h) + (1.-weight)*self.wind_power(h)
                else:
                    # use forecast only
                    return self.wind_forecast(h)
                # END IF
            # END OF DIFINITION

            self.wind = wind_power_forecast
        elif self.wind_model == 'log-forecast-hybrid':
            def wind_log_forecast(h):
                if h<=0.1:
                    # to avoid log(0)
                    h = 0.1

                #boundary_alt = 100.
                #transition = 20.

                # NOTE: 2018/10/08: changed parameters for Izu-Riku Nov 2018
                boundary_alt = 200.
                transition = 100.

                if h <= boundary_alt - transition:
                    # use log law only
                    return self.wind_log(h)
                elif h <= boundary_alt + transition:
                    # use both
                    weight = (h - (boundary_alt-transition) ) / (2*transition)
                    return weight*self.wind_forecast(h) + (1.-weight)*self.wind_log(h)
                else:
                    # use forecast only
                    return self.wind_forecast(h)

            self.wind = wind_log_forecast
        elif self.wind_model == 'power-statistics-hybrid':
                def wind_power_statistics(h):
                    if h < 0.:
                        h = 0.

                    boundary_alt = 1000.
                    transition = 500.

                    if h <= boundary_alt - transition:
                        # use power law only
                        return self.wind_power(h)
                    elif h <= boundary_alt + transition:
                        # use both
                        weight = (h - (boundary_alt-transition) ) / (2*transition)
                        return weight*self.wind_statistics(h) + (1.-weight)*self.wind_power(h)
                    else:
                        # use statistics only
                        return self.wind_statistics(h)
                    # END IF
                # END OF DIFINITION
                self.wind = wind_power_statistics
        else:
            raise ParameterDefineError('wind model definition is wrong.')

    def setup_forcast(self):
        try:
            # import weather forecast info
            df = pd.read_csv(self.wind_forecast_csvname)
        except:
            raise FileNotFoundError(' Wind forecast data file not found')

        # altitude array
        alt = np.array(df['altitude'])

        # Wind: west to east (x)
        wind_W2E_tmp = np.array(df['Wind (from west)'])
        # south to north (y)
        wind_S2N_tmp = np.array(df['Wind (from south)'])
        # Upward (z)
        wind_UP = np.array(df['Wind (vertical)'])

        # magnetic angle correction
        theta = np.deg2rad(8.9)
        wind_W2E = wind_W2E_tmp * np.cos(theta) + wind_S2N_tmp* np.sin(theta)
        wind_S2N =  - wind_W2E_tmp * np.sin(theta) + wind_S2N_tmp* np.cos(theta)
        # set as vector (blowing TO)
        wind_vec_fore = np.c_[wind_W2E, wind_S2N, wind_UP].T

        # setup wind_forecast interpolation function
        self.wind_forecast = interpolate.interp1d(alt, wind_vec_fore, fill_value='extrapolate')

    def setup_statistics(self):
        try:
            with open(self.wind_statistics_filename, 'r') as f:
                params = json.load(f)
        except:
            raise FileNotFoundError(' Wind statistics data file not found')

        print('statistics parameterfile loaded')
        alt = params['alt_axis']
        # alt_index_std = params['altitude_idx_std']
        n_alt = len(alt)
        # wind_std = params['mu4'][alt_index_std][2:]

        '''
        wind_tmp = getStatWindVector(
                        statistics_parameters=params,
                        wind_std=np.array(wind_std))
        '''

        wind_tmp = getStatWindVector(
                        statistics_parameters=params,
                        wind_direction_deg=self.wind_direction
                        )

        wind_vec_stat = np.c_[wind_tmp[0], wind_tmp[1], [0] * n_alt].T
        # setup wind_statistics interpolation function
        self.wind_statistics = interpolate.interp1d(
                                alt,
                                wind_vec_stat,
                                fill_value='extrapolate')

    # definition of wind log law
    def wind_log(self, h):
        if h <= 0.1:
            # to avoid log(0)
            h = 0.1
        # END IF

        # roughness of surface
        roughness_surf = 0.0003

        wind_vec = self.wind_unitvec * self.wind_speed *\
           (np.log10(h/roughness_surf)/np.log10(self.wind_alt_std/roughness_surf))

        return wind_vec

    # definition of wind power law
    def wind_power(self, h):
        if h<0.:
            h = 0.
        # END IF

        # wind velocity in local fixed coordinate (wind vector = direction blowing TO. Wind from west = [1,0,0])
        wind_vec = self.wind_unitvec * self.wind_speed * (h/self.wind_alt_std)**self.Cwind

        return wind_vec
