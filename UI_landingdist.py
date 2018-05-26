#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:53:42 2018

@author: shugo
"""

import numpy as np
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D
from scipy import fftpack, interpolate, integrate, optimize
import pandas as pd
import subprocess
from Rocket_simu import Rocket
from postprocess_plot import PostProcess_dist, JudgeInside

# User Interface for 

class TrajecSimu_UI():

    def __init__(self, csv_filename, loc='noshiro_sea'):
        # =============================================
        # This method is called when an instance is created
        #
        # input: csv_filename = file name of a csv file that contains parameter informations
        #        lco          = 'izu' or 'noshiro_sea'
        # =============================================

        # create an instance 
        # provide csv file name that contains parameters
        self.myrocket = Rocket(csv_filename)
        
        self.launch_location = loc
        


    def run_single(self):
        # =============================================
        # A method for a single trajectory simulation and post-processing
        # =============================================
        
        # run a main computation
        self.myrocket.run()
            
        # post-process
        self.myrocket.postprocess('all')
            
        return None
    
    def run_loop(self, n_winddirec = 16, max_windspeed = 8., windspeed_step = 1.):
        # =============================================
        # A method for running loop to get landing distribution
        #
        #        wind_direction_array = np.array that contains wind directions to be computed
        #        wind_speed_array     = np.array that contains wind speedsto be computed
        # =============================================
        
        # keep initial parachute deployment definition
        t_para_delay_original = self.myrocket.params_dict['t_para_delay']
        t_deploy_original = self.myrocket.params_dict['t_deploy']
        
        wind_direction_array = np.linspace(0., 360., n_winddirec+1)
        wind_speed_array = np.arange(1., max_windspeed + windspeed_step, windspeed_step)
        
        # --------------------
        # initialize arrays 
        # --------------------
        n1 = len(wind_speed_array)
        n2 = len(wind_direction_array)
        self.loc_bal      = np.zeros((n1, n2, 2))    # ballistic landing location
        self.loc_para     = np.zeros((n1, n2, 2))    # parachute fall landing location
        self.max_alt      = np.zeros((n1, n2))       # max altitude
        self.max_vel      = np.zeros((n1, n2))       # max velocity
        self.max_Mach     = np.zeros((n1, n2))       # max Mach number
        self.max_Q        = np.zeros((n1, n2))       # max dynamic pressure
        self.v_launch_clear = np.zeros((n1, n2))     # lanuch clear air speed
        self.v_para_deploy  = np.zeros((n1, n2))       # parachute deploy air speed
          
        # initialize array for parameter update
        params_update = [ ['wind_speed', 0.], ['wind_direction', 0.], ['t_para_delay', 0.], ['t_deploy', 0.] ]
        
        # """
        # --------------------
        # loop
        # --------------------
        # initialize loop count
        i_speed = 0
        i_angle = 0
        # loop over wind speed
        for wind_speed in wind_speed_array:
            # overwrite wind speed
            params_update[0][1] = wind_speed
            
            # loop over wind direction
            i_angle = 0
            for wind_angle in wind_direction_array[:-1]:
                # overwrite wind speed
                params_update[1][1] = wind_angle
                
                
                # -----------------------------------
                #  landing point for ballistic fall  
                # -----------------------------------
                # overwrite parachute opening delay time to inf.
                params_update[2][1] = 1.e7
                params_update[3][1] = 1.e7
                
                # overwrite parameters
                self.myrocket.overwrite_dataframe(params_update)
                # run a single trajectory simulation
                self.myrocket.run()    
                # post-process 
                self.myrocket.postprocess('maxval')
                # record landing location
                self.record_loop_result('bal', i_speed,i_angle)
                
                
                #
                # ---------------------------------
                # landing point for parachute fall
                # ---------------------------------
                # overwrite parachute opening delay time to 1s.
                params_update[2][1] = t_para_delay_original
                params_update[3][1] = t_deploy_original
                
                # overwrite parameters
                self.myrocket.overwrite_dataframe(params_update)
                # run main computation
                self.myrocket.run()
                # post-process and get landing location
                self.myrocket.postprocess('maxval')
                # record results
                self.record_loop_result('para', i_speed,i_angle)
                # record other flight detail 
                self.record_loop_result('vals', i_speed,i_angle)
                #
                
                # loop count
                i_angle += 1
            #END FOR
            
            # loop count
            i_speed += 1
        # END FOR
        
        # close wind direction loop
        self.loc_bal[:,-1,:] = self.loc_bal[:,0,:]
        self.loc_para[:,-1,:] = self.loc_para[:,0,:]
        
        # """
        
        """
        tmpx = np.array([ [0., 100., 200., 100. ,0],  [0., 200., 400., 200. ,0], [0., 50., 100., 50. ,0] ] )
        tmpy = np.array([ [0., 100., 0., -100. ,0],   [0., 200., 0., -200. ,0],  [0., 50., 0., -50. ,0] ])
        self.loc_bal = np.dstack( (tmpx, tmpy) )
        self.loc_para = np.dstack( (tmpx, tmpy) )
        """
        
        # -------------------------------
        #  plot
        # -------------------------------
        mypost = PostProcess_dist(self.launch_location)
        elev_angle = self.myrocket.trajectory.elev_angle  # launcher elev angle        
        mypost.plot_sct(self.loc_bal,  wind_speed_array, elev_angle, 'Ballistic')
        mypost.plot_sct(self.loc_para, wind_speed_array, elev_angle, 'Parachute')
        
        # -------------------------------
        # judge ballistic landing points 
        # -------------------------------
        tmp_centers = np.array([mypost.xy_rail, mypost.xy_switch, mypost.xy_tent])
        # create instance
        autojudge_bal = JudgeInside(mypost.xy_range, tmp_centers)
        # initialize array for result
        self.res_bal = np.ones( (self.loc_bal.shape[0], self.loc_bal.shape[1]-1) ) * False  
        # loop over wind speed
        for j in range(self.loc_bal.shape[0]):
            # loop over wind direction
            for i in range(self.loc_bal.shape[1]-1):
                if autojudge_bal.judge_inside(self.loc_bal[j, i, :]):
                    #if true, Go for launch
                    self.res_bal[j][i] = True
                # END IF 
            # END IF
        # END IF
        
        # -------------------------------
        # judge parachute landing points 
        # -------------------------------
        # create instance
        autojudge_para = JudgeInside(mypost.xy_range)
        # initialize array for result
        self.res_para = np.ones( (self.loc_para.shape[0], self.loc_para.shape[1]-1) ) * False  
        # loop over wind speed
        for j in range(self.loc_para.shape[0]):
            # loop over wind direction
            for i in range(self.loc_para.shape[1]-1):
                if autojudge_para.judge_inside(self.loc_para[j, i, :]):
                    #if true, Go for launch
                    self.res_para[j][i] = True
                # END IF 
            # END IF
        # END IF              
        
        # ------------------------
        # judge both at once
        # ------------------------
        self.list_res_both = [ ['NoGo' for i in range(self.loc_bal.shape[1]-1)] for j in range(self.loc_bal.shape[0])] 
        # loop over wind speed
        for j in range(self.loc_para.shape[0]):
            # loop over wind direction
            for i in range(self.loc_para.shape[1]-1):
                # Go if both cases are go
                if np.logical_and(self.res_bal[j][i], self.res_para[j][i]):
                    self.list_res_both[j][i] = elev_angle
                    
                # END IF 
            # END IF 
        # END IF
        
        # ------------------------------
        # save information to excel file
        # ------------------------------
        self.output_loop_result(wind_speed_array, wind_direction_array)
        
        return None
    
    
    def record_loop_result(self, record_type, i_speed, i_angle):
        # =============================================
        # A method for getting and recording flight result 
        #
        # input: record type = 'bal' or 'para' or 'vals'
        #        i_speed = index for wind speed
        #        i_angle = index for wind direction
        # =============================================
        
        if record_type == 'bal':
            # record ballistic landing location
            self.loc_bal[i_speed,i_angle,:] = self.myrocket.res['landing_loc']
            
        elif record_type == 'para':
            # record parachute fall landing location
            self.loc_para[i_speed,i_angle,:] = self.myrocket.res['landing_loc']
            
        elif record_type == 'vals':
            # record other flight information
            tmp_dict1 = self.myrocket.res['flight_detail']
            self.max_alt[i_speed,i_angle]        = tmp_dict1['max_altitude'][0]        # max altitude
            self.max_vel[i_speed,i_angle]        = tmp_dict1['max_speed'][0]           # max velocity
            self.max_Mach[i_speed,i_angle]       = tmp_dict1['max_Mach'][0]            # max Mach number
            tmp_dict2 = tmp_dict1['max_Q_all']
            self.max_Q[i_speed,i_angle]          = tmp_dict2['max_Q']               # max dynamic pressure
            self.v_launch_clear[i_speed,i_angle] = tmp_dict1['v_launch_clear']      # lanuch clear air speed
            self.v_para_deploy[i_speed,i_angle]  = tmp_dict1['v_para_deploy']       # parachute deploy air speed
                  
        return None
    
    def output_loop_result(self,wind_speed_array, wind_direction_array):
        # convert np.arrays into pandas dataframes
        ws = wind_speed_array
        wd = wind_direction_array
        loc_x_bal      = pd.DataFrame(self.loc_bal[:,:,0], index = ws, columns = wd)
        loc_y_bal      = pd.DataFrame(self.loc_bal[:,:,1], index = ws, columns = wd)
        loc_x_para     = pd.DataFrame(self.loc_para[:,:,0], index = ws, columns = wd)
        loc_y_para     = pd.DataFrame(self.loc_para[:,:,1], index = ws, columns = wd)
        max_alt        = pd.DataFrame(self.max_alt[:,:-1], index = ws, columns = wd[:-1])
        max_vel        = pd.DataFrame(self.max_vel[:,:-1], index = ws, columns = wd[:-1])
        max_Mach       = pd.DataFrame(self.max_Mach[:,:-1], index = ws, columns = wd[:-1])
        max_Q          = pd.DataFrame(self.max_Q[:,:-1], index = ws, columns = wd[:-1])
        v_launch_clear = pd.DataFrame(self.v_launch_clear[:,:-1], index = ws, columns = wd[:-1])
        v_para_deploy  = pd.DataFrame(self.v_para_deploy[:,:-1], index = ws, columns = wd[:-1])
        bal_judge      = pd.DataFrame(self.res_bal, index = ws, columns = wd[:-1])
        para_judge     = pd.DataFrame(self.res_para, index = ws, columns = wd[:-1])
        judge_both     = pd.DataFrame(self.list_res_both, index = ws, columns = wd[:-1])
        
        
        # define output file name
        output_name = 'results/output_' + str(int(self.myrocket.trajectory.elev_angle)) + 'deg.xlsx'
        excel_file = pd.ExcelWriter(output_name)
 
        # write dataframe with sheet name
        judge_both.to_excel(excel_file, 'Go-NoGo判定 ')
        max_alt.to_excel(excel_file, '最高高度 ')
        max_vel.to_excel(excel_file, '最高速度 ')
        max_Mach.to_excel(excel_file, '最大マッハ数')
        v_launch_clear.to_excel(excel_file, 'ランチクリア速度 ')
        v_para_deploy.to_excel(excel_file, 'パラ展開時速度 ')
        max_Q.to_excel(excel_file, '最大動圧 ')
        loc_x_bal.to_excel(excel_file, '弾道 x ')
        loc_y_bal.to_excel(excel_file, '弾道 y ')
        loc_x_para.to_excel(excel_file, 'パラ x ')
        loc_y_para.to_excel(excel_file, 'パラ y ')
        bal_judge.to_excel(excel_file, '弾道判定 ')
        para_judge.to_excel(excel_file, 'パラ判定 ')
        
        # save excel file
        try:    
            excel_file.save()
        except:
            subprocess.run(['mkdir', 'results'])
            excel_file.save()
        
        return None
    
    def run_rapid_design(self, m_dry, obj_type='altitude', obj_value=10000):
        # =============================================
        # A method for running simulations for rapid design toolbox 
        #
        # INPUT: m_dry         = target dry mass
        #        obj_type      = design objective type. 'altitude' or 'Mach'
        #        obj_value     = design objective value. [m] for 'altitude'.
        # =============================================
        
        print('==========================')
        print(' Rapid Design Toolbox')
        print(' m_dry: ', m_dry)
        print(' target type: ', obj_type)
        print(' target value:', obj_value)
        print('==========================')
        
        #
        params_df = self.params_df
        params_df.loc[params_df.parameter == 'm_dry', 'value'] = m_dry
        
        # thrust magnification factor setting
        thrust_mag_array = np.linspace(2., 4., 9)
        time_mag_array = np.zeros(len(thrust_mag_array))
        
        # initialize resulting array
        mass_all = np.zeros(len(thrust_mag_array))
        max_alt_all = np.zeros(len(thrust_mag_array))
        max_mach_all = np.zeros(len(thrust_mag_array))
        
        # loop over thrust_mag_factor
        for i in range(len(thrust_mag_array)):
            # overwrite thrust_mag_factor
            params_df.loc[params_df.parameter == 'thrust_mag_factor', 'value'] = thrust_mag_array[i]
            
            # ------------------------------------
            def obj_all(time_mag_factor):
                # define a function that for a given time_mag_array, compute trajectory 
                # then return (simulation result - objective value) 
                
                # variable echo
                print('[thrust, time] =', thrust_mag_array[i], time_mag_factor)
                # --------------------------
                # variable setup: 
                # --------------------------
                # overwrite time_mag_factor
                params_df.loc[params_df.parameter == 'time_mag_factor', 'value'] = time_mag_factor
                # estimate mass of propellant from mag_factors
                It_mag_factor = time_mag_factor * thrust_mag_array[i]
                m_prop_per10000Ns =7.   # mass of propellant[kg] for 10000N.s
                """
                reference raw data of the UM Rocket team
                fuel:     2.4kg 
                oxidizer: 13.75 kg
                total impulse: 32100 N.s
                   -> 5kg propellant / 10000N.s 
                   85% of prop mass = ox, 15% = fuel
                """
                m_prop = m_prop_per10000Ns * It_mag_factor # mass of propellant for current design
                # overwrite m_propellant
                params_df.loc[params_df.parameter == 'm_prop', 'value'] = m_prop
                
                # --------------------------
                # trajectory computation
                # --------------------------
                # compute trajectory
                self.run()
                # post-process and get objective value
                self.postprocess('maxval')
                # obtain result
                max_alt = self.res['flight_detail']['max_altitude']
                max_mach = self.res['flight_detail']['max_Mach']
                if obj_type == 'altitude':
                    # --- design objective: altitude ---
                    result = max_alt[0]
                
                elif obj_type == 'Mach':
                    # --- design objective: Mach ---
                    result = max_mach[0]
                    
                # return residual
                return result-obj_value, max_alt, max_mach, m_prop
            
            def obj_residual(time_mag_factor):
                tmp,_,_,_ = obj_all(time_mag_factor)
                return tmp
            # ------------------------------------
            
            
            # solve " objective=0 "
            # time_mag_sol = optimize.fsolve(obj_residual, 1.5, xtol=1.e-06)
            time_mag_sol = optimize.brentq(obj_residual, 0.1, 5., xtol=1.e-06)
            time_mag_array[i] = time_mag_sol
            
            # re-do optimal computation for recording
            _, max_alt, max_mach,m_prop = obj_all(time_mag_sol)
            # record
            mass_all[i] = m_dry + m_prop
            max_alt_all[i] = max_alt[0]
            max_mach_all[i] = max_mach[0]
            
            print('---------------------------------------')
            print('mag. factor for [thrust, time] =', thrust_mag_array[i], time_mag_sol)
            print('total impulse: ', 10000. * thrust_mag_array[i] * time_mag_sol)
            print('max mach: ', max_mach)
            print('max alt: ', max_alt)
            print('liftoff mass: ', mass_all[i])
            print('---------------------------------------')
            print(' ')
            
            # solve for 
        # END FOR
        
        # plot 
        print('==========================')
        print(' Rapid Design Toolbox (recap)')
        print(' m_dry: ', m_dry)
        print(' target type: ', obj_type)
        print(' target value:', obj_value)
        print('==========================')
        time_dim = time_mag_array*10
        thrust_dim = thrust_mag_array*1000
        impulse_dim = time_dim*thrust_dim
        # ---- plot 1 ------
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(time_dim, thrust_dim, label='thrust', color='r')  # burn time vs. averaged thrust
        plt.legend()
        ax2 = ax1.twinx() 
        ax2.plot(time_dim, impulse_dim, label='It', color='b') # burn time vs. total impulse
        
        ax1.set_xlabel('burn time[s]')
        ax1.set_ylabel('averaged thrust[N]')
        ax1.grid(True)
        ax2.set_ylabel('total impulse')
        plt.legend()
        target_str = 'm_dry='+str(m_dry)+'[kg], target '+obj_type+'='+str(obj_value)
        plt.title('Required engine property: ' + target_str)
        plt.savefig('engine.eps')
        # plt.show()

        # ---- plot 2 ------
        plt.figure()
        plt.plot(time_dim, mass_all)
        plt.xlabel('burn time[s]')
        plt.ylabel('lift off mass')
        plt.grid()
        plt.title('Lift off mass: ' + target_str)
        plt.savefig('wetmass.eps')
        
        # ---- plot 3 ------
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(time_dim, max_alt_all, label='altitude', color='r')  # burn time vs. averaged thrust
        plt.legend()
        ax2 = ax1.twinx() 
        ax2.plot(time_dim, max_mach_all, label='Mach', color='b') # burn time vs. total impulse
        plt.legend()
        
        ax1.set_xlabel('burn time[s]')
        ax1.set_ylabel('max altitude [m]')
        ax1.grid(True)
        ax2.set_ylabel('max mach')
        plt.legend()
        plt.title('Flight detail: '+ target_str)
        plt.savefig('maxval.eps')
        # plt.show()
        
        # create a new directory 
        dirname = 'RDT_' + target_str
        subprocess.run(['mkdir', dirname])
        # move SU2 output files to the directory
        subprocess.run(['mv', 'maxval.eps', 'wetmass.eps', 'engine.eps', dirname])
        
        
        return None
        
    
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
        #plt.plot(0,0,color='r',marker='*',markersize=12)
        #plt.plot(x_circle, y_circle,color='r',lw=5)
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
        # plt.plot(0,0,color='r',marker='*',markersize=12)
        # plt.plot(x_circle, y_circle,color='r',lw=5)
        # plt.show
        
        return None
    
        