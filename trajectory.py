#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 23:06:17 2017

@author: shugo
"""

import numpy as np
from scipy.integrate import ode, odeint
from scipy import fftpack, interpolate, integrate
import pandas as pd
import quaternion 
from Rocket_simu import Rocket_simu
import matplotlib.pyplot as plt

class trajec_main(Rocket_simu):
    """
    ====================================================
    This class is a sub-class for main ODE trajecotry computation
    ====================================================
    """
    
    """
    ----------------------------------------------------
        Method for initial setup    
    ----------------------------------------------------
    """
    def __init__(self, params_df):
        # =============================================
        # this method is called when instance is created. Parameter setup is also done by this method.
        #
        # INPUT: params_df = parameters contains all parameters needed
        # =============================================
        # setup parameters in the instance by calling superclasses method
        super().__init__()
        self.overwrite_parameters(params_df)   

        return None     

        
    """
    ----------------------------------------------------
        Method for main ODE integration  
    ----------------------------------------------------
    """
    
    def ODE_main(self):
        # =============================================
        # this method runs ODE integration given all the parameters
        #     
        # OUTPUT: solution: shape (len(t), len(y0))
        #                   containing the value of y for each desired time in t, with the initial value y0 in the first row.
        # =============================================
        
        # ---------------------------------------------
        #      Initialization
        # ---------------------------------------------
        # set initial value
        t0 = 0.              # start time
        u0 = self.SetupICs() # initiate state vector

        # number of iteration
        self.N_iter = 0
            
        # initialize a list for backup
        # backup = [time,flag,u]
        self.backup = [np.r_[t0,0,u0]]

        # landing time initialization: 
        self.landing_time = self.t_max
            
        # set flag = 1 (on launch rail)
        self.flag = 1
    
        """
        print('----------------------------')
        print('  We are GO for launch')
        print('----------------------------')
        print(' ')
        """
        
        if self.integ == 'lsoda_odeint':
            # ---------------------------------------------
            #      use scipy.odeint
            # ---------------------------------------------
            # create time array
            
            self.t = np.arange(t0,self.t_max,self.dt)
            try:
                # ---  run trajectory computation   ---
                self.solution = odeint(self.f_main, u0, self.t)
            except:
                print('ode integration failed!')
                pass
            
        else:
            # ---------------------------------------------
            #      use scipy.ode
            # ---------------------------------------------            
            # create ODE integration instance of scipy.integrate.ode
            r = ode(self.f_main).set_integrator(self.integ)
            # set initial value
            r.set_initial_value(u0, t0)
            
            # ---  run trajectory computation   ---
            # loop until landing (or maximum time) 
            while r.successful() and r.t < self.t_max and self.flag<5:
                # call ode integration
                r.integrate(r.t+self.dt)
                #END IF
            #END WHILE
        #END IF
        
        """        
        print('----------------------------')
        print('   Completed trajectory computation ')
        print('----------------------------')
        print(' ')
        print(' ')
        """ 

        return None   
        

    """
    ----------------------------------------------------
        Other methods for simulation      
    ----------------------------------------------------
    """
    
    def SetupICs(self):
        # =======================================
        # this method sets up initial conditions
        # and returns initial state vector u0: 13*1 array
        #  u0[0:3]  = (1*3) vector: translation xyz      referring to local fixed coord.
        #  u0[3:6]  = (1*3) vector: velocity uvw         referring to body coord. 
        #  u0[6:10] = quaternion:   attitude             convert from local to body
        #  u0[10:]  = (1*3) vector: abgular velocity pqr referring to body coord. 
        # =======================================
        
        # initial location
        x0 = np.zeros(3)
         
        # initial velocity
        v0 = np.zeros(3)
        
        # initial angular velocity
        omega0 = np.zeros(3)
        
        # compute initial quaternion 
        # convert body coord. to local fixed coord.
        angle1 = (90.-self.azimuth)/2. * (np.pi/180.)
        qz = np.quaternion(np.cos(angle1),0.,0.,np.sin(angle1))  # around z
        angle2 = -self.elev_angle/2. * (np.pi/180.)
        qy = np.quaternion(np.cos(angle2),0.,np.sin(angle2),0.)  # around y
        # convert local -> rotate around z -> rotate around y -> body
        q0 = qz * qy
        
        # convert quaternion to float array
        q01 = quaternion.as_float_array(q0)
        
        # initial state "vector"
        u0 = np.r_[x0,v0,q01,omega0]
    
        return u0
        

    def f_main(self,u,t):
        # =======================================
        # this method is the RHS function of ODE
        #
        # INPUT: t    = time (scalar) 
        #        u    = state vector u (13*1)
        #
        # NOTE:  self.flag = 0 before ignition
        #                    1 : launch rail
        #                    2 : thrusted flight
        #                    3 : inertial flight
        #                    4 : parachute deployed
        #                    5 : landed
        # =======================================
        
        # if the rocket has already landed, return 0
        if self.flag == 5:
            return u*0.
            
        # swap input when we use scipy.integrate.ode
        if self.integ != 'lsoda_odeint':
            tmp = u
            u = t
            t = tmp
        #END IF
    
        # count the number of function call
        self.N_iter += 1
        
        # backuping 
        if np.mod(self.N_iter, self.N_record) == 0:
            self.add_backup(t,u)
        #END IF
        
        # --------------------------
        #   extract vectors
        # --------------------------
        # x =     translation         :expressed in local-fixed coordinate
        # v =     velocity            :expressed in body coordinate
        # q =     atitude quaternion  :conversion from local-fixed to body
        # omega = angular velocity    :expressed in body coordinate
        x,v,q,omega = self.state2vecs_quat(u)  
         
            
        # ----------------------------
        #    Direction Cosine Matrix for input q
        # ----------------------------
        # Tbl = transform from local(fixed) coordinate to body coord.
        #     note: as_rotation_matrix is for vector rotation
        #         -> for coordinate rotation, input conj(q)
        Tbl = quaternion.as_rotation_matrix(np.conj(q))
        
        # ----------------------------
        #   flight mode classification
        # ----------------------------
        if self.flag==1 and x[2] > self.rail_height:
            # detect launch clear
            print('----------------------------')
            print('  Launcher-clear at t = ',t,'[s]')
            print('  ground speed: ',np.linalg.norm(v),'[m/s]')
            print('----------------------------')
            print(' ')
            
            # record history
            self.add_backup(t,u)
            # switch into thrusted flight
            self.flag = 2   
            
        elif (self.flag==1 or self.flag==2) and t >= self.t_MECO:
            # detect MECO
            print('----------------------------')
            print('  MECO at t = ',t,'[s]')
            print('  current altitude: ',x[2],'[m]')
            print('  ground speed:    ',np.linalg.norm(v),'[m/s]')
            print('----------------------------')
            print(' ')
            
            # record history
            self.add_backup(t,u)
            # switch into inertial flight
            self.flag = 3   
            
        elif self.flag==3 and t >= self.t_deploy:
            # detect parachute deployment
            # air speed
            air_speed = np.linalg.norm( -v + np.dot( Tbl,self.wind(x[2]) ) ) 
            print('----------------------------')
            print('  Parachute deployed at t = ', t, '[s]')
            print('  current altitude: ', x[2], '[m]')
            print('  ground speed:    ', np.linalg.norm(v), '[m/s]')
            print('  true air speed: ', air_speed, '[m/s]')
            print('----------------------------')
            print(' ')
            
            # record history
            self.add_backup(t,u)
            # switch into parachute fall
            self.flag = 4  
            
            # stop rotation
            omega = np.zeros(3)  
            
        elif self.flag > 1 and self.flag < 5 and x[2] < 0. :
            # detect landing
            print('----------------------------')
            print('  Landing at t = ',t,'[s]')
            print('  landing speed: ',np.linalg.norm(v),'[m/s]')
            print('          location x = ',x[0],'[m]')
            print('                   y = ',x[1],'[m]')
            print('----------------------------')
            print(' ')
            # record landing time
            self.landing_time = t
            
            # record history
            self.add_backup(t,u)
            # flag: landed
            self.flag = 5 
            # quit integration
            if self.integ == 'lsoda_odeint':
                return u*0.                
        #END IF 
        
        
        # ----------------------------
        #    1. Translation
        # ----------------------------
        # translation time rate = velocity
        # convert v from body coord. to fixed coord.
        dx_dt = np.dot(Tbl.T,v)
        
        # call mass properties 
        mass,MOI,d_dt_MOI,CG = self.mass_MOI(t)
        
        # ----------------------------
        #    2. Velocity
        # ----------------------------
        # velocity time rate = mass * force
        # force = rotation effect + aero + thrust + gravity 
        #      where  aero   = aero(u):   function of state u
        #             thrust = thrust(t): function of time t
        #             mass   = mass(t):   function of time t   
    
        # call aerodynamic force/moment
        aeroF,aeroM = self.aero(x,v,omega,Tbl,CG)     
        
        grav = np.array([0.,0.,-9.81])  # gravitational accel. in fixed coord.
        
        # set external force depending on the state
        if self.flag == 1:
            # on launch rail -> du/dx only. Consider rail-rocket friction force 
            # total acceleration 
            dv_dt = -np.cross(omega,v) + np.dot(Tbl,grav) + (aeroF + self.thrust(t) + self.friction() ) / mass
            # cancell out y,z
            dv_dt = np.array([dv_dt[0],0.,0.])
            
            if dv_dt[0] < 0:
                # when du/dx is negative (i.e. weight is greater that thrust)
                # -> rocket is hold up. return zero acceleration
                dv_dt = np.zeros(3)
            #END
            
        elif self.flag == 2:
            # thrust ON
            # total acceleration 
            dv_dt = -np.cross(omega,v) + np.dot(Tbl,grav) + (aeroF + self.thrust(t)) / mass
            
        elif self.flag == 3 or self.flag == 5:
            # thrust OFF
            # total acceleration 
            dv_dt = -np.cross(omega,v) + np.dot(Tbl,grav) + aeroF / mass

        elif self.flag == 4:
            # parachute deployed
            dv_dt = np.dot(Tbl,grav) + self.parachute_F(x,v,Tbl) / mass
        #END IF
        
        
        # ----------------------------
        #    3. Atitude 
        # ----------------------------
        # represented in quaternion which rotates from fixed to body
        # quaternion differential equation
        
        # stop rotation when parachute ihas deployed
        if self.flag == 4:
            omega *= 0.
        #END IF
        
        # convert omega to quaternion
        q_omega = np.r_[[0.], omega]
        q_omega2 = quaternion.as_quat_array(q_omega)
        
        # dq/dt to be returned
        dq_dt2 = 0.5 * q * q_omega2
        
        # convert back to float array
        dq_dt = quaternion.as_float_array(dq_dt2)
        
        
        # ----------------------------
        #    4. Angular velocity
        # ----------------------------
        # angular velocity time rate: comes from Euler equation
        #      where  MOI       = MOI(t):       function of time
        #             d_dt(MOI) = d_dt(MOI)(t): function of time
        #             aeroM     = aeroM(u):     function of state 
        #             ThrustM   = ThrustM(t)=0: function of time
        
        if self.flag == 1 or self.flag == 4:
            # on launch rail /parachute deployed -> no angular velocity change
            domega_dt = np.zeros(3)

        else:
            # MOI1 = MOI(t)           # moment of inertia     
            # d_dt_MOI1 = d_dt_MOI(t) # time derivative of MOI
            # aeroM1 = aeroM(u)       # aerodynamic moment
            
            # Euler eqn of rotation
            #tmp1 = 1./MOI[0] * ( (MOI[1]-MOI[2])*omega[1]*omega[2] - d_dt_MOI[0]*omega[0] + aeroM[0]) 
            #tmp2 = 1./MOI[1] * ( (MOI[2]-MOI[0])*omega[0]*omega[2] - d_dt_MOI[1]*omega[1] + aeroM[1]) 
            #tmp3 = 1./MOI[2] * ( (MOI[0]-MOI[1])*omega[0]*omega[1] - d_dt_MOI[2]*omega[2] + aeroM[2]) 
            #domega_dt = np.array([tmp1,tmp2,tmp3])
            domega_dt = (-np.cross(omega,MOI*omega) - d_dt_MOI*omega + aeroM) / MOI
        # END IF

        # ----------------------------
        #    Set variables back in the state vector form
        # ----------------------------
        du_dt = np.r_[dx_dt,dv_dt,dq_dt,domega_dt]

        # ----------------------------
        #      apogee detection 
        # ----------------------------
        if dx_dt[2] < 0.:
            self.apogee_count += 1
            
            if self.apogee_count >= 10:
                # detect apogee and set parachute deployment time
                # parachute is deployed at t = t_deploy, 
                # which is either "t_deploy" [s] after ignition or
                # "t_para_delay" [s] after apogee detection
                
                self.t_deploy = min( self.t_deploy, t + self.t_para_delay )
                self.apogee_count = -1.e10
            #END IF
        #END IF
        
        return du_dt
        
        
    def mass_MOI(self,t):
        # =======================================
        # this method returns mass properties of rocket
        # 
        # INPUT:  t = time
        # OUTPUT: mass = total mass of rocket
        #         MOI = total moment of inertia wrt CG
        #         d_dt_MOI = MOI time rate
        #         CG = center of gravity location from the nose tip
        # =======================================

        if t >= self.t_MECO:
            # mass
            mass = self.m_dry
            # moment of inertia
            MOI = self.MOI_dry
            # time rate of MOI
            d_dt_MOI = np.zeros(3)
            # total CG location
            CG = self.CG_dry
            
        else:
            # mass
            mass = self.m_dry + (self.t_MECO-t)/self.t_MECO * self.m_fuel
    
            # fuel comsumption rate (assumed linear)
            r = (self.t_MECO-t)/self.t_MECO 
            # total CG location
            CG = (self.CG_dry*self.m_dry + self.CG_fuel*r*self.m_fuel) / (self.m_dry + r*self.m_fuel)
            # total MOI using parallel axis theorem
            tmp = np.array([0.,1.,1.])
            MOI = self.MOI_dry + tmp*self.m_dry*(CG-self.CG_dry)**2. + r*self.MOI_fuel + tmp*(CG-self.CG_fuel)*(r*self.m_fuel)**2.
    
            # finite differencing
            h = 1.E-5
            r2 = (self.t_MECO-t+h)/self.t_MECO 
            # total CG location
            CG2 = (self.CG_dry*self.m_dry + self.CG_fuel*r2*self.m_fuel) / (self.m_dry + r2*self.m_fuel)
            # total MOI using parallel axis theorem
            tmp = np.array([0.,1.,1.])
            MOI2 = self.MOI_dry + tmp*self.m_dry*(CG2-self.CG_dry)**2. + r2*self.MOI_fuel + tmp*(CG2-self.CG_fuel)*(r2*self.m_fuel)**2.
    
            # finite differencing
            r3 = (self.t_MECO-t-h)/self.t_MECO 
            # total CG location
            CG3 = (self.CG_dry*self.m_dry + self.CG_fuel*r3*self.m_fuel) / (self.m_dry + r3*self.m_fuel)
            # total MOI using parallel axis theorem
            tmp = np.array([0.,1.,1.])
            MOI3 = self.MOI_dry + tmp*self.m_dry*(CG3-self.CG_dry)**2. + r3*self.MOI_fuel + tmp*(CG3-self.CG_fuel)*(r3*self.m_fuel)**2.
    
            d_dt_MOI = (MOI2 - MOI3) / (2*h)
        #END IF
        
        return mass, MOI, d_dt_MOI, CG
                
    
    def thrust(self,t):
        # =======================================
        # returns thrust force 
        # 
        # INPUT:  t = time
        # OUTPUT: T = thrust vector in body coord.
        # =======================================
        
        # rectangle thrust
        # T = np.array([self.thrustforce,0.,0.])
        
        # use interp1d
        thrust = self.thrust_function(t)
        T = np.array([thrust,0.,0.])
        
        return T
        
        
    def friction(self):
        # =======================================
        # returns friction force (rail-rocket contact) 
        # 
        # INPUT:  t = time
        # OUTPUT: fric = friction force vector in body coord.
        # =======================================
        friction = np.zeros(3)
        return friction
            
        
            
    def aero(self,x,v,omega,Tbl,CG):
        # =======================================
        # returns aerodynamic force and moment 
        #
        # INPUT:  x     = translation         :in fixed coordinate
        #         v     = velocity            :in body coordinate
        #         omega = angular velocity    :in body coordinate
        #         Tbl   = transform matrix from local(fixed) coordinate to body coord.
        #         CG    = current CG location
        # OUTOUT: force_all  = aerodynamic force vector in body coord.
        #         moment_all = aerodynamic moment vector wrt. CG in body coord.
        # =======================================
        
        # --------------------------------------------------
        #   Compute air velocity, angle-of-attack, roll-angle
        # --------------------------------------------------
        # air velocity = -rocket_velocity + wind_velocity
        #    NOTE: wind(x) is wind velocity in local coord. need conversion to body coord.
        v_air = -v + np.dot( Tbl,self.wind(x[2]) )   
        air_speed = np.linalg.norm(v_air) # air speed (positive scalar)
    
        # total angle-of-attack
        if air_speed == 0:
            alpha = 0.
        else:
            alpha = np.arccos( -v_air[0]/air_speed )
            #if v_air[0] > 0:
            #    alpha = -alpha
            #END IF
        #END IF
        
        #if np.isnan(alpha):
        #    alpha = 0.
        
        # roll-angle
        if v_air[2]==0:
            # if w = 0, atan(v/w) is not defined 
            if -v_air[1] > 0:
                phi = np.pi /2.
            else:
                phi = -np.pi /2.    
        else:
            phi = np.arctan( -v_air[1]/ -v_air[2] )
            if -v_air[2]<0:
                # if w<0, that is in 3rd or 4th quadrant in yz-plane, add pi to phi
                #  so that sign of sin(phi), cos(phi) will be correctly calculated.
                phi += np.pi
            #END IF
        #END IF
                
            
        #if np.isnan(phi):
        #    phi = np.arctan( v_air[1]+0.0001/v_air[2]+0.0001 )
            
        
        # ----------------------------------
        #   force on body excluding fins
        # ----------------------------------
        # air property at the altitude
        _,_,rho,a = self.standard_air(x[2])
        
        #print('h',x[2],'T',T,'p',p,'rho',rho)
        #print('h',x[2],'rho',rho)
        
        # Mach number
        Mach = air_speed/a
    
        # drag/lift coefficient of body
        Cd,Cl = self.rocket_coeff_nofins(Mach,alpha)
        
        # convert coefficient to body coord.
        cosa = np.cos(alpha)
        sina = np.sin(alpha)
        
        #C1b = -Cl*sina + Cd*cosa
        #C2b = -(Cl*cosa + Cd*sina)*np.sin(phi)
        #C3b = (Cl*cosa + Cd*sina)*np.cos(phi)
        C = np.array([ (-Cl*sina + Cd*cosa), \
                      (Cl*cosa + Cd*sina)*np.sin(phi), \
                      (Cl*cosa + Cd*sina)*np.cos(phi)])    # need check here
        
        # force on CP of body
        force_all = 0.5 * rho * air_speed**2. * self.X_area * (-C)
        
        # moment generated by body wrt CG
        moment_all = np.cross( np.array([CG-self.CP_body,0.,0.]) , force_all )
        
        # print('Cd,Cl', Cd,Cl, 'alpha', alpha*180/np.pi)
        #print('')
        
        
        if self.aero_fin_mode == 'indiv':
            # ----------------------------------
            #   compute force on fins individually
            # ----------------------------------
            # force and moment (wrt fin Leading edge)
            force_fin, moment_fin_tmp = self.fin_aero(v_air,omega[0],rho)
            # moment generated by fin wrt CG
            moment_fin = moment_fin_tmp + np.cross( np.array([CG-self.LE_fins,0.,0.]) , force_fin )
            
            # total force
            force_all += force_fin
            # total moment
            moment_all += moment_fin
        # END IF 
        
        return force_all, moment_all
        
    def fin_aero(self,v_air,omega_x,rho):
        # ==============================================
        # return aerodynamic force and moment generated by fins
        # reference point of moment = [Leading Edge * root] point
        # ==============================================
        
        if all(v_air==0):
            v_air = np.array([0.0001,0.,0.])
            
        # each component of air velocity 
        u = v_air[0]  
        v = v_air[1]
        w = v_air[2]
        
        # --------------------
        #    fin1: z=0, y>0
        # --------------------
        # net air speed on xz plane. (vertical to the fin) Take roll-spin into account.
        U1 = np.sqrt(u**2. + ( w - self.r_arm*omega_x )**2.)  # array as func. of r_arm
                    
        # net angle-of-attack (air flow, roll-spin, fin attachment angle are considered)
        alpha1 = np.arcsin(( w - self.r_arm*omega_x )/U1) + self.alpha_fins
    
        # lift force distribution
        lift1 = np.pi * rho * U1**2. * self.fin_len * np.sin(alpha1) * self.dy_fin  
        # convert into body coordinate
        force1x = - lift1 * np.sin(alpha1)
        force1z = lift1 * np.cos(alpha1)
        
        # roll moment distribution
        rollmoment1 = force1z * self.r_arm
        # pitch moment distribution
        pitchmoment1 = force1z * self.xFP
        
        # total force of fin1
        f1 = np.array([np.sum(force1x),0.,np.sum(force1z)])
        
        # total moment of fin1
        m1 = np.array([np.sum(rollmoment1),np.sum(pitchmoment1),0.])
        
        
        # --------------------
        #    fin2: y=0, z>0
        # --------------------
        # net air speed on xz plane. (vertical to the fin) Take roll-spin into account.
        U2 = np.sqrt(u**2. + ( v + self.r_arm*omega_x )**2.)  # array as func. of r_arm
        
        # net angle-of-attack (air flow, roll-spin, fin attachment angle are considered)
        alpha2 = np.arcsin(( v + self.r_arm*omega_x )/U2) - self.alpha_fins
    
        # lift force distribution
        lift2 = np.pi * rho * U2**2. * self.fin_len * np.sin(alpha2) * self.dy_fin  
        # convert into body coordinate
        force2x = - lift2 * np.sin(alpha2)
        force2y = lift2 * np.cos(alpha2)
        
        # roll moment distribution
        rollmoment2 = - force2y * self.r_arm  # positive y-force generates negative roll moment
        # pitch moment distribution
        yawmoment2 = - force2y * self.xFP  # positive y-force generates negative yaw moment
        
        # total force of fin1
        f2 = np.array([np.sum(force2x),np.sum(force2y),0.])
        
        # total moment of fin1
        m2 = np.array([np.sum(rollmoment2),0.,np.sum(yawmoment2)])
        
    
        # --------------------
        #    fin3: z=0, y<0
        # --------------------
        # net air speed on xz plane. (vertical to the fin) Take roll-spin into account.
        U3 = np.sqrt(u**2. + ( w + self.r_arm*omega_x )**2.)  # array as func. of r_arm
        
        # net angle-of-attack (air flow, roll-spin, fin attachment angle are considered)
        alpha3 = np.arcsin(( w + self.r_arm*omega_x )/U3) - self.alpha_fins
    
        # lift force distribution
        lift3 = np.pi * rho * U3**2. * self.fin_len * np.sin(alpha3) * self.dy_fin  
        # convert into body coordinate
        force3x = - lift3 * np.sin(alpha3)
        force3z = lift3 * np.cos(alpha3)
        
        # roll moment distribution
        rollmoment3 = -force3z * self.r_arm  # positive z-force generates negative moment
        # pitch moment distribution
        pitchmoment3 = force3z * self.xFP
        
        # total force of fin3
        f3 = np.array([np.sum(force3x),0.,np.sum(force3z)])
        
        # total moment of fin3
        m3 = np.array([np.sum(rollmoment3),np.sum(pitchmoment3),0.])
        
        
        # --------------------
        #    fin4: y=0, z<0
        # --------------------
        # net air speed on xz plane. (vertical to the fin) Take roll-spin into account.
        U4 = np.sqrt(u**2. + ( v - self.r_arm*omega_x )**2.)  # array as func. of r_arm
        
        # net angle-of-attack (air flow, roll-spin, fin attachment angle are considered)
        alpha4 = np.arcsin(( v - self.r_arm*omega_x )/U2) + self.alpha_fins
    
        # lift force distribution
        lift4 = np.pi * rho * U4**2. * self.fin_len * np.sin(alpha4) * self.dy_fin  
        # convert into body coordinate
        force4x = - lift4 * np.sin(alpha2)
        force4y = lift4 * np.cos(alpha2)
        
        # roll moment distribution
        rollmoment4 = force4y * self.r_arm  # positive y-force generates positive roll moment
        # pitch moment distribution
        yawmoment4 = - force4y * self.xFP  # positive y-force generates negative yaw moment
        
        # total force of fin1
        f4 = np.array([np.sum(force4x),np.sum(force4y),0.])
        
        # total moment of fin1
        m4 = np.array([np.sum(rollmoment4),0.,np.sum(yawmoment4)])
        
        # ---------------------
        #   total force/moment by fins
        # ---------------------
        force_all = f1+f2+f3+f4
        moment_all = m1+m2+m3+m4
        
        return force_all, moment_all
    
    def parachute_F(self,x,v,Tbl):
        # =======================================
        # returns aerodynamic force and moment 
        #
        # INPUT:  x     = translation         :in fixed coordinate
        #         v     = velocity            :in body coordinate
        #         Tbl   = transform matrix from local(fixed) coordinate to body coord.
        # OUTOUT: parachute_drag  = parachute drag force in body coord.
        # =======================================
        
        # air velocity = -rocket_velocity + wind_velocity
        #    NOTE: wind(x) is wind velocity in local coord. need conversion to body coord.
        v_air = -v + np.dot( Tbl,self.wind(x[2]) )   

        # air property at the altitude
        _,_,rho,_ = self.standard_air(x[2])

        # parachute drag force 
        parachute_drag = 0.5 * rho * np.linalg.norm(v_air) * v_air * self.S_para * self.Cd_para

        # print('v_air', v_air, 'speed', np.linalg.norm(v_air))
        return parachute_drag

        
    def standard_air(self,h):
        # ==============================================
        # returns air property given an altitude 
        # INPUT: h = altitude [m]
        # ==============================================
        
        # gas constant
        R = 287.15  # [J/kg.K]
        # gravitational accel.
        g = 9.81  # [m/s^2]
        
        
        """
        # temperature goes down 0.0065K/m until it reaches -56.5C (216.5K)
        #                                       it is approximately 11km
        T = self.T0 - 0.0065*h # [K]
        
        
        # temperature is const at 216.5 K for alt. < 20km
        if type(T) == np.ndarray:
            T[T<216.5] = 216.5
            
        elif T < 216.5:
            T = 216.5
        
        # pressure
        p = self.p0 * (T/self.T0)**5.256  #[Pa]
        """
        
        
        if h <= 11.*10**3:
            # *** Troposphere ***
            # temperature lapse rate
            gamma = -0.0065
            # temperature 
            T = self.T0 + gamma * h # [K]
            #pressure
            p = self.p0 * (T/self.T0)**(-g/(gamma*R)) #[Pa]
            
        elif h <= 20.*10**3:
            # *** Tropopause ***
            # temperature is const at 11km-20km
            # p11 = pressure at 11km alt.
            T,p11,_,_ = self.standard_air(11000.)
            # pressure 
            p = p11 * np.exp( (-g/(R*T)) * (h-11000.) )
            
        elif h <= 32.*10**3:
            # *** Stratosphere 1 ***
            # temp, pressure at 20km alt.
            T20,p20,_,_ = self.standard_air(20000.)
            # temperature lapse rate
            gamma = 0.001
            # temperature 
            T = T20 + gamma * (h-20000.) # [K]
            #pressure
            p = p20 * (T/T20)**(-g/(gamma*R)) #[Pa] 
            
        elif h <= 47.*10**3:
            # *** Stratosphere 2 ***
            # temp, pressure at 32km alt.
            T32,p32,_,_ = self.standard_air(32000.)
            # temperature lapse rate
            gamma = 0.0028
            # temperature 
            T = T32 + gamma * (h-32000.) # [K]
            #pressure
            p = p32 * (T/T32)**(-g/(gamma*R)) #[Pa] 
            
        #END IF
        
            
        # density
        rho = p/(R*T) #[kg/m^3]
        
        # acoustic speed
        a = np.sqrt(1.4*R*T) # [m/s]
        
        return T,p,rho,a 
        
    
    def wind(self,h):
        # ==============================================
        # returns wind vector given an altitude
        #  follows "wind profile power law"
        #
        # INPUT: h = altitude [m]
        # ==============================================
        
        # wind velocity in local fixed coordinate
        wind_vec = self.wind_unitvec * self.wind_speed * (h/10.)**self.Cwind  

        return wind_vec
        
        
        
    def rocket_coeff_nofins(self, Mach,alpha):
        # input: Mach number
        #        alpha: angle of attack[rad]
        # -------------------
        # Lift Coefficient
        # -------------------
        
        if self.aero_fin_mode == 'indiv':
            # lift coefficient slope for fin-body individual computations
            #  : slope for body Cl
            k1 = 1.
        else:
            # lift coefficient slope for fin-body integrated computations
            k1 = 15.0
            k1 = 5.8
        # END IF
        Cl = k1 * alpha 
        
        #
        AoA_sep = 20.  # AoA at when flow separation occurs
        if alpha > AoA_sep*np.pi/180.:
            # if AoA is larger than 15 deg, assume separation and loss of Cl
            Cl_sep = k1 * AoA_sep*np.pi/180. # Cl at alpha=15 
            A = 0.1
            Cl = Cl_sep * np.exp( -A*( alpha-AoA_sep*np.pi/180. ) )
        #END IF
        
        # -------------------
        # Drag Coefficient
        # -------------------
        # drag coefficient "amplitude" for cosign curve fit
        Cd_bar = 5.
        # Cd = (self.Cd0 + k2*alpha) / np.sqrt(1-(Mach-0.05)**2.)
        Cd = self.Cd0 + Cd_bar*( np.cos(2*alpha + np.pi) +1. ) 
        
        # compressiblity correction
        if Mach > 0.93:
            Mach = 0.93
        # END IF
        
        Cd /= np.sqrt(1-Mach**2.)
        
        # print('AoA',alpha*180/np.pi, 'CL',Cl)
        """            
        # TSRP model
        if Mach < 0.6:
            Cd00 = 0.3
        elif Mach < 0.8:
            Cd00 = 0.3 + (Mach-0.6)/0.2 * (0.4-0.3)
        elif Mach < 1.0:
            Cd00 = 0.4 + (Mach-0.8)/0.2 * (0.62-0.4)
        elif Mach < 1.2:
            Cd00 = 0.62 + (Mach-1.)/0.2 * (0.70-0.62)
        elif Mach < 1.4:
            Cd00 = 0.7 + (Mach-1.2)/0.2 * (0.64-0.7)
        else:
            Cd00 = 0.5
        k2 = Cd00 / 0.17
        
        Cd = Cd00 + k2*alpha
        #print('Cd',Cd)
        """               

        return Cd, Cl
        
        
         
    def state2vecs_quat(self,u):
        # convert state vector u to vectors
        x = u[0:3]     # translation         :in fixed coordinate
        v = u[3:6]     # velocity            :in body coordinate
        q = u[6:10]    # atitude quaternion  :convert from fixed to body
        omega = u[10:] # angular velocity    :in body coordinate
    
        # convert to quaternion
        q2 = quaternion.as_quat_array(q)
    
        return x,v,q2,omega
        
        
                    
    def add_backup(self,t,u):
        # ==============================================
        # backup time, flag, state u
        # append vectors to the array "history"
        # ==============================================
               
        tmp = np.r_[t,self.flag,u]
        #try:
        self.backup = np.append(self.backup,[tmp],axis=0)
        #except:
        #    pass
        
        
    