#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:01:31 2023

@author: seba
"""

import numpy as np
from File_load import load_from_txt
from scipy import interpolate
import warnings
import math

class ODE: 
    """

    """
    def __init__(self, 
        turbine: dict, 
        t_array: np.array,
        w0: float,
        v: np.array,
        v_filt: np.array,
        pathfile_cp_ct_cq: str = "Cp_Ct_Cq.NREL5MW.txt",
        N_g: int = 97,
        eta_g: float = 0.944):
    
        self.turbine = turbine 
        self.t_array = t_array 
        self.w0 = w0
        self.dt = t_array[1] - t_array[0]
        self.v = v 
        self.v_filt = v_filt

        self.Data=load_from_txt(pathfile_cp_ct_cq)
        
        self.rho = self.turbine['rho']
        self.R = self.turbine['R']
        self.J = self.turbine['J']
        self.A = np.pi*self.R**2
        self.w_rated = self.turbine['w_rated']
        self.beta_rated = self.turbine['beta_rated']
        self.v_rated = self.turbine['v_rated']
        self.max_p_r=self.turbine['max_pitch_rate']
        self.max_t_r = self.turbine['max_torque_rate']
        self.N_g = N_g
        self.eta_g = eta_g
        self.tsr_rated=self.w_rated*self.R/self.v_rated
        self.tau_rated = 0.5*self.rho*self.A*self.interp_fun(self.beta_rated,self.tsr_rated)/(self.w_rated*self.N_g*self.eta_g)*self.v_rated**3
        
        index = np.argmax(self.Data[2])
        row, col = np.unravel_index(index, self.Data[2].shape)
        self.beta_opt=self.Data[0][col]
        self.tsr_opt=self.Data[1][row]
        self.cp_max=self.Data[2][row][col]
        
        self.tau_rated = 0.5*self.rho*self.A*self.R*self.cp_max/(self.tsr_opt*self.N_g)*self.eta_g*self.v_rated**2

        np.seterr(over='ignore')
        
        if w0==0:
            raise ValueError("You can not start the simulation with w0=0, choose another value")
        
    def interp_fun(self,beta,tsr):
        warnings.filterwarnings("ignore", category=DeprecationWarning) 
        interp = interpolate.interp2d(self.Data[0], self.Data[1], self.Data[2], kind='cubic')
        cp_interp = interp(beta,tsr)
        return cp_interp

    def sim( self, ind_func): 
        w = np.ones_like(self.t_array)*self.w0
        beta = np.zeros(len(self.t_array))
        tau = np.ones(len(self.t_array)) * self.tau_rated
        flag = False
        for i, t in enumerate(self.t_array):
            if i == 0:
                try:
                   beta[i] = ind_func(self.v_filt[i],w[i],0)
                except (ValueError, OverflowError):
                   beta[i] = np.pi/6
                   flag = True
                continue
            dt = self.t_array[i] - self.t_array[i - 1]
            try:
               w[i] = self.RK4(dt, w[i-1], self.v[i-1], beta[i-1], tau[i-1])
            except (ValueError, OverflowError):
               w[i] = w[i-1]
               flag = True
            try:
                e_i = np.trapz(w[0:i]-self.w_rated,None,dt)
            except (ValueError, OverflowError):
               e_i = 0
               flag = True
            try:
               beta[i] = ind_func(self.v_filt[i],w[i],e_i)
            except (ValueError, OverflowError):
               beta[i] = 0
               flag = True
            if (beta[i]-beta[i-1])/dt>self.max_p_r:
                beta[i]=self.max_p_r*dt+beta[i-1]
            if (beta[i]-beta[i-1])/dt<-self.max_p_r:
                beta[i]=-self.max_p_r*dt+beta[i-1]
        return w,beta,tau,flag
    
    def RK4(self,dt,w,v,beta,tau):
        k1 = dt * self.ode(w, v, beta, tau)
        k2 = dt * self.ode(w + 0.5 * k1, v, beta, tau)
        k3 = dt * self.ode(w + 0.5 * k2, v, beta, tau)
        k4 = dt * self.ode(w + k3, v, beta, tau)
        w_new = w + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return w_new

    def ode(self,w,v,beta,tau):
        if v==0:
            taua=0
        else:
            lam=w*self.R/v
            cp=self.interp_fun(beta,lam)
            taua=0.5*self.rho*self.A*cp/w*v**3
        dw_dt=1/self.J*(taua-self.N_g*tau/self.eta_g)
        return dw_dt

    




    
