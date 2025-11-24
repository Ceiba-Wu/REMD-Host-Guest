import pandas as pd
import numpy as np
import math

def w_cal(df):
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    #fitting
    coef = np.polyfit(x,y,2)
    y_fit = np.polyval(coef, x)
    f = np.poly1d(coef)
    w = np.sqrt(np.abs(np.polyder(f,2)))/(2*math.pi)
    return w, x,  y_fit

def kramer_rate_cal(w_0, w_m, Ea, D, T=300):
    #Ea: kcal/mol
    #D: 10^-9*m^2*s^-1
    #T: K
    #k_rate: s^-1
    kB = 1.380649*pow(10,-23)
    NA_constant = 6.022*pow(10,23)
    friction_constant = (kB*T/D)*pow(10,12)
    A = (2*(np.pi)*w_0*w_m)/(friction_constant)
    k_rate = A*np.exp(-(4.1859*Ea*pow(10,3))/(kB*T*NA_constant))
    k_rate_format = format(k_rate, "e")
    return k_rate_format
