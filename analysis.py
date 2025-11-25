import pandas as pd
import numpy as np
import math
import sys

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

def calculate_energy_difference(probability_1, probability_2, temperature):
    """
    Calculate the energy difference ΔE in both joules (J) and kcal/mol.

    :param probability_1: Probability of state 1 (P1)
    :param probability_2: Probability of state 2 (P2)
    :param temperature: Temperature in Kelvin (T)
    :return: Energy difference in joules (J) and kcal/mol
    """
    # Boltzmann constant (J/K)
    boltzmann_constant = 1.38e-23

    # Validate probability values
    if probability_1 <= 0 or probability_2 <= 0:
        raise ValueError("Probabilities must be positive numbers.")

    # Calculate energy difference in joules
    delta_e_joules = -boltzmann_constant * temperature * math.log(probability_2 / probability_1)

    # Convert joules to kcal/mol
    joule_to_kcal = 2.39e-4  # Conversion factor from J to kcal
    avogadro_number = 6.022e23  # Particles per mole
    delta_e_kcal_per_mol = delta_e_joules * joule_to_kcal * avogadro_number

    return delta_e_joules, delta_e_kcal_per_mol
