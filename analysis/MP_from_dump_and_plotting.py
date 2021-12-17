#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 17:37:31 2021

@author: christopher
"""
import pandas as pd
pd.options.mode.chained_assignment = None
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as t_test
from scipy.stats import stats
import itertools

import convert_LAMMPS_output as convert


##-------------------------------
# MEASURMENT
##-------------------------------

def extract_constants_from_log(log_path, var_name_measurment_time, var_name_box_cross_section, var_name_height):
    """Extract the names and values of the variables defined in the LAMMPS log
    given by log_path.

    INPUT:
        log_path: string

    OUTPUT:
        dictionary of (string, number)
    """
    variables = {}

    with open(log_path) as log_file:

        for row in log_file:

            if not row.startswith('variable'):
                continue

            words = row.split()
            if words[1] == var_name_measurment_time:
                variables[words[1]] = int(words[3])
            elif words[1] == var_name_box_cross_section:
                try:
                    variables[words[1]] = eval(words[3].replace('PI', 'np.pi').replace('^', '**'))
                except:
                    pass
            elif words[1] == var_name_height:
                try:
                    variables[words[1]] = eval(words[3])
                except:
                    pass
    return variables


def create_bin_bounds(box_height, n_bins):
    bin_size = box_height/n_bins
    
    bin_egdes_lower = np.append(np.arange(-box_height/2, 0.0, bin_size), 0.0)
    z_lower = (bin_egdes_lower[1:] + bin_egdes_lower[:-1])/2
    
    bin_egdes_upper = np.append(np.arange(0.0, box_height/2,bin_size), box_height/2)
    z_upper = (bin_egdes_upper[1:] + bin_egdes_upper[:-1])/2
    
    return bin_egdes_lower, z_lower, bin_egdes_upper, z_upper


def CreateMeanBins(data_frame, bin_edges, bin_value, bin_column='z', data_column='vx'):
    vx_mean = []
    for i, j in zip(bin_edges, bin_edges[1:]):
        vx_mean.append(data_frame.loc[(data_frame[bin_column] > i) & (data_frame[bin_column] <= j), data_column].mean())
    return pd.DataFrame([bin_value, vx_mean], index=[bin_column + '_bin', data_column + '_mean']).transpose()


def MullerPlathe(path, log_path, n_bins=100, bin_column='z', data_column='vx', measurment_time='VISC_TIME', measurment_fraction=0.5, box_base='BOX_BASE', box_height='BOX_HEIGHT'):
    log = pd.read_csv(path + log_path + '.csv')
    dump = pd.read_csv(path + 'dump.' + log_path[4:] + '.csv')
    
    const = convert.extract_constants_from_log(path + log_path)#, measurment_time, box_base, box_height)
    
    t = const[measurment_time]
    A = (2*const[box_base])**2
    height = 2*const[box_height]

    P_x = log['f_MULLER_PLATHE'].iloc[-1]
    bin_egdes_lower, z_lower, bin_egdes_upper, z_upper = create_bin_bounds(height, n_bins)
    data = dump.iloc[int(measurment_fraction*len(dump)):].loc[:,[bin_column, data_column]]
    
    lower_data = data[data[bin_column] < 0]
    upper_data = data[data[bin_column] > 0]
    
    lower_regress_data = CreateMeanBins(lower_data, bin_egdes_lower, z_lower)
    lower_regress_data.drop(lower_regress_data.tail(1).index, inplace=True)
    lower_regress_data.drop(lower_regress_data.head(1).index, inplace=True)
    lower_regress_data.dropna(inplace=True)
    
    upper_regress_data = CreateMeanBins(upper_data, bin_egdes_upper, z_upper)
    upper_regress_data.drop(lower_regress_data.tail(1).index, inplace=True)
    upper_regress_data.drop(lower_regress_data.head(1).index, inplace=True)
    upper_regress_data.dropna(inplace=True)
    
    bin_name = bin_column + '_bin'
    mean_name = data_column + '_mean'
    lower_regression = stats.linregress(lower_regress_data[bin_name].values, lower_regress_data[mean_name].values)
    upper_regression = stats.linregress(upper_regress_data[bin_name].values, upper_regress_data[mean_name].values)
    
    #print(log_path)
    #plt.scatter(lower_regress_data[bin_name], lower_regress_data[mean_name], label='Data')
    #plt.plot(lower_regress_data[bin_name], lower_regression.intercept + lower_regression.slope*lower_regress_data[bin_name], 'r', label='Fitted line')
    #plt.title('Lower regression')
    #plt.xlabel('z')
    #plt.ylabel('v_x')
    #plt.legend()
    #plt.show()
    #plt.close()

    #plt.scatter(upper_regress_data[bin_name], upper_regress_data[mean_name], label='Data')
    #plt.plot(upper_regress_data[bin_name], upper_regression.intercept + upper_regression.slope*upper_regress_data[bin_name], 'r', label='Fitted line')
    #plt.title('Upper regression')
    #plt.xlabel('z')
    #plt.ylabel('v_x')
    #plt.legend()
    #plt.show()
    #plt.close()

    tinv = lambda p, df: abs(t_test.ppf(p/2, df))
    ts_lower = tinv(0.05, len(lower_regression)-2)
    lower_error = ts_lower*lower_regression.stderr
    ts_upper = tinv(0.05, len(upper_regression)-2)
    upper_error = ts_upper*lower_regression.stderr
      
    shear_rate = (abs(lower_regression.slope) + abs(upper_regression.slope))/2
    shear_rate_error = np.sqrt(lower_error**2 + upper_error**2)/2
    
    error = abs((P_x/2/t/A)*(1/shear_rate**2)*shear_rate_error)
    
    momentum_flux = P_x/2/t/A
    eta = -momentum_flux/shear_rate
    
    #print(log_path)
    #print('Eta: ', eta, 'P_x: ', P_x, 'time: ', t, 'Area: ', A, 'Shear rate: ', shear_rate)
    
    return eta, error, shear_rate, shear_rate_error, momentum_flux

    

def Create_CSV_PHS(selection='', measurment_time='VISC_TIME'):
    csv = []
    for file in os.listdir():
        if file.startswith('log') and file.endswith('visc') and selection in file:
            print(file)
            xi = float('.'.join(file.split('.')[2:-1]).split('--')[1].split('_')[1])
            rho = 6*xi/np.pi
            eta, eta_error, shear_rate, shea_rate_error, momentum_flux = MullerPlathe(file, measurment_time=measurment_time)
            csv.append([rho, xi, eta, eta_error, shear_rate, shea_rate_error, momentum_flux])
    file_name = file.split('.')[1]
    if len(selection) > 0:
        file_name = selection + '.' + file_name
    csv = pd.DataFrame(csv, columns=['rho', 'xi', 'eta', 'eta_error', 'shear_rate', 'shea_rate_error', 'momentum_flux'])
    csv.to_csv('./'+'plotting_data.'+file_name+'.csv', index=False)



##-------------------------------
# Plotting
##-------------------------------

def eta_0_reduced_Christopher_calc(temp_reduced):
    return (5.0/16.0)*(temp_reduced/np.pi)**(1/2)
    
def V_excl_calc(sigma):
    return (2.0/3.0) * np.pi * (sigma**3)

def RDF_CS_calc(xi):
    return (1.0 - 0.5*xi)/((1.0 - xi)**3)

def Enskog_HS_reduced_Christopher(xi, sigma=1.0, temp=1.5):
    rho = 6*xi/np.pi#/sigma**3
    eta_0 = eta_0_reduced_Christopher_calc(temp)
    V_excl = V_excl_calc(sigma)
    RDF = RDF_CS_calc(xi)
    return eta_0*((1/RDF) + 0.8*V_excl*rho + 0.776*(V_excl**2)*(rho**2)*RDF)



def plotting_single_param_data_from_csv(
        plotting_file, 
        var_x='xi', 
        var_y='eta_norm', 
        background=None,#pd.DataFrame([[0.1,0.5],[1,1]]).T,
        output_name='viscosity_plot',
        plot_title=None,
        legend=None,
        x_label='', 
        y_label='',
        color_scheme=None, 
        fmt_scheme='o', 
        fill_scheme=None,
        Enskog_choice=Enskog_HS_reduced_Christopher
        ):
    """
    Doc-string
    """
    
    plot = plt.figure(figsize=(15,10))
    ax = plot.add_subplot(1, 1, 1)
    if background is not None:
        ax.plot(background.iloc[:,0], background.iloc[:,1], label=background.columns[1])
        
    plotting_data = pd.read_csv(plotting_file)
    plotting_data['eta_enskog'] = [Enskog_choice(xi) for xi in plotting_data['xi']]
    plotting_data['eta_norm'] = plotting_data['eta']/plotting_data['eta_enskog']
    plotting_data['eta_norm_error'] = plotting_data['eta_error']/plotting_data['eta_enskog'] 
    
    fmt_string = fmt_scheme
    color_string = color_scheme
    fill_string = fill_scheme
    
    if var_y + '_error' in plotting_data.columns:
        ax.errorbar(plotting_data[var_x], plotting_data[var_y], plotting_data[var_y + '_error'], fmt=fmt_string, color=color_string, fillstyle=fill_string, markersize=9, label=legend)
    else:
        ax.plot(plotting_data[var_x], plotting_data[var_y], marker=fmt_string, color=color_string, fillstyle=fill_string, markersize=9, linestyle='none', label=legend)

    ax.legend(fontsize=10)
    ax.set_xlabel(x_label, fontsize=15)#'\u03BE (Volume density)', fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)#('\u03B7*/\u03B7_Enskog', fontsize=15)
    ax.set_ylim(0.8,1.2)
    ax.set_title(plot_title, fontsize=20)
    ax.tick_params(labelsize=15)
    ax.grid()
    plot.savefig(output_name)
