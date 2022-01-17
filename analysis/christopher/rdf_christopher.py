#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def boxData(file):
    """
    The functions opens a DUMP file and extracts the box corner coordinates and
    number of particles.
    The function returns a list of sizes ([Lx, Ly, Lz]), each value representing 
    lengtgh of a corresponding edge, and an integer (N) representing the number of particles
    
    Arguments:
        file = string (path to DUMP file)
    
    Returns:
        dim = array([Lx, Ly, Lz]) (box dimensions)
        N = int (number of particles)
    """
    
    dim = [] #Placeholder for box dimensions
    with open(file) as dump_file:
        check_bounds = False #Will switch to TRUE when ITEM: BOX BOUNDS is found
        check_N = False #Will switch to TRUE when ITEM: NUMBER OF ATOMS is found
        for row in dump_file: #Searches through dump file for BOX BOUNDS
            if check_N: #This row will be the number of particles
                N = int(row) #Stores N as integer
                check_N = False #Must be sqitched back when N is stored
            elif check_bounds:
                dim.append(tuple([float(sci_num) for sci_num in row.split(' ')]))#re-formats cooardinate values from scientific notation as floats and stores in tuple
                if len(dim) == 3:#After three lines, all coordinates have been extracted
                    break
            elif row.startswith('ITEM: NUMBER OF ATOMS'): #Next line will be N
                check_N = True
            elif row.startswith('ITEM: BOX BOUNDS'): #Next three lines will include box-coordinates for x,y,z
                check_bounds = True
                
    return np.array([i[1] - i[0] for i in dim]), N


def loadDUMP(file):
    """
    Opens a DUMP file and extracts the box corner coordinates. Requires csv_file to exist
    Returns a pandas DataFrame with colums 'time_step', 'id', 'x', 'y' and 'z'.
    
    Arguments:
        file = string (path to DUMP file (not the csv version))
    
    Returns:
           time_step   id      x      y      z
        0        int  int  float  float  float
        1        int  int  float  float  float
        2        int  int  float  float  float
        ...      ...  ...    ...    ...    ...
        n        int  int  float  float  float
    """
    return pd.read_csv('./'+file+'.csv').loc[:,['time_step', 'id', 'x', 'y', 'z']] #reads dump file
     

def extractCoordinates(dump_as_df, time_step):
    """
    Extracts coordinates of all particles at given time step from pandas DataFrame.
    Returns 2D numpy array, (coor), with sets of coordinates
    
    Arguments:
        dump_as_df = DataFrame (pandas DataFrame of given dump file)
        time_step = int (time_step being extracted)
    
    Returns:
        coor = array([[x1, y1, z1], 
                      [x2, y2, z2],
                      [x3, y3, z3],
                      ...,
                      [xn, yn, zn]])
    """
    
    coor = dump_as_df[dump_as_df['time_step'] == time_step][['x', 'y', 'z']].values
    return coor


def calcPeriodicDistance(coor, dim):
    """
    Calculates the periodic inter-particle distances in the fluid from
    numpy coordinate array (coor) and box dimensions (dim)
    Returns a numpy array with all distances.
    
    Arguments:
        coor = array([[x1, y1, z1], 
                      [x2, y2, z2],
                      [x3, y3, z3],
                      ...,
                      [xn, yn, zn]])
    
        dim = array([Lx, Ly, Lz])dim = array([Lx, Ly, Lz]) (box dimensions)
        
    Returns:
        periodic_dist = array([dist_1, dist_2, dist_3, ..., dist_n])
    """
    diff_vec = np.array([np.subtract.outer(particle, particle) for particle in coor.T]).T #Creates 3D numpy array of all absolute distance vectors
    periodic_diff_vec = np.remainder(diff_vec + dim/2.0, dim) - dim/2.0 #Accounts for periodic boundary conditions.
    periodic_dist = np.linalg.norm(periodic_diff_vec, axis=2) #Calculates all periodic distances.
    periodic_dist = periodic_dist[np.nonzero(periodic_dist)] #Removes self-pairings
    
    return periodic_dist

def binningVolume(bin_edges):
    """
    Calculates the volume of the binning spheres at distances r from origin
    Returns two numpy arrays with distance to centre of bin (r) and binning volume (V_bin)
    
    Arguments:
        bin_edges = array([0, dr, 2dr, ..., (N+1)dr]) (the bin edges)
    
    Returns:
        r = array([r1, r2, r3, ..., r_max])
        V_bin = array([V1, V2, V3, ..., V_max])
    """
    r1 = bin_edges[0:-1] #Lower binning bounds
    r2 = bin_edges[1::] #Upper binning bounds
    r_func = lambda r1, r2: (r1+r2)*0.5 #Calculates ditance to centre of bin
    V_func = lambda r1, r2: 4/3 * np.pi * (r2**3 - r1**3) #Calculates volume of bin
    r = r_func(r1, r2) #Applies function on arrays
    V_bin = V_func(r1, r2) #Applies function on arrays
    
    return r, V_bin


def binningAlgorithm(r_max, dr):
    """
    Calculates binning edges, given a max distance (r_max) and bin thickness (dr)
    Returns the binning edges as 1D array.
    
    Arguments:
        r_max = float (the max plotting distance away from origin)
        
    Returns:
        bin_edges = array([edge1, edge2, edge3, ..., edge_r_max])
    """
    N_bins = np.ceil(r_max/dr) #Number of bins rounded up to nearest int
    bin_edges = np.arange(N_bins+1)*dr #N_bins+1 gives upper binning edge
    
    return bin_edges


def RDFAtTimestep(dist, bin_edges, V_bin, dim, N):
    """
    Calculates g(r) (g_r) for given set of distance measurments. The 
    measurments are binned acording to the provided bin edge argument using 
    np.histogram().
    Returns a 1D array of g(r)
    
    Arguments:
        dist = array([dist1, dist2, ..., dist_max])
        bin_edges = array([edge1, edge2, edge3, ..., edge_max])
        V_bin = array([V1, V2, V3, ..., V_max])
        dim = array([Lx, Ly, Lz]) (box dimensions)
        N = int (number of particles in system)
        
    Returns:
        g_r = array([g1, g2, g3])
    """
    V_box = np.prod(dim)
    g_r = np.histogram(dist, bins=bin_edges)[0]*V_box/V_bin/N**2 #Deviding by N accounts for double counting
    
    return g_r


def calcRDF(file, every, repeat, freq, r_max, dr, only_component_2=False):
    """
    Calculates an average g(r) accross multiple timesteps of a simulation.
    Saves CSV-file with plotting data for RDF
    
    Arguments:
        file = string (path to dump file)
        every = int (number of recorded time steps between averaging steps)
        repeat = int (number of points to include in average)
        freq = int (number of locations in dump sampled. Creates a seperat g_avg for each sample)
        r_max = float (maximum distance from origin plotted in RDF)
        dr = float (bin thickness, typically roughly 0.01. 
                    Number of bins = r_max/dr is always rounded up so that r_max <= N_bins*dr)
        
    Returns: (Saves CSV file for easy independent plotting)
        Void
    """
    
    #The following loads dump file and calculates all common parameters
    dim, N = boxData(file) #Extracts box dimensions and N from dump file
    dump_as_df = loadDUMP(file) #Loads dump.csv into memory for quick access

    # This removes particles according to their ID. 
    # Particles of ID > 500 are component 2.
    if only_component_2:
        dump_as_df = dump_as_df.drop(dump_as_df[dump_as_df.id < 500].index)
        N = N - 500
    bin_edges = binningAlgorithm(r_max, dr) #Defines the bin_edge param for the calculation
    r, V_bin = binningVolume(bin_edges) #Uses bin_edges to calculate distance and volume of bins
    
    #The following creates a list time stamps to be averaged over
    time_steps = np.unique(dump_as_df['time_step'].values) #Stores 1D np.array of recorded time steps in dump
    samples = np.split(time_steps[time_steps.size%freq::], freq) #Devides time_steps array according to freq argument
    samples = [sample[(sample.size-1)%every::every][-repeat::] for sample in samples] #Selects time_steps based on every and repeat argument
    
    #The following calculates time average RDF g_avg at time steps in samples and stores in numpy array g_all
    g_all = [] #Empty list for appending averages
    g_err = [] #Empty list for appending averages
    for sample in samples:
        g_avg = [] #Empty list for appending g_r at recorded time_steps
        for step in sample:
            coor = extractCoordinates(dump_as_df, step) #Extracts coordinates for given time step
            dist = calcPeriodicDistance(coor, dim) #Calculates the peeriodic distances for given coordinates
            g_avg.append(RDFAtTimestep(dist, bin_edges, V_bin, dim, N)) #Averages g_r accros time and appends to g_avg
        g_all.append(np.mean(g_avg, axis=0)) #Appends result to g_all
        g_err.append(np.std(g_avg, axis=0)/np.sqrt(len(g_avg))) #Appends result to g_all
    
    #The following code creates DataFrame and stores as csv for plotting
    file_name = file.replace('dump', 'RDF')+'.csv' #Generates csv file name from dump name
    col_names = ['r'] + ['g' + str(i) for i in np.arange(1,freq+1)] + ['err' + str(i) for i in np.arange(1,freq+1)] #Generates col_names for pandas
    plotting_csv = pd.DataFrame([r]+g_all+g_err, index=col_names).transpose() #Transposing because of how pandas handles lists of arrays
    plotting_csv.to_csv(file_name, index=False) #Saves CSV-file
    
    #Creates a simple plot of g in g_all
    #for g in g_all:
    #    plt.plot(r, g)
    #    plt.legend([file]) #Legend is file name. Usefull if looping through multiple DUMP-files
    #plt.show()
    #plt.close()
    
    return #Nothing to return. Data saved as CSV


def plotRDF_fromCSV(path='./', file_name='RDF', plot_name=None, fig_size=(10,6), fnt_size=15, legend_keys=['xi', 'L', 'A', 'Ealign']):#, color_scheme=None):
    """
    Plotting function for g_r CSV files. Will iterate through contents of directory and plot all RDF.csv files
    
    Arguments:
        path = string (Defaults current directory if not specified)
        file_name = string (Defaults to 'RDF' if not specified)
        plot_name = string (Defaults to plot_name = file_name if not specified)
        fig_size = tuple (int, int) (Size of plotting bok. Defaults to (15, 10))
        fnt_size = int (font size of chart)
        legend_keys = list [param1, param2, ...] (List of keys to include in legend)
        color_scheme = dict (Defaults to None. format: {filename: fmt})
        
    Returns:
        Void
    """
    legend_list = []
    
    plt.figure(figsize=fig_size)
    plt.title((lambda a, b: b if b is not None else a) (file_name, plot_name), fontsize=fnt_size*1.5) #Uses file_name if plot_name not given
    plt.xlabel('g(r)', fontsize=fnt_size*1.2)
    plt.ylabel('r', fontsize=fnt_size*1.2)
    plt.xticks(np.arange(0,10,0.5), fontsize=fnt_size)
    plt.yticks(np.arange(0,10,1), fontsize=fnt_size)
    
    files = sorted([file for file in os.listdir(path) if file.startswith('RDF') and file.endswith('.csv')], key=lambda f: (f))
    for i, file in enumerate(files):
        if file.startswith('RDF.') and file.endswith('.csv'):
            file_type = file.split('.')[1].replace('_', ' ').replace(' MP', '')
            file_param = [var.split('_') for var in '.'.join(file.split('.')[2:-2]).split('--')]
            leg_param = ',  '.join(pair[0]+': '+pair[1] for pair in file_param if pair[0] in legend_keys).strip(', ')
            leg = file_type + ':  ' + leg_param
            legend_list.append(leg)
            #legend_list.append(file)
            data = pd.read_csv(file)
            for col in data.columns:
                if 'g' in col:
                    plt.plot(data['r'], data[col])#, color_scheme[i])
    plt.xlim((0, round(data['r'].iloc[-1], 2)))
    plt.legend(legend_list, fontsize=fig_size[0])
    plt.grid(b=True)
    plt.savefig(file_name)
    plt.show()
    plt.close()

    return

