# This script analyses the RDF of a system simulated 
# in LAMMPS, working from LAMMPS dump files.

# Most of this script was created by Christopher.

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import os
import sys
import logging

#import numba as nb
#from numba import njit
#from numba.np.ufunc import parallel
#from numba import prange

import files
import save
import utils
import convert
import theory
import rdfpy

sysargs = sys.argv
log = logging.getLogger()
log.addHandler(logging.StreamHandler(sys.stdout))
if "debug" in sysargs:
    # logging.Debug is more correct to use, 
    # but info is cleaner.
    log.setLevel(logging.INFO)

### Wrapper function created by Jonas. ###

def compute_all_rdfs(
        directory, 
        theory_functions, 
        computation_params,
        cut=0.9,
        dr=0.05,
        recompute=True
    ):
    path = directory
    filenames = files.get_all_filenames(directory)
    rdf_list = theory_functions
    N = computation_params["particle_types"]
    data = save.create_data_array(filenames, rdf_list, N, extra_values=2)
    freq = 2
    every = 100
    repeat = 1
    for (i, f) in enumerate(filenames):
        utils.status_bar(i, len(filenames), fmt="percent")
        log_name = f"{path}/" + f[1]
        #fix_name = f"{path}/" + f[2]
        savename = log_name.replace("log", "rdf") + ".csv"
        log.info(f"Loading file\t{log_name}")
        C = convert.extract_constants_from_log(log_name)
        if recompute:
            dump_name = f"{path}/" + f[2]
            log.info(f"Loading file\t{dump_name}")
            r_max=cut*C["LX"]
            # Note that this gives out g_sigma and error,
            # and that fast_rdf does not.
            rdf, r, g_sigma, error = get_rdf_from_dump(dump_name, log_name, r_max)
            # fast_rdf uses the rdfpy library. It is parallellized and
            # memory-efficient, but does not give RDFs that 
            # approach 1 at long distances (known issue).
            #rdf, r, t = fast_rdf(dump_name, log_name, freq, every, repeat, cut, dr)
            # For fast_rdf, compute an average over all times
            #rdf, std = rdf_average(rdf)
            # Trim zeros at the end, and drop corresponding values of rdf.
            r = np.trim_zeros(r, trim="b")
            rdf = rdf[:len(r)]
            save_data = np.zeros((len(rdf),3))
            save_data[:len(r),0] = r
            save_data[:len(rdf),1] = rdf
            save_data[:len(rdf),2] = error
            np.savetxt(savename, save_data, delimiter=", ", header="r, g, error", comments="")
            log.info(f"Saved to file\t{savename}")
            g_sigma = np.amax(rdf)
            j = np.where(rdf == g_sigma)[0][0]
            #error = std[j]
        else:
            rdf_data = pd.read_csv(savename, sep=", ", engine="python")
            rdf = np.array(rdf_data["g"])
            r = np.array(rdf_data["r"])
            g_max = np.amax(rdf)
            sigma_index = np.abs(r - 1).argmin()
            g_one = rdf[sigma_index]

            sigma_eff = theory.LJ_effective_sigma(C["TEMP"], 1.0)
            sigma_eff_index = np.abs(r-sigma_eff).argmin()
            g_eff = rdf[sigma_eff_index]

            # Compute g_sigma from Z and U as well?
            #Z = 
            #g_F = 

            # Error can not be included here in current version.
            # To compute error, recompute must be True.
            error = np.zeros_like(g_max)

            #data = np.array([g_max, error])
        # In both cases, save everything to data.
        theoretical_values = [theory.get_rdf_from_C(C, g) for g in theory_functions]
        #values = np.array([g_max, error])
        values = np.array([g_max, g_one, g_eff, error])
        values = np.append(values, theoretical_values)
        save.insert_results_in_array(data, values, C, i)
    print("")
    return data


def get_g_sigma_from_directory(
        directory, 
        theory_functions, 
        computation_params,
        cut=0.9,
        dr=0.05
    ):
    path = directory
    filenames = files.get_all_filenames(directory)
    rdf_list = theory_functions
    N = computation_params["particle_types"]
    data = save.create_data_array(filenames, rdf_list, N)
    for (i, f) in enumerate(filenames):
        utils.status_bar(i, len(filenames), fmt="percent")
        dump_name = f"{path}/" + f[2]
        #log_name = f"{path}/" + f[1]
        rdf_name = dump_name.replace("dump", "rdf") + ".csv"
        log.info(f"Loading file\t{dump_name}")
        log.info(f"Loading file\t{log_name}")

        data = pd.read_csv(rdf_name)
        rdf = data["g"]
        g_sigma = np.amax(rdf)

        #j = np.where(rdf == g_sigma)[0][0]
        #error = std[j]

        theoretical_values = [theory.get_rdf_from_C(C, g) for g in theory_functions]
        values = np.array([g_sigma, error])
        values = np.append(values, theoretical_values)
        save.insert_results_in_array(data, values, C, i)
    print("")
    return data


def fast_rdf(dump_name, log_name, freq, every, repeat, cut, dr):
    # The following loads dump file and calculates all common parameters
    # Extracts box dimensions and N from dump file
    dim, N = boxData(dump_name) 
    log_table = files.load_system(log_name)
    C = convert.extract_constants_from_log(log_name)
    # Loads dump.csv into memory for quick access
    dump_as_df = loadDUMP(dump_name) 

    # Stores 1D np.array of recorded time steps in dump
    time_steps = np.unique(dump_as_df['time_step'].values) 
    # Divides time_steps array according to freq argument
    samples = np.split(time_steps[time_steps.size%freq::], freq) 
    # Selects time_steps based on every and repeat argument
    samples = [sample[(sample.size-1)%every::every][-repeat::] for sample in samples] 
    N_r = int(C["LX"]*cut//dr) + 10 # The +10 is to avoid crashes. 
                                    # Trailing zeros are removed later.
    N_t = len(time_steps)
    rdf_of_t = np.zeros((N_t, N_r))
    radii = np.zeros(N_r)
    for i in range(N_t):
        particle_positions = extractCoordinates(dump_as_df, time_steps[i])
        g_r, radii = rdfpy.rdf(particle_positions, dr, rho = 6*C["PF"]/np.pi, rcutoff=cut, progress=True)
        rdf_of_t[i,:len(radii)] = g_r
    return rdf_of_t, radii, time_steps


def rdf_average(rdf):
    return np.mean(rdf, axis=0), np.std(rdf, axis=0)



def get_rdf_from_dump(dump_name, log_name, r_max, interaction_number=1):
    # This saves the result to a file, with the same name 
    # the original dump file, with "dump" exchanged for "rdf".
    calcRDF(dump_name, 1, 100, 2, r_max, 0.05)
    rdf_name = dump_name.replace("dump", "rdf") + ".csv"
    g_sigma, g_r, r, std = export_rdf(rdf_name, interaction_number)
    return g_r, r, g_sigma, std



# Given a directory, compute the RDF in the system, 
# as a function of time, during equillibration. 
def export_rdf(rdf_name, interaction_number=1):
    """
        Takes a directory with RDF files and creates a csv for use in pgfplots.
        The output data gile contains g(sigma) for all rdfs in the directory.
        The original RDF file must be created with calcRDF from RDF_from_DUMP.
    """

    data = pd.read_csv(rdf_name)
    # Array of rdf measured at different r
    g_r = np.array(data[f"g{interaction_number}"])   
    std = np.array(data[f"err{interaction_number}"])

    r = np.array(data["r"])         # Array of r
    g_sigma = np.amax(g_r)          # Value of rdf measured at sigma
    g_sigma_index = np.argmax(g_r)  # Index of g_sigma
    std = std[g_sigma_index]
    return g_sigma, g_r, r, std


def show_an_rdf(rdf_name):
    data = pd.read_csv(rdf_name)
    g_r = np.array(data["g1"])      # Array of rdf measured at different r
    std = np.array(data["err1"])
    r = np.array(data["r"])         # Array of r


### The remaining is Christopher's original code. ###

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
        check_bounds = False    # Will switch to TRUE when ITEM: BOX BOUNDS is found
        check_N = False         # Will switch to TRUE when ITEM: NUMBER OF ATOMS is found
        for row in dump_file:   # Searches through dump file for BOX BOUNDS
            if check_N:         # This row will be the number of particles
                N = int(row)    # Stores N as integer
                check_N = False # Must be sqitched back when N is stored
            elif check_bounds:
                # re-formats cooardinate values from scientific notation as
                # floats and stores in tuple
                dim.append(tuple([float(sci_num) for sci_num in row.split(' ')]))
                #After three lines, all coordinates have been extracted
                if len(dim) == 3:
                    break
            elif row.startswith('ITEM: NUMBER OF ATOMS'): 
                #Next line will be N
                check_N = True
            elif row.startswith('ITEM: BOX BOUNDS'): 
                #Next three lines will include box-coordinates for x,y,z
                check_bounds = True
                
    return np.array([i[1] - i[0] for i in dim]), N


def loadDUMP(file):
    """
    Opens a DUMP file and extracts the box corner coordinates. 
    Requires csv_file to exist Returns a pandas DataFrame with 
    colums 'time_step', 'id', 'x', 'y' and 'z'.
    
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
    #reads dump file
    return pd.read_csv('./'+file+'.csv').loc[:,['time_step', 'id', 'x', 'y', 'z']] 
     

def extractCoordinates(dump_as_df, time_step):
    """
    Extracts coordinates of all particles at given time step from pandas
    DataFrame.  Returns 2D numpy array, (coor), with sets of coordinates
    
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


def calcPeriodicDistance(coor, dim, r_max=np.inf):
    """
    Calculates the periodic inter-particle distances in the
    fluid from numpy coordinate array (coor) and box
    dimensions (dim). Returns a numpy array with all
    distances.
    
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
    # Creates 3D numpy array of all absolute distance vectors
    diff_vec = np.array([np.subtract.outer(particle, particle) for particle in coor.T]).T 
    #diff_vec = np.array([np.subtract.outer(particle, particle) for particle in coor.T]).T 
    # Accounts for periodic boundary conditions.
    periodic_diff_vec = np.remainder(diff_vec + dim/2.0, dim) - dim/2.0 
    # Calculates all periodic distances.
    periodic_dist = np.linalg.norm(periodic_diff_vec, axis=2) 
    # Remove distances longer than r_max. This has no effect on speed.
    #periodic_dist = periodic_dist[(np.abs(periodic_dist) < r_max)]
    # Removes self-pairings
    periodic_dist = periodic_dist[np.nonzero(periodic_dist)] 
    return periodic_dist


def calcPeriodicDistanceOptimized(coor, dim, r_max=np.inf):
    """
    Calculates the periodic inter-particle distances in the
    fluid from numpy coordinate array (coor) and box
    dimensions (dim). Returns a numpy array with all
    distances.
    
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
    # Creates 3D numpy array of all absolute distance vectors
    # TODO: Optimization: Create three 1D arrays instead, 
    # and perform most of the computation in 1D.

    # This does not seem to solve the issue, so code is unfinished.
    # Create arrays for each coordinate axis
    x, y, z = coor[0], coor[1], coor[2]
    diff_vec = np.zeros_like(coor)
    ignored_indices = np.arange(len(x))
    for i in range(3):
        c = coor[i][ignored_indices]                        # Coordinates in 1D
        diff = np.array([c - c[j] for j in range(len(c))])  # Distances
        diff = np.remainder(diff + dim/2.0, dim) - dim/2.0  # Periodic BCs
        # Ignore particles that are further away than r_max. 
        # These are ignored in all dimensions.
        ignored_indices = np.delete(ignored_indices, np.where(diff > r_max))
         

    diff_vec = np.array([np.subtract.outer(particle, particle) for particle in coor.T]).T 
    # Accounts for periodic boundary conditions.
    periodic_diff_vec = np.remainder(diff_vec + dim/2.0, dim) - dim/2.0 
    # Calculates all periodic distances.
    periodic_dist = np.linalg.norm(periodic_diff_vec, axis=2) 
    # Remove distances longer than r_max. This has no effect on speed.
    #periodic_dist = periodic_dist[(np.abs(periodic_dist) < r_max)]
    # Removes self-pairings
    periodic_dist = periodic_dist[np.nonzero(periodic_dist)] 
    return periodic_dist


def subtract_outer_optimized(arr):
    block_size=1024 # Divide the array into blocks of size 1024, to simplify 
    A = np.zeros(arr.shape[0])
    arr_length = arr.shape[0]//block_size
    for ii in nb.prange(arr_length):
        for jj in range(arr_length):
            for i in range(block_size):
                tmp = A[ii*block_size+i]
                for j in range(block_size):
                    tmp += arr[ii*block_size+i] - arr[jj*block_size+j]
                A[ii*block_size+i] = tmp
    for i in nb.prange(arr.shape[0]):
        tmp = A[i]
        for j in range(arr_length*block_size, arr.shape[0]):
            tmp += arr[i] - arr[j]
        A[i] = tmp
    for i in nb.prange(arr_length*block_size, arr.shape[0]):        
        tmp = A[i]
        for j in range(arr.shape[0]):
            tmp += arr[i] - arr[j]
        A[i] = tmp
    return A


def binningVolume(bin_edges):
    """
    Calculates the volume of the binning spheres at
    distances r from origin Returns two numpy arrays with
    distance to centre of bin (r) and binning volume (V_bin)
    
    Arguments:
        bin_edges = array([0, dr, 2dr, ..., (N+1)dr]) (the bin edges)
    
    Returns:
        r = array([r1, r2, r3, ..., r_max])
        V_bin = array([V1, V2, V3, ..., V_max])
    """
    r1 = bin_edges[0:-1]                    # Lower binning bounds
    r2 = bin_edges[1::]                     # Upper binning bounds
    r_func = lambda r1, r2: (r1+r2)*0.5     # Calculates ditance to centre of bin
    V_func = lambda r1, r2: 4/3 * np.pi * (r2**3 - r1**3) # Calculates volume of bin
    r = r_func(r1, r2)      # Applies function on arrays
    V_bin = V_func(r1, r2)  # Applies function on arrays
    
    return r, V_bin


def binningAlgorithm(r_max, dr):
    """
    Calculates binning edges, given a max distance (r_max)
    and bin thickness (dr) Returns the binning edges as 1D
    array.
    
    Arguments:
        r_max = float (the max plotting distance away from origin)
        
    Returns:
        bin_edges = array([edge1, edge2, edge3, ..., edge_r_max])
    """
    N_bins = np.ceil(r_max/dr)          # Number of bins rounded up to nearest int
    bin_edges = np.arange(N_bins+1)*dr  # N_bins+1 gives upper binning edge
    return bin_edges


def RDFAtTimestep(dist, bin_edges, V_bin, dim, N):
    """
    Calculates g(r) (g_r) for given set of distance
    measurments. The measurments are binned acording to the
    provided bin edge argument using np.histogram().
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
    # Dividing by N accounts for double counting
    g_r = np.histogram(dist, bins=bin_edges)[0]*V_box/V_bin/N**2 
    return g_r


def calcRDF(file, every, repeat, freq, r_max, dr, particle_types=1):
    """
        Calculates an average g(r) accross multiple
        timesteps of a simulation.
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
    
    # The following loads dump file and calculates all common parameters
    # Extracts box dimensions and N from dump file
    dim, N = boxData(file) 
    # Loads dump.csv into memory for quick access
    dump_as_df = loadDUMP(file) 
    # Defines the bin_edge param for the calculation
    bin_edges = binningAlgorithm(r_max, dr) 
    # Uses bin_edges to calculate distance and volume of bins
    r, V_bin = binningVolume(bin_edges) 
    
    # The following creates a list of
    # time stamps to be averaged over
    # Stores 1D np.array of recorded time steps in dump
    time_steps = np.unique(dump_as_df['time_step'].values) 
    # Divides time_steps array according to freq argument
    samples = np.split(time_steps[time_steps.size%freq::], freq) 
    # Selects time_steps based on every and repeat argument
    samples = [sample[(sample.size-1)%every::every][-repeat::] for sample in samples] 
    
    # TODO: Divide the system into particles according to type.
    # g.shape -> (N,3)
    # g.append() 3 times (1+m, where m is the number of components)
    # add ij, ii, jj to file header and data entries

    # The following calculates time average RDF g_avg at
    # time steps in samples and stores in numpy array g_all
    g_all = np.zeros((len(r), particle_types)) # Empty list for appending averages
    for (i, sample) in enumerate(samples):
        # Empty list for appending g_r at recorded time_steps
        g_avg = np.zeros((len(sample), len(r), freq))
        for (j, step) in enumerate(sample):
            g_r = np.zeros((len(r), particle_types))
            for k in range(particle_types):
                # Extracts coordinates for given time step
                # Compute rdf between particles of 
                #   - different types (k == 0)
                #   - the same type (k > 0)
                p = N
                # This computes the RDF from all to all particles:
                if k == 0: 
                    coor = extractCoordinates(dump_as_df, step) 
                # This computes the RDF for particle 1:
                elif k == 1:
                    p = N//2
                    coor = extractCoordinates(
                        dump_as_df.loc[dump_as_df['id'] < p],
                        step
                    ) 
                # This computes the RDF for particle 2:
                elif k == 2:
                    p = N//2
                    coor = extractCoordinates(
                        dump_as_df.loc[dump_as_df['id'] > p],
                        step
                    ) 
                # Calculates the periodic distances for given coordinates
                dist = calcPeriodicDistance(coor, dim) 
                # Averages g_r accros time and appends to g_avg
                g_r[:,k] = RDFAtTimestep(dist, bin_edges, V_bin, dim, p)
            g_avg[j] = g_r
        # Appends result to g_all
        g_all = np.mean(g_avg, axis=0)
        std = np.std(g_avg, axis=0)/np.sqrt(len(g_avg[:,0]))
    
    # The following code creates DataFrame and stores as csv for plotting.
    # Generates csv file name from dump name
    file_name = file.replace('dump', 'rdf')+'.csv' 
    # Generates col_names for pandas
    # TODO: add ij, ii, jj to file.
    col_names   = [f"g{i}" for i in np.arange(1,freq+1)] 
    std_names   = [f"err{i}" for i in np.arange(1,freq+1)] 
    if particle_types == 2:
        col_names = ( [f"g{i}_ii" for i in np.arange(1,freq+1)] 
                    + [f"g{i}_jj" for i in np.arange(1,freq+1)]
                    + [f"g{i}_ij" for i in np.arange(1,freq+1)]
                )
        std_names = ( [f"err{i}_ii" for i in np.arange(1,freq+1)]
                    + [f"err{i}_jj" for i in np.arange(1,freq+1)]
                    + [f"err{i}_ij" for i in np.arange(1,freq+1)]
                )
    # Transposing because of how 
    # pandas handles lists of arrays
    g_all = pd.DataFrame(g_all, columns=col_names)
    std = pd.DataFrame(std, columns=std_names)
    r = pd.DataFrame(r, columns=["r"])
    plotting_csv = pd.concat([r, g_all, std], axis=1)
    # Saves CSV-file og g(r) for this system:
    plotting_csv.to_csv(file_name, index=False) 
    
    # Creates a simple plot of g in g_all
    #for g in g_all:
    #    plt.plot(r, g)
    #    plt.legend([file])  # Legend is file name. Useful if 
    #                        # looping through multiple DUMP-files
    #plt.show()
    #plt.close()

    # Nothing to return. Data saved as CSV
    return 

#def plotRDF_fromCSV(path='./', file_name='RDF', plot_name=None, fig_size=(10,6), fnt_size=15, legend_keys=['xi', 'L', 'A', 'Ealign']):#, color_scheme=None):
#    """
#    Plotting function for g_r CSV files.  
#    Will iterate through contents of
#    directory and plot all RDF.csv files
#    
#    Arguments:
#        path = string (Defaults current directory if not specified)
#        file_name = string (Defaults to 'RDF' if not specified)
#        plot_name = string (Defaults to plot_name = file_name if not specified)
#        fig_size = tuple (int, int) (Size of plotting bok. Defaults to (15, 10))
#        fnt_size = int (font size of chart)
#        legend_keys = list [param1, param2, ...] (List of keys to include in legend)
#        color_scheme = dict (Defaults to None. format: {filename: fmt})
#        
#    Returns:
#        Void
#    """
#    legend_list = []
#    
#    plt.figure(figsize=fig_size)
#    # Uses file_name if plot_name not given
#    plt.title((lambda a, b: b if b is not None else a) (file_name, plot_name), fontsize=fnt_size*1.5) 
#    plt.ylabel('g(r)', fontsize=fnt_size*1.2)
#    plt.xlabel('r', fontsize=fnt_size*1.2)
#    plt.xticks(np.arange(0,10,0.5), fontsize=fnt_size)
#    plt.yticks(np.arange(0,10,1), fontsize=fnt_size)
#    
#    files = sorted([file for file in os.listdir(path) if file.startswith('RDF') and file.endswith('.csv')], key=lambda f: (f))
#    for i, file in enumerate(files):
#        if file.startswith('RDF.') and file.endswith('.csv'):
#            file_type = file.split('.')[1].replace('_', ' ').replace(' MP', '')
#            file_param = [var.split('_') for var in '.'.join(file.split('.')[2:-2]).split('--')]
#            leg_param = ',  '.join(pair[0]+': '+pair[1] for pair in file_param if pair[0] in legend_keys).strip(', ')
#            leg = file_type + ':  ' + leg_param
#            legend_list.append(leg)
#            #legend_list.append(file)
#            file = path + file
#            data = pd.read_csv(file)
#            for col in data.columns:
#                if 'g' in col:
#                    plt.plot(data['r'], data[col])#, color_scheme[i])
#    if 'r' in col:
#        plt.xlim((0, round(data['r'].iloc[-1], 2)))
#    plt.legend(legend_list, fontsize=fig_size[0])
#    plt.grid(b=True)
#    plt.savefig(file_name)
#    plt.show()
#    plt.close()
#
#    return

