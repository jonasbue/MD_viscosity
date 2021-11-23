#####################################################
# This file contains various utility functions      #
# used for array manipulation etc., and is used     #
# in many parts of the remaining code.              #
#####################################################

import numpy as np
import sys

def remove_nans(arr):
    """ Removes all instances of np.nan in arr. """
    return arr[~np.isnan(arr)]


def status_bar(i, n, l=30, fmt="percent"):
    p = int((i/n)*l)
    r = l-p-int(l/10)
    oo = " O o"
    if fmt=="percent":
        print(f"\r{100*i/n:.0f}\t%  ".expandtabs(2), end="")
    elif fmt=="arrow":
        print("\r[" + "="*p + ">" + " "*r + "]", end="")
    elif fmt=="train":
        if i == 0:
            print("\n\n\n")
        if i % 2:
            oo = "O Oo"
        print(f"""\033[A\033[A\033[A\033[A\033[A
            \r|{' '*p}____{oo}    {' '*r}|
            \r|{' '*p}|DD|____T_  {' '*r}|
            \r|{' '*p}|_ |_____|< {' '*r}|
            \r|{'_'*p}_@-@-@-oo\__{'_'*r}|\n""", 
        end="")


def get_avg(lower, upper):
    """ Returns an average (typically of slopes).
        Inputs:
            lower:  float.
            upper:  float.
        Outputs:
            avg:    float. The average of lower and upper.
    """
    return (np.abs(lower) + np.abs(upper))/2


def make_time_dependent(arr, t, number_of_chunks):
    """ Takes an array with multiple values for each timestep,
        and converts it to a 2D array in which first dimension
        is the time and second dimension contains the values
        at that time.
    """
    t = np.unique(t)

    # Ad hoc: With current data, two values are missing from first timestep.
    arr = np.insert(arr, 0, np.nan)
    arr = np.insert(arr, 0, np.nan)
    arr = np.reshape(arr, (len(t), number_of_chunks))
    return t, arr


def cut_time(cut_fraction, arr):
    """ Removes the first values of arr,
        to remove early values from the analysis.
    """
    if np.isscalar(arr):
        return arr
    cut = int(cut_fraction*len(arr))
    if "verbose" in sys.argv:
        print(f"Cutting the first {cut} values.")
    arr = arr[cut:]
    return arr
