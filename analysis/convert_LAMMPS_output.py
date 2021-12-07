# This code was provided by Christopher, and converts LAMMPS
# logs, dumps and fixes to csv format.
import pandas as pd
import numpy as np

def convert_log_to_csv(log_path):
    """Converts a LAMMPS log file given by log_path into a csv file.

    INPUT:
        log_path: string.
    """
    with open(log_path) as file:
        file_length = sum(1 for row in file) + 1

    with open(log_path) as file:
        row_number = 0
        boundaries = []
        rows = {}

        for row in file:
            row_number += 1

            if row.startswith('Per MPI rank memory allocation'):
                rows['over'] = row_number
            elif row.startswith('Loop time of'):
                rows['below'] = file_length - row_number
                boundaries.append(rows)
                rows = {}

    data_sets = []

    for boundary in boundaries:
        data_set = pd.read_csv(
            log_path,
            skiprows=boundary['over'],
            skipfooter=boundary['below'],
            delim_whitespace=True,
            engine='python'
        )
        data_sets.append(data_set)

    pd.concat(
        data_sets,
        keys=range(1, len(data_sets) + 1),
        names=['Run'],
        sort=False
    ).reset_index('Run').to_csv(f"{log_path}.csv", index=False)


def convert_fix_to_csv(fix_path):
    """Converts a LAMMPS fix file, given by fix_path, into a csv file.

    INPUT:
        fix_path: string
    """
    with open(fix_path) as fix_file, open(f"{fix_path}.csv", 'w') as csv_file:
        header = ''

        for row in fix_file:

            if row[0] == '#' and ' fix ' not in row:
                header += row[2:].replace(' ', ',').replace('-', '_').rstrip('\n') + ','
            elif row[0] != '#':
                subheader = row.rstrip('\n')
                break

        csv_file.write(header.rstrip(',') + '\n')

        for row in fix_file:

            if row[0] == ' ':
                csv_row = subheader + row[1:]
                csv_file.write(csv_row.replace(' ', ','))
            elif row[0].isdecimal():
                subheader = row.rstrip('\n')


def convert_dump_to_csv(dump_path, constant_volume=True, constant_atoms=True):
    """
        Converts a LAMMPS dump file, given 
        by dump_path, into a csv file.

    INPUT:
        fix_path: string
        constant_atoms: boolean
        constant_volume: boolean
    """
    header = ''

    with open(dump_path) as dump_file:

        for row in dump_file:

            if row.startswith('ITEM: TIMESTEP'):
                header += 'time_step,'
            elif row.startswith('ITEM: NUMBER OF ATOMS') and not constant_atoms:
                header += 'number_of_atoms,'
            elif row.startswith('ITEM: BOX BOUNDS') and not constant_volume:
                header += 'x_lo,x_hi,y_lo,y_hi,z_lo,z_hi,'
            if row.startswith('ITEM: ATOMS'):
                header += ','.join(row.split()[2:])
                break

    header += '\n'

    with open(dump_path) as dump_file, open(f"{dump_path}.csv", 'w') as csv_file:
        csv_file.write(header)
        flag = None

        for row in dump_file:

            if row.startswith('ITEM:'):
                pass
            elif flag == 'atoms':
                atoms = row.replace(' \n', '\n').replace(' ', ',')
                csv_file.write(time_step + number_of_atoms + box_bounds + atoms)
            elif flag == 'time_step':
                time_step = row.rstrip('\n') + ','
            elif flag == 'number_of_atoms' and not constant_atoms:
                number_of_atoms = row.rstrip('\n') + ','
            elif flag == 'box_bounds' and not constant_volume:
                box_bounds += row.replace(' ', ',').replace('\n', ',')

            if row.startswith('ITEM: ATOMS'):
                flag = 'atoms'
                atoms = ''
            elif row.startswith('ITEM: TIMESTEP'):
                flag = 'time_step'
                time_step = ''
            elif row.startswith('ITEM: NUMBER OF ATOMS'):
                flag = 'number_of_atoms'
                number_of_atoms = ''
            elif row.startswith('ITEM: BOX BOUNDS'):
                flag = 'box_bounds'
                box_bounds = ''


def extract_constants_from_log(log_path):
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
            variables[words[1]] = words[3]

    for variable, value in variables.items():
        if "^" in value:
            value = value.replace("^", "**")
        variables[variable] = eval(
            value, 
            {
                "PI":   np.pi,
                "INF":  np.inf,
            }
        )

    return variables
