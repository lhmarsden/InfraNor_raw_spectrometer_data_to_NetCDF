#!/usr/bin/env python3
import xarray as xr
import os
import pandas as pd
import numpy as np
import yaml
import sys
from datetime import datetime
import argparse
from glob import glob
import logging
from functools import reduce

current_dir = os.path.dirname(os.path.abspath(__file__))
config_dir = os.path.join(current_dir, 'config')

valid_file_patterns = [
    'Incoming_radiance_FLUO_*.csv',
    'Incoming_radiance_FULL_F*.csv',
    'Reflectance_FLUO_*.csv',
    'Reflectance_FULL_F*.csv',
    'Reflected_radiance_FLUO_*.csv',
    'Reflected_radiance_FULL_F*.csv'
]


def all_valid_input_files(input_dir):
    '''
    Creates a dictionary of all valid input files, divided by data type.
    '''
    all_files = glob(os.path.join(input_dir, '**'), recursive = True)
    valid_files = {}
    for pattern in valid_file_patterns:
        pattern_start, pattern_end = pattern.split('*')
        valid_files[pattern_start] = []
        for filepath in all_files:
            filename = os.path.basename(filepath)
            if filename.startswith(pattern_start) and filename.endswith(pattern_end):
                valid_files[pattern_start].append(filepath)
    return valid_files


def load_variables_config():
    config_path = os.path.join(config_dir, 'variables.yaml')
    with open(config_path, 'r') as file:
        variables_config = yaml.safe_load(file)
    return variables_config


def load_global_attributes_dic():
    config_path = os.path.join(config_dir, 'global_attributes.yaml')
    with open(config_path, 'r') as file:
        global_attributes_dic = yaml.safe_load(file)
    return global_attributes_dic


def timestamp_to_ISO8601(datetime_obj):
    '''
    Converts from: 2019-06-23 18:52:04
    To: 2019-06-23T18:52:04Z
    '''
    iso8601_utc = datetime_obj.strftime('%Y-%m-%dT%H:%M:%SZ')
    return iso8601_utc


class Input_data():
    '''
    Each file contains one column for wavelength and an individual column for each time a measurement is taken
    Wavelength column: wl
    Other columns for each time a measurement is taken: HH_MM_SS
    '''

    def __init__(self,filepath):
        self.filepath = filepath
        self.read_csv()


    def read_csv(self):
        self.df = pd.read_csv(self.filepath, delimiter=';')


    def create_melted_df(self):
        file_date = self.filepath.split('/')[-2] # File date in YYMMDD
        date_formatted = datetime.strptime(file_date, '%y%m%d').strftime('%Y-%m-%d')
        # Melt the DataFrame to combine time columns into 'timestamp' and 'counts' columns
        self.melted_df = self.df.melt(id_vars='wl', var_name='timestamp', value_name='counts')
        self.melted_df['timestamp'] = date_formatted + ' ' + self.melted_df['timestamp'].str.replace('_', ':')
        # Replace non-numeric values with NaN
        self.melted_df['counts'] = pd.to_numeric(self.melted_df['counts'], errors='coerce')


class Variable_DF():
    '''
    Variable dataframe for combining data from all the CSV files for one variable
    '''
    def __init__(self):
        column_headers = [
            'timestamp',
            'wl',
            'counts'
            ]
        self.df = pd.DataFrame(columns=column_headers)


    def append_contributing_df(self, contributing_df):
        self.df = pd.concat([self.df, contributing_df], ignore_index=True)


    def time_seconds_since_epoch(self):
        # Calculate time differences in seconds from the epoch
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df['epoch_seconds'] = (self.df['timestamp'] - pd.Timestamp('1970-01-01')) // pd.Timedelta('1s')

    def rename_counts_column(self, name):
        self.df = self.df.rename(columns={'counts': name})

    def sort_dataframe(self):
        self.df = self.df.sort_values(by=['epoch_seconds', 'wl'], ascending=[True, True])

    # def reshape_for_2d_NetCDF_variable(self):
    #     # Reshape the data to be used for NetCDF
    #     self.df_for_2d_var = self.df.pivot(index='epoch_seconds', columns='wl', values='counts')



class NetCDF_file():

    def __init__(self,filepath,fluo_df,full_df):
        self.filepath = filepath
        self.fluo_df = fluo_df
        self.full_df = full_df


    def add_dimensions_and_variables(self, variables_config):
        '''
        Dimensions: wavelength, time
        Variables: wavelength, time, incoming radiance, reflectance and reflected radiance (FLUO and FULL)
        '''

        self.ds = xr.Dataset(
            coords={
                'time': sorted(self.fluo_df['epoch_seconds'].unique()),
                'wavelength_FLUO': sorted(self.fluo_df['wl'].unique()),
                'wavelength_FULL': sorted(self.full_df['wl'].unique())
            }
        )

        for variable, info in variables_config.items():
            if 'wavelength' in variable or variable == 'time':
                self.ds[info['variable_name']].attrs = info['variable_attributes']
            elif 'FLUO' in variable and 'wavelength':
                counts_data = self.fluo_df[variable]
                # Convert the 'counts' column to a 2D array
                pivot_df = self.fluo_df.pivot_table(index='epoch_seconds', columns='wl', values=variable)
                # Get a list of all unique values in the 'wl' column
                all_wl_values = self.fluo_df['wl'].unique()
                # Add missing columns (if any) to the pivot table
                missing_columns = [col for col in all_wl_values if col not in pivot_df.columns]
                for col in missing_columns:
                    pivot_df[col] = np.nan  # Fill the missing columns with nans
                counts_2D_array = pivot_df.to_numpy()
                self.ds[info['variable_name']] = (('time', 'wavelength_FLUO'), counts_2D_array)
                self.ds[info['variable_name']].attrs = info['variable_attributes']
            elif 'FULL' in variable:
                # Convert the 'counts' column to a 2D array
                pivot_df = self.full_df.pivot_table(index='epoch_seconds', columns='wl', values=variable)
                # Get a list of all unique values in the 'wl' column
                all_wl_values = self.full_df['wl'].unique()
                # Add missing columns (if any) to the pivot table
                missing_columns = [col for col in all_wl_values if col not in pivot_df.columns]
                for col in missing_columns:
                    pivot_df[col] = np.nan  # Fill the missing columns with nans
                counts_2D_array = pivot_df.to_numpy()
                self.ds[info['variable_name']] = (('time', 'wavelength_FULL'), counts_2D_array)
                self.ds[info['variable_name']].attrs = info['variable_attributes']


    def add_global_attributes(self, global_attributes_dic):
        self.ds.attrs = global_attributes_dic


    def add_global_attributes_from_data_or_code(self):
        combined_timestamps = pd.concat([self.fluo_df['timestamp'], self.full_df['timestamp']])
        self.ds.attrs['time_coverage_start'] = combined_timestamps.min().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.ds.attrs['time_coverage_end'] = combined_timestamps.max().strftime('%Y-%m-%dT%H:%M:%SZ')

        time_now = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        self.ds.attrs['date_created'] = time_now
        self.ds.attrs['history'] = f'{time_now}: File created using Python'


    def write_netcdf(self):
        self.ds.to_netcdf(self.filepath)


def main(input_dir, output_file):

    # Log to console
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    log_info = logging.StreamHandler(sys.stdout)
    log_info.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(log_info)

    logger.info("Loading in files from config subdir")
    variables_config = load_variables_config()
    global_attributes_dic = load_global_attributes_dic()

    logger.info("Creating a dictionary for the filepaths of all CSV files to include.")
    valid_files_dic = all_valid_input_files(input_dir)

    logger.info("Creating a list for variable dataframes - one dataframe per data variable.")
    variable_dfs_FLUO = []
    variable_dfs_FULL = []
    for data_type, valid_files_list in valid_files_dic.items():
        logger.info(f"\n\n***********\nStarting processing all {data_type} files\n***********\n")
        logger.info("Creating variable dataframe to append all data from CSV files to")
        variable_DF = Variable_DF()
        for valid_file in valid_files_list:
            logger.info(f"Processing {valid_file}")
            # Loading in the CSV file into a dataframe and processing it
            input_data = Input_data(valid_file)
            input_data.create_melted_df()
            variable_DF.append_contributing_df(input_data.melted_df)
        logger.info("Finished appending to variable dataframe")
        logger.info("Calculating time in seconds since 1970-01-01T00:00:00")
        variable_DF.time_seconds_since_epoch()
        logger.info(f"Renaming counts column to {data_type}")
        variable_DF.rename_counts_column(data_type)
        logger.info("Sorting dataframe in time, wavelength order")
        variable_DF.sort_dataframe()
        logger.info("Appending variable dataframe to list of dataframes")

        if 'FULL' in data_type:
            variable_dfs_FULL.append(variable_DF.df)
        elif 'FLUO' in data_type:
            variable_dfs_FLUO.append(variable_DF.df)

    logger.info("Merging the variable dataframes together into two master dataframes, one for FULL and one for FLUO")
    merged_FULL_df = reduce(lambda left, right: pd.merge(left, right, on=['epoch_seconds', 'timestamp', 'wl']), variable_dfs_FULL)
    merged_FLUO_df = reduce(lambda left, right: pd.merge(left, right, on=['epoch_seconds', 'timestamp', 'wl']), variable_dfs_FLUO)
    merged_FULL_df = merged_FULL_df.sort_values(by=['epoch_seconds', 'wl'], ascending=[True, True])
    merged_FLUO_df = merged_FLUO_df.sort_values(by=['epoch_seconds', 'wl'], ascending=[True, True])

    logger.info("Creating NetCDF file")
    # Creating NetCDF file
    netcdf = NetCDF_file(output_file, merged_FLUO_df, merged_FULL_df)
    logger.info("Adding dimensions, variables and variable attributes")
    netcdf.add_dimensions_and_variables(variables_config)
    logger.info("Adding global attributes from config file")
    netcdf.add_global_attributes(global_attributes_dic)
    logger.info("Adding global attributes from master dataframe or code")
    netcdf.add_global_attributes_from_data_or_code()
    logger.info("Writing out NetCDF file")
    netcdf.write_netcdf()
    logger.info(f"Created NetCDF file: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''
    Script to create CF-NetCDF files from CSV files for InfraNor raw spectrometer data.

    Should be called like this:
    python3 main.py -i /path/to/input_data/ -o /path/to/write/netcdf/files/

    1 NetCDF file is created for 1 year of data.
    ''')
    parser.add_argument('-i', '--input_dir', required=True, help='''
                        Path where the input files are stored.
                        The named path should include 1 subdirectory per day.

                        Each subdirectory should include:
                        Incoming_radiance_FLUO_%H%M%S.csv
                        Incoming_radiance_FULL_F%H%M%S.csv
                        Reflectance_FLUO_%H%M%S.csv
                        Reflectance_FULL_F%H%M%S.csv
                        Reflected_radiance_FLUO_%H%M%S.csv
                        Reflected_radiance_FULL_F%H%M%S.csv

                        Where %H%M%S is a 6 digit number representing hour, minute, second.
                        ''')
    parser.add_argument('-o', '--output_file', required=True, help='''
                        Full filepath (including filename) to where output NetCDF file will be written
                        ''')
    args = parser.parse_args()
    main(args.input_dir, args.output_file)
