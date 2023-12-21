# InfraNor raw spectrometer data to NetCDF

Python script to generate a NetCDF file from raw spectrometer data for the InfraNor project.

Run like this from your terminal when in the same directory:

python3 main.py -i /path/to/input_data/ -o /path/to/write/netcdf/files/

-i is the full filepath to where the input files are stored.
The named path should include 1 subdirectory per day.

Each subdirectory should include:
* Incoming_radiance_FLUO_%H%M%S.csv
* Incoming_radiance_FULL_F%H%M%S.csv
* Reflectance_FLUO_%H%M%S.csv
* Reflectance_FULL_F%H%M%S.csv
* Reflected_radiance_FLUO_%H%M%S.csv
* Reflected_radiance_FULL_F%H%M%S.csv

Where %H%M%S is a 6 digit number representing hour, minute, second.

-o is the full filepath (including file name) to where you want the NetCDF file to be written.

1 NetCDF file is created. I advise that you create 1 file per year of data.