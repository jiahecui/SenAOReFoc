"""
This document explains how to use the Python scripts for sensorbased AO using a Shack-Hartmann wavefront sensor 
within this folder

:Institute: University of Oxford, Department of Engineering Science, Dynamic Optics and Photonics Group
:Author: Jiahe Cui
:Email: jiahe.cui@eng.ox.ac.uk
"""

**Scripts included in this folder and their functions**

AO_slopes.py - Performs closed-loop AO correction via direct slopes
AO_zernikes.py - Performs closed-loop AO correction via Zernikes
calibration_zern.py - Generates DM control matrix
calibration.py - Performs DM calibration
centroid_acquisition.py - Performs centroiding of SH spots
conversion.py - Performs slope - zernike conversion
HDF5_dset.py - Methods for generating and appending data to HDF5 datasets
image_acquisition.py - Performs SHWS image acquisition
mirror.py - Performs DM initialisation
sensor.py - Performs SHWS initialisation
zernike.py - Generates Zernike polynomials and derivatives
config.py - Sets configuration file path
config.yaml - Configuration file for custom parameters used in each script (please change and add as needed)

**Notes for usage**

The main app script is included as an example of the hardware initialisation and data storage, as well as the signal - 
slot architecture used to run processes within independent scripts. According to specific needs, a custom GUI 
interface can be written to trigger signals or the software can be modified to work with no GUI interface.

Variables names throughout all scripts should be self explanatory, and dense comments are included for easy 
interpretation of the code. Changeable configuration parameters for SHWS, DM, closed-loop AO correction 
process, etc. are independenly specified in config.yaml and units / comments are provided for easy modification.

An HDF5_dset.py script is included as an example of how to store data using an HDF5 file. The centroid acquisition
process in centroid_acquisition.py needs to load images from the HDF5 file for fast access rather than directly passing
them into the method.

