"""
This document explains how to use the Python scripts for sensorbased AO and remote focusing using a Shack-Hartmann 
wavefront sensor within this folder

:Institute: University of Oxford, Department of Engineering Science, Dynamic Optics and Photonics Group
:Author: Jiahe Cui
:Email: jiahe.cui@eng.ox.ac.uk
"""

**Statement of need**

**Installation instructions**


**Scripts included in this folder and general descriptions**

app.py - Sets up GUI and event handlers, displays the GUI, instantiates devices such as DM and SHWS
AO_slopes.py - Performs closed-loop AO correction via direct slopes (zonal correction)
AO_zernikes.py - Performs closed-loop AO correction via Zernikes (modal correction)
calibration_RF.py - Performs remote focusing calibration using closed-loop AO correction
calibration_zern.py - Generates DM control matrix
calibration.py - Performs DM calibration
centroid_acquisition.py - Methods for centroiding of SH spots
centroiding.py - Performs system aberration calibration
config.py - Sets configuration file path
config.yaml - Configuration file for custom parameters used in each script (please change and add as needed)
conversion.py - Performs slope - zernike conversion
data_collection.py - Performs automated characterisation of microscope AO performance
HDF5_dset.py - Methods for generating and appending data to HDF5 datasets
image_acquisition.py - Methods for image acquisition
log.py - Method to set up event log
mirror.py - Performs DM initialisation
SB_geometry.py - Performs initialisation of search block geometry
SB_position.py - Performs repositioning of search blocks
sensor.py - Performs SHWS initialisation
SH_acquisition.py - Performs SHWS image acquisition
zernike.py - Generates Zernike polynomials and derivatives


**Functionality documentation**

The graphic user interface (GUI) exhibits modular widget arrangements and clear labelling to help the user navigate through
different software functions. A selection of interactive commands are also available for additional flexibility. General software
design concepts and an overview of the GUI are first given before describing the detailed functionalities of each modular unit.

SenAORefoc is developed under PySide2, which is a Python binding of the Qt application framework, and the GUI was
built using Qt Designer. Main processes are executed as separate worker threads from 'app.py' to allocate and recycle resources. 
Communication of events during processes are then handled using Qt's signals and slots mechanism. Common events 
between different threads, such as displaying an image or error message, reuse the same slots.

Hardware devices, such as the DM and SHWS are loaded upon software initialisation. In the case that no hardware devices 
are connected, the user can also choose to run the software in 'debug' mode to initialise dummy devices for development 
and test purposes. User controllable parameters can be freely accessed and modified in a separate configuration file 'config.yaml'
that is loaded upon software initialisation. Parameters that require user input after software initialisation can be modified 
from the GUI. Parameters calculated when running the software, as well as important result data, are grouped and saved 
in a separate HDF5 file that can be read with HDFView 3.0 (https://portal.hdfgroup.org/display/support/Download+HDFView).
Options are also available to export result data into .mat files for further analysis.

SenAORefoc consists of 5 main units, the SHWS initialisation and DM calibration unit, the Zernike aberration input unit, 
the AO control and data collection unit, the miscellaneous control unit, and the remote focusing unit. There are also 2 
auxiliary units, a message box which informs the user of required inputs and the current task progress, and a software 
termination module. Options are available to perform both direct slope (zonal) and Zernike (modal) control.

*SHWS initialisation and DM calibration unit*

[Initialise SB] determines the search block (SB) geometry given user informed parameters of the lenslet pitch (the diameter 
of one lenslet), the dimension and pixel size of the camera sensor, as well as the incident pupil diameter. Using this 
information, the software first calculates the number of pixels across one SB and the number of SBs across the camera sensor,
before determining the number and geometrical arrangement of the SBs within the pupil region of the incident beam. Then,
their reference centroid coordinates with regard to the camera sensor is calculated, before the full SB region is displayed on 
the GUI. This button also exercises the DM membrane by generating a user specified number of random patterns on the DM, 
which helps to remove hysteresis and creep of the membrane for magnetic mirrors upon initialisation.

[Position SB] repositions the SB region on the camera sensor such that it is concentric with the incident beam to allow for 
maximum dynamic range. This can be achieved in two ways depending on whether or not the DM requires recalibration. 
User input is needed at this stage by entering 'y' or 'n' on the keyboard. Upon entering 'y', the user can then reposition the
SB region by entering 'up', 'down', 'left', 'right' arrows on the keyboard, which will shift the entire SB region by one pixel at a
time, and display it at its new location. When the central SH spot(s) is centred within its corresponding SB, by pressing 'Enter'
on the keyboard, the user can confirm the new SB position and trigger the process of automatically calculating new SB 
reference centroid coordinates according to the number of pixels shifted in each direction. Upon entering 'n', SB and DM 
calibration related parameters from the last session will be automatically loaded from the HDF5 file. 

[Calibrate-Sys] records centroid coordinates of the SH spots within each SB using the CoG algorithm with either a DM flat file 
or a DM system flat file applied. These will be used as the actual reference centroids for aberration analysis. Due to the presence
of system aberrations, coordinates recorded in the former case are usually different to those of SB reference centroids (geometrical 
centre of SB).

[Calibrate-S] performs DM calibration to retrieve the control matrix (CM) for direct slopes (zonal) control. The positive and 
negative bias control voltages provided by the user are sequentially applied to each DM actuator to retrieve the 'x' and 'y'
slope values, which are then used to calculate the DM influence function (IF) matrix and slope CM by computing its 
pseudo-inverse.

[S-Z Conv] calculates the conversion matrix between raw slope values and Zernike coefficients. This can be used to calculate
the Strehl ratio (SR) using for evaluation of the image quality during direct slope (zonal) control, or to further acquire the DM
IF matrix and CM for Zernike (modal) control. 

[Calibrate-Z] retrieves the CM for Zernike (modal) control by first calculating the DM IF matrix using raw slope values acquired
in [Calibrate-S] and slope-Zernike conversion matrix calculated in [S-Z Conv]. Before performing pseudo-inverse of the DM 
IF matrix, the results after singular value decomposition are first evaluated to identify a cutoff value for small singular values.
This is usually selected as the value after which a significant drop is seen. System modes below this value are removed during 
pseudo-inverse for acquisition of a stable Zernike CM.

*Zernike aberration input unit*

The Zernike aberration input unit is mainly used for thorough characterisation of the system AO performance by specifying 
certain amounts of Zernike modes to generate on the DM membrane. Zernike modes are ordered according to the OSA/ANSI
standard and piston is removed such that mode 1 starts with tip. [Zernike mode spin box] allows the user to specify a single 
mode number, the amplitude of which is given in [Zernike value spin box] in the unit of microns. Mode combinations can be
specified in [Zernike array edit box] by providing an array of coefficients in the correct mode order. Irrelevant modes can be 
set as zero and only modes up to the maximum mode number controlled during AO correction (user-defined) can be generated.

*AO control and data collection unit*

This is the core software unit responsible for closed-loop AO control and automated data collection. Both Zernike (modal) 
and direct slopes (zonal) control can be performed in 4 different modes, which will be discussed below.

[Zernike AO 1] performs basic closed-loop Zernike (modal) AO control by updating DM control voltages with the Zernike CM
obtained in [Calibrate-Z]. Termination of the control loop is realised after the SR satisfies the Marechal criterion (SR > 0.8) 
or the number of iterations exceeds the maximum limit (user-defined). The same criteria is used for subsequent AO control 
modes. There are 2 sub-modes under this button that can be specified using mode flags in 'config.yaml'. Sub-mode 1 starts 
AO correction from a DM system flat file. Sub-mode 2 generates aberrations specified in the Zernike aberration input unit 
on the DM membrane before performing closed-loop AO correction. The latter sub-mode is especially useful for
characterisation of the system AO performance.

[Zernike AO 2] is a modified version of [Zernike AO 1] that takes into account missing or ill-formed spots (obscured SBs) 
detected by the SHWS due to arbitrarily shaped pupils [Dong2018OpticsExpress,Ye2015OpticsExpress,cui_j_2020_3885508].
In this case, Zernike polynomials which are defined over a unit circle lose orthogonality and original calibration matrices lead 
to unreliable interpretations of the wavefront. Therefore, a new set of orthogonal modes is established and an updatable 
control method is required to dynamically modify the CM during closed-loop AO correction. 

[Zernike AO 3] is a modified version of [Zernike AO 1] that neglects the correction of Zernike defocus such that refocusing 
of the focal spot is minimised. For Zernike (modal) control, this is simply achieved by setting the detected value of Zernike 
mode 4 to zero before updating the DM control voltages using the original Zernike CM obtained in [Calibrate-Z].

[Slope AO 1] performs basic closed-loop direct slopes (zonal) AO control by updating DM control voltages with the direct 
slope CM obtained in [Calibrate-S]. The 2 sub-modes described in [Zernike AO 1] also apply.

[Slope AO 2] is a modified version of [Slope AO 1] fulfilling the same purpose as [Zernike AO 2]. 

[Slope AO 3] is a modified version of [Slope AO 1] fulfilling the same purpose as [Zernike AO 3].

[Zernike Full] and [Slope Full] perform closed-loop AO control taking into account both obscured SBs and suppression of 
Zernike defocus at the same time.

[Data Collection] is designed to perform automated AO performance characterisations of the microscope system, 
though any data collection process that benefits from automation can be written beneath the hood. Each operation mode is
identified by a unique mode flag in 'config.yaml', which should be determined before software initialisation. Example
execution of integrated modes on a real microscope system can be found in the 'examples' folder.

*Miscellaneous control unit*

This unit controls independent image acquisition when no other background process thread is running, as well as 
miscellaneous functions that are used by multiple units.

Independent image acquisition can be performed in 3 modes. [Live Acq] continuously acquires images and displays them on
the GUI at the user-defined frame rate and exposure time until terminated by clicking the button a second time. [Burst Acq]
acquires a user-defined number of frames in burst mode before automatically terminating. [Single Acq] only acquires one 
image when clicked.

[Reset DM] resets the actuators to their neutral position. [Camera exposure time spin box] controls the exposure time of the
SHWS given in units of microseconds. [Maximum loop no. spin box] determines the maximum number of iterations before 
termination of the AO control loop.

*Remote focusing unit*

The remote focusing unit controls all parameters and procedures relevant to the calibration and execution of remote focusing
using a DM.

[CALIBRATE] triggers the calibration of control voltages that enable DM actuators to deform the membrane for fine axial 
refocusing. For details of the RF calibration process, please refer to [Cui2021BiophotonicsCongress, Cui2021OpticsExpress]. 
The software performs this as a 2-step process that proceeds along the negative direction before the positive. Messages are
displayed in the message box to inform users of the current calibration direction and step number, as well as to provide users
with options to: 1) confirm that the sample has been moved to the next calibration step; and 2) save or discard calibrated 
voltages and exit the thread. Commands can then be entered via the keyboard. A typical example of the interactive command 
message is: Move sample to positive position. Press [y] to confirm. Press [s] to save and exit. Press [d] to discard and exit.
The software automatically saves calibrated voltages to the HDF5 file after reaching the maximum user-defined step number, 
or does so on request at an arbitrary step. It also saves results of the slope values, Zernike coefficients, and SR acquired
before and after performing closed-loop AO correction at each calibration step for further analysis of the wavefront. 
After interpolation has been performed between voltages acquired at each calibration step to acquire those for finer axial 
steps, all RF voltages can then be loaded upon software initialisation.

[Scan Focus] controls whether remote focusing is to be performed at only one depth or at multiple depths. When unticked, 
only the first spin box on the left-hand-side panel is accessible while others are disabled. When ticked, the otherwise is true.

[MOVE] performs remote focusing at only one depth by first retrieving control voltages corresponding to that specified in 
[Focus Depth (microns)], and then applying those voltages on the DM to move the focus relative to the natural focal plane. 
Negative values move towards -z-axis and positive ones +z-axis.

[SCAN] performs remote focusing at multiple depths sequentially by applying different sets of voltages to the DM membrane
and scanning the focus depth-wise. [Step Increment (microns)] controls the displacement interval between consecutive
remote focusing planes, [Step Number] the number of planes to be accessed, [Start Depth (microns)] the depth at which to 
start the remote focusing process relative to the natural focal plane, and [Depth Pause Time (s)] the time for which the focal 
spot remains stable at each remote focusing plane. Entering a negative value for [Step Increment (microns)] permits RF in 
the -z-direction.

[AO Correction Type] allows the user to choose whether or not to apply AO correction at each remote focusing plane and 
which type to apply. Four options are available from the drop box [Zernike AO 3], [Zernike Full], [Slope AO 3], and [Slope Full],
the features of which have been discussed above. All options are applicable to the remote focusing process behind both 
[MOVE] and [SCAN], default is set to [None].

[Remote focusing toggle bar] permits random access of different remote focusing planes during imaging by simply dragging
the bar to a desired depth. Default is set to the natural focal plane and max/min limits are provided by the user according to 
the RF calibration results.


**Example usage**

An 

**Automated tests**


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

1. Bing Dong and Martin J. Booth, "Wavefront control in adaptive microscopy using Shack-Hartmann sensors with arbitrarily
    shaped pupils," Opt. Express 26, 1655-1669 (2018). https://doi.org/10.1364/OE.26.001655.
2. Jingfei Ye, Wei Wang, Zhishan Gao, Zhiying Liu, Shuai Wang, Pablo Benítez, Juan C. Miñano, and Qun Yuan, 
    "Modal wavefront estimation from its slopes by numerical orthogonal transformation method over general shaped aperture,"
    Opt. Express 23, 26208-26220 (2015). https://doi.org/10.1364/OE.23.026208.
3. Jiahe Cui, Bing Dong, and Martin J. Booth, "Shack-Hartmann sensing with arbitrarily shaped pupil (1.0)," 
    Zenodo (2020). https://doi.org/10.5281/zenodo.3885508.
4. Jiahe Cui, Raphaël Turcotte, Karen Hampson, Nigel J. Emptage, and Martin J. Booth, "Remote-Focussing for Volumetric 
    Imaging in a Contactless and Label-Free Neurosurgical Microscope," in Biophotonics Congress 2021, C. Boudoux, 
    K. Maitland, C. Hendon, M. Wojtkowski, K. Quinn, M. Schanne-Klein, N. Durr, D. Elson, F. Cichos, L. Oddershede, V. Emiliani, 
    O. Maragò, S. Nic Chormaic, N. Pégard, S. Gibbs, S. Vinogradov, M. Niedre, K. Samkoe, A. Devor, D. Peterka, P. Blinder, 
    and E. Buckley, eds., OSA Technical Digest (Optical Society of America, 2021), paper DTh2A.2. 
5. Jiahe Cui, Raphaël Turcotte, Nigel J. Emptage, and Martin J. Booth, "Extended range and aberration-free autofocusing via
    remote focusing and sequence-dependent learning," Opt. Express 29, 36660-36674 (2021). https://doi.org/10.1364/OE.442025.


