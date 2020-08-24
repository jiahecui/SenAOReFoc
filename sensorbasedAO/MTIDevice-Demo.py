import numpy as np 
import mtidevice
from mtidevice import MTIError, MTIAxes, MTIParam, MTIDataMode, MTISync, MTIDataFormat
import TrackingExamples
import RGBExamples
from Utilities import getch, kbhit, clear_screen
import os
import sys
from time import sleep

def IsRGBModule( mti ):
    dataFormat = mti.GetDeviceParam( MTIParam.DataFormat )
    if (( dataFormat == MTIDataFormat.Reduced_XYRGB ) or ( dataFormat == MTIDataFormat.Standard_XYRGB ) or ( dataFormat == MTIDataFormat.Standard_XYMRGB )):
        return True
    else:
        return False

def main():
    # Create an instance of the MTIDevice class to access its methods,
    # attributes, and the target device at serial port
    mti = mtidevice.MTIDevice()

    portName = SelectIODevice(mti)
    if portName == None:
        return 0
    
    mti.ConnectDevice(portName)
    lasterror = mti.GetLastError()
    if lasterror != MTIError.MTI_SUCCESS:
        mti.SendSerialReset()
        print('Unable to connect with any device at port {}. Press any key to exit...'.format(portName))
        getch()
        return 0
    
    params = mti.GetDeviceParams()	# Get current info and parameters from controller
    rgbCapable = IsRGBModule( mti )
    # then apply any specific preferred parameters by replacing the existing values in the structure 
    # the first two parameters are critical to match device limitations provided in datasheet
    # VdifferenceMax should not exceed maximum voltage shown in page 1 of datasheet
    # HardwareFilterBw should not significantly exceed recommended filter setting in page 1 of datasheet
    params.VdifferenceMax = 177
    params.HardwareFilterBw = 2200

    # The rest of the parameters are more user-requirement specific
    params.DataMode = MTIDataMode.Sample_Output			# This is default mode, reiterated for demo
    params.DataScale = 0.8    							# Scale data/content to 80%
    params.SampleRate = 20000
    params.DeviceAxes = MTIAxes.Normal					# This is default mode, reiterated for demo
    params.SyncMode = MTISync.Output_DOut0				# This is default mode, reiterated for demo
    params.Vbias = 90

    mti.SetDeviceParams( params )	# Now, set the controller's params with the params in the structure
    lastError = mti.GetLastError()	# Query API handle for last error - helpful for debugging

    # Another option is to check for any parameters provided in an ini file such as mtidevice.ini
    # Following two lines load the device parameters from the ini file and updates the controller if valid
    lparams = mti.LoadDeviceParams("mtidevice.ini")
    mti.SetDeviceParams(lparams)

    # Now we can check if all the parameters we wanted are set on the controller
    params = mti.GetDeviceParams()		# Get current info and parameters from controller

    # Another way to apply parameters one at a time without the use of the MTIDeviceParams structures above is SetDeviceParam (NOT SetDeviceParams):
    mti.SetDeviceParam( MTIParam.OutputOffsets, 0, 0 )
    mti.SetDeviceParam( MTIParam.DataScale, 1 )
    mti.SetDeviceParam( MTIParam.DataRotation, 0 )

    # Now we move to the demo.
    # Recommended begginning by resetting device position to origin (data and offsets zeroed out)
    mti.ResetDevicePosition() 				# Send analog outputs (and device) back to origin in 25ms
    lastError = mti.GetLastError()			# User can check on error after every call to the API

    mti.SetDeviceParam( MTIParam.MEMSDriverEnable, True )		# Turn the MEMS Driver on for all examples below

    # Check if the Controller includes Tracking functionality (Tracking Bundle Add-On)
    trackingSupport = mti.IsTrackingSupported()
    if trackingSupport:
        # Set some tracking parameters that need to be default values for this demo
        tparams = mti.GetTrackParams()
        tparams.TangentialGain = 0
        tparams.HitRatio = 1
        mti.SetTrackParams( tparams )
        pass

    # try/except to contain the main loop functions
    # This is the recommended approach in the event that a function fails, the connection to the device may be properly closed.
    try:
        runFlag = True
        while ( runFlag ):
            clear_screen()
            print("\n************ MTIDevice-Demo 10.5 - Python SDK Examples ***************\n")
            print("** Warning: Examples should use proper limit for VdifferenceMax")
            print("** and HardwareFilterBw to avoid MEMS device damage.\n")
            print("General Examples:")
            print("\t0: Quit\n\t1: Point to Point Demo\n\t2: Scanning Demo\n\t3: Import File with Samples Demo\n\t4: Import File with Keypoints Demo")
            print("\t5: Import ILDA File") 
            print("\t6: Import File with Keypoints and Time Demo\n\t7: Slow raster Demo\n\t8: Follow WASD Keys Demo\n\t9: Read Analog Input Values Demo\n\tA: Read Analog Input Buffer Demo")
            print("\tB: Analog Input to Output DataMode Demo") 
            if ( trackingSupport ):
                print("\nTracking Examples:")
                print("\tC: Tracking with DataMode 'AutoTrack' (Simple Lissajous Pattern)")
                print("\tD: Tracking with DataMode 'AutoTrack' (Raster Pattern)")
                print("\tE: Tracking with DataMode 'Sample_And_Analog_Input_Track' (Raster)")
                print("\tF: Imaging with  DataMode 'Sample_And_Analog_Input_Track' (Raster)")

            mode = input('\n\nEnter menu option: ')
            clear_screen()
            mti.ResetDevicePosition() 				# Stop then send analog outputs (and device) back to origin in 25ms
            if (mode == '0'):
                runFlag = False
            elif (mode == '1'):
                if (rgbCapable):
                    PointToPointDemoRGB(mti)
                else:
                    PointToPointDemo(mti)
            elif (mode == '2'):
                if (rgbCapable):
                    ScannningDemoRGB(mti)
                else:
                    ScanningDemo(mti)
            elif (mode == '3'):
                ImportFileDemo(mti)
            elif (mode == '4'):
                KeyPointDemo(mti)
            elif (mode == '5'):
                if (rgbCapable):
                    ImportILDAFileDemoRGB(mti)
                else:
                    ImportILDAFileDemo(mti)
            elif (mode == '6'):
                ImportTimeFileDemo(mti)
            elif (mode == '7'):
                RasterDemo(mti)
            elif (mode == '8'):
                if (rgbCapable):
                    ArrowKeysDemoRGB(mti)
                else:
                    ArrowKeysDemo(mti)
            elif (mode == '9'):
                AnalogInputValues(mti)
            elif (mode == 'A'):
                AnalogInputBuffer(mti)
            elif (mode == 'B'):
                AnalogInputToOutputDemo(mti)
            elif (mode == 'C') and trackingSupport:
                SinglePointTracking_Lissajous(mti)
            elif (mode == 'D') and trackingSupport:
                SinglePointTracking_Raster(mti)
            elif (mode == 'E') and trackingSupport:
                DataMode7Raster(mti)
            elif (mode == 'F') and trackingSupport:
                Imaging(mti)
    except:
        CloseDeviceConnection(mti)
        print(sys.exc_info()[1]) # Print the last error from user's sys
        return 0

    # End the program by returning the device to origin
    CloseDeviceConnection(mti)
    return 0

def CloseDeviceConnection(mti):
    """
    Safely close connection to MTIDevice
    Resets device position, disables MEMSDriver, and disconnects device
    """
    mti.ResetDevicePosition()
    mti.SetDeviceParam(MTIParam.MEMSDriverEnable, False)
    mti.DisconnectDevice()
    del mti

def SelectIODevice(mti):
    """
    Function to handle user's choice of a device at the beginning of the demo
    Utilizes MTIDevice.GetAvailableDevices, MTIDevice.ListAvailableDevices
    """
    clear_screen()
    print('Searching for available Mirrorcle MEMS Controller devices. Please wait...\n')
    # Call to MTIDevice.GetAvailableDevices returns table of devices with their respective COM ports
    table=mti.GetAvailableDevices() # Also prints this table to the command line.
    if table.NumDevices == 0:
        print('There are no devices available. Press any key to exit...')
        getch()
        return None
    mti.ListAvailableDevices(table) # Print the MTIAvailableDevices* table to terminal
    print('\n')
    portnumber = input('Input serial port number to connect to (e.g. \'99\' for COM99): ')
    if os.name == 'nt':
        portnumber = 'COM' + portnumber
    else:
        portnumber = '/dev/ttyUSB' + portnumber
    return portnumber

def PointToPointDemo(mti):
    """
    PointToPointDemo demonstrates MTIDevice's GoToDevicePosition and SendDataStream methods for Point-To-Point beam steering.
    Loop and ask user which position to go to.  If position exceeds -1 to 1, exit program
    """
    tStep = 10
    fastestStep = False

    print("\nInput position the device should step and settle to.\n")
    print("Use normalized positions from -1 to 1 for each axis\n")
    print("To EXIT, enter any value out of the valid range\n\n")
    
    # First get the desired step time between last point and the next point
    line = input("Choose step time in ms (0 for fastest step, >=5ms for timed step): ")

    # Input handling to ensure user inputs int
    if all([type(el)==int for el in line]):
        tStep = int(line)
    if (tStep == 0):
        fastestStep = True
    tStep = max( tStep, 5 )

    x, y = 0, 0
    m = 255	# Digital output is set to this value (255) during movement and at final point

    # Define X/Y/M data containers
    x1, y1, m1 = np.ndarray(1), np.ndarray(1), np.ndarray(1)
    x1[0], y1[0], m1[0] = x, y, m

    mti.ResetDevicePosition() 	# Send analog outputs (and device) back to origin in 25ms.
    runFlag = True
    while ( runFlag ):
        if (fastestStep):
            x1[0], y1[0], m1[0] = x, y, m
            mti.SendDataStream( x1, y1, m1, 1 )
        else:
            mti.GoToDevicePosition( x, y, m, tStep )
        x, y = -10, -10
        line = input("Input go to position in X,Y format: ")

        if (len(line) >= 3 and ',' in line):
            x,y = float(line.split(',')[0]),float(line.split(',')[1])
            runFlag = abs(x) <= 1 and abs(y) <= 1
        else:
            runFlag = False
    mti.ResetDevicePosition() 				# Send analog outputs (and device) back to origin in 25ms.

def ScanningDemo(mti):
    """
    ScanningDemo demonstrates basic content generation and execution functions
    This demo prepares a lissajous pattern, then sends the data to the Controller
    """
    i, j, k = 0, 0, 0 
    npts = 256*40
    dt = np.pi * 2 / npts

    spshold = mti.GetDeviceParam( MTIParam.SampleRate ) # Store the sample rate in order to restore at the end of the demo.
    mti.SetDeviceParam ( MTIParam.SampleRate, npts )	# Sample rate and number of points equal, so 1 second of data in one repeated frame.
    mti.SetDeviceParam ( MTIParam.DigitalOutputEnable, 1 ) # Ensure that digital output enable is on for the digital port to output "m" values.
    
    print( "\nStarting scanning demo...\n\n" )

    # Create some sample data
    x, y, m = np.ndarray(npts, dtype=np.float32), np.ndarray(npts, dtype=np.float32), np.ndarray(npts, dtype=np.uint8)
    user_input = ''
    while ( user_input != chr(27) ):
        k = j % 8 + 1  # Integer that changes every iteration to generate different Lissajous patterns.
        # Prepare 1 second of data to be repeated
        for i in range(0, npts):
            x[i] = (np.sin( 10 * k * i * dt ))						# X-Axis position follows a sin curve from -1.0 to +1.0 normalized position (* DataScale setting)
            y[i] = 0.9 * np.cos((5 * (k + 1) + 1) * i * dt )		# Y-Axis position follows a cos curve from -0.9 to +0.9 normalized position at another frequency
            m[i] = (i/4)%256							# Digital signals P0.0 to P0.7 doing 8-bit counter toggles every four samples
        mti.SendDataStream( x, y, m, npts, 0, False )				# This call will download the buffer of data to the controller and run it when existing frame ends
        print("Cycle: {}. Press any key to change waveform or press ESC to leave demo...\n".format(j))
        user_input = getch()
        j = j+1 

    # Example of how to save the current state into flash - and command to AutoRun the same pattern and settings on boot
    # mti.SaveToFlash( MTIFlash.Device_Params )
    # mti.SaveToFlash( MTIFlash.Data_In_Buffer )
    # mti.SetDeviceParam( MTIParam.BootSetting, MTIBoot.Boot_With_Flash_Data_And_Autorun )

    mti.ResetDevicePosition() 									# Send analog outputs (and device) back to origin in 25ms
    mti.SetDeviceParam ( MTIParam.SampleRate, spshold )			# Now restore the sample rate value for other demos

def ImportFileDemo(mti):
    """
    ImportFileDemo demonstrates the use of the MTIDataGenerator class to load user-defined
    point files (in x/y/m format).

    The demo loads the butterfly.smp point file, then sends it directly to the Controller for display
    """
    sps, npts = int(), int()
    spshold = mti.GetDeviceParam( MTIParam.SampleRate )	# Preserve sample rate from other examples to restore at the end of the demo.
    datagen = mtidevice.MTIDataGenerator()

    # Imported *.smp or *.txt file should have 3 columns.  
    # First 2 columns are floating point numbers for X and Y, and 3rd column is uint8 0 (laserOff) or 1-255 (laserOn).
    npts = datagen.LoadPointFile( "butterfly.smp" ) # Read the point file from 
    print("\n{} XYM points read from the \"butterfly.smp\" file\n".format(npts))
    sps = datagen.GetPointFileSps() # Grab the sample rate from the .smp file
    x = np.ndarray(npts, dtype=np.float32)
    y = np.ndarray(npts, dtype=np.float32)
    m = np.ndarray(npts, dtype=np.uint8)
    npts, x, y, m = datagen.PointFileDataStream() # Copy DataGenerator X/Y/M data read from the file to local variables x/y/m a to send to device.

    if (sps>0):
        print("\nSample rate of {} specified in the file\n\n".format(sps))
    else:
        print("Sample file did not specify SPS setting.  Please enter desired sample rate.\n")
        print("Note that your sample rate should be same or lower as used")
        print("to create sample data to avoid potential device damage.")
        sps = int(input("\nEnter SPS setting (e.g. 10000): "))
    limits = mti.GetDeviceParams ()
    sps_max = limits.DeviceLimits.SampleRate_Max # Retrieve the SampleRate min/max from controller
    sps_min = limits.DeviceLimits.SampleRate_Min

    # Ensure that the sample rate from the .smp does not exceed the device limits
    if( sps > sps_max or sps <= sps_min ):
        print("Invalid sample rate. Defaulting to 5000\n")
        sps=5000

    mti.SetDeviceParam( MTIParam.SampleRate, sps ) # Set the sample rate to that specified by .smp
    mti.SendDataStream( x, y, m, npts, 0, False ) # Send the data stream read from .smp to the device

    print("Press any key to stop the waveform and device scanning...\n")
    getch()
    mti.ResetDevicePosition() 				# Stop then send analog outputs (and device) back to origin in 25ms.
    mti.SetDeviceParam( MTIParam.SampleRate, spshold )	# Restore sample rate from other examples.

def KeyPointDemo(mti):
    """
    KeyPointDemo loads a set of keypoints from a file 'KeypointFileExample.kpt' which define
    desired device trajectory.

    File must have 3 columns for X, Y, and M (modulation or blanking).
    X and Y range from -1 to 1. M is 0 to 255 and may represent laser modulation (brightness)
    or a combination of digital output triggers.

    This example then calls MTIDataGenerator's InterpolateData method to create a complete
    list of samples of appropriate length to send to the Controller.
    """

    datagen = mtidevice.MTIDataGenerator() # For this demo, we will be using the MTIDataGenerator class

    i, j, nKey, nSample = int(), int(), int(), int()
    rr = 20
    
    # Set a basic sample rate for output
    mti.SetDeviceParam ( MTIParam.SampleRate, 60000 )			# Sample rate and number of points equal, so 1 second of data in one repeated frame.

    # Read in keypoints from 'KeypointFileExample.kpt'
    nKey = datagen.LoadPointFile( "KeypointFileExample.kpt" )
    print("\nRead {} X/Y/M keypoints from the \"KeypointFileExample.kpt\" file\n".format(nKey))
    xKey = np.ndarray(nKey*2, dtype=np.float32)
    yKey = np.ndarray(nKey*2, dtype=np.float32)
    mKey = np.ndarray(nKey*2, dtype=np.uint8)

    nKey, xKey, yKey, mKey = datagen.PointFileDataStream() # Read keypoints from MTIDataGenerator object to local variables
    nKey, xKey, yKey, mKey = datagen.CloseCurve( xKey, yKey, mKey, nKey, 1, False )
    
    # Next step is interpolation to create a full list of samples to cover the keypoints and space inbetween.
    # Specify how many points you want back from the InterpolateData function. These will be the actual interpolated sample points.
    # For example: if you want refresh rate 12.5Hz and have 25000 sample rate (sps) you need 25000/12.5 = 2000 points.
    rr = float(input( "Enter refresh rate: " ))
    nSample = int(np.floor( mti.GetDeviceParam( MTIParam.SampleRate ) / rr ))
    xSample = np.ndarray(nSample*3, dtype=np.float32)
    ySample = np.ndarray(nSample*3, dtype=np.float32)
    mSample = np.ndarray(nSample*3, dtype=np.uint8)

    dOn		= .08	# Proportional accel distance for laser-on segments: 0.05 to 0.2 are recommended.
    dOff	= .50	# Proportional accel distance for laser-off segments: best results with 0.5 | Numbers >0.5 will make a mess as you need at least as much time to deccelerate as accelerate
    npts, xSample, ySample, mSample = datagen.InterpolateData( xKey, yKey, mKey, nKey, nSample, 1.5, dOn, dOff )

    mti.SendDataStream( xSample, ySample, mSample, npts, 0, False )
    print("\nPress any key to start offset scanning...")
    getch()
    for i in range(1):
        for j in range(1000):
            mti.SetDeviceParam( MTIParam.OutputOffsets, float(j-500)/500, float((j%320)-160)/160 )
            sleep(.002)

    mti.SetDeviceParam( MTIParam.OutputOffsets, 0.0, 0.0 ) # Remove any offsets.  MTIResetDevicePosition also removes offsets so in this case it is not necessary.

    print("\nPress any key to start data scale scanning...")
    getch()
    for j in range(100):
        mti.SetDeviceParam( MTIParam.DataScale, float(j)/100 )
        sleep(.002)

    for j in range(100,-1,-1):
        mti.SetDeviceParam( MTIParam.DataScale, float(j)/100 )
        sleep(.002)

    mti.SetDeviceParam( MTIParam.DataScale, 1.0 ) # Remove any datascale.
    print("\nPress any key key to start data rotating...")
    getch()

    for j in range(361):
        angleRad = float((j/1)*np.pi/180)
        mti.SetDeviceParam( MTIParam.DataRotation, angleRad ) # Rotation in radians.
        sleep(.001)

    mti.SetDeviceParam( MTIParam.DataRotation, 0 ) # Remove any rotation

    print("\nPress any key to stop the waveform and device scanning...")
    getch()
    mti.ResetDevicePosition() 				# Stop then send analog outputs (and device) back to origin in 25ms.

def ImportILDAFileDemo(mti):
    """
    ImportILDAFileDemo demonstrates the user of the MTIDataGenerator class to load content
    from the commonly-encountered laser display format "ILDA"

    More info on the ILDA format can be found here: https://www.ilda.com/resources/StandardsDocs/ILDA_IDTF14_rev011.pdf
    """
    npts = 50000 # Start with a large estimate of points to allocate plenty of memory
    nkpts = 50000

    datagen = mtidevice.MTIDataGenerator() # Instantiate MTIDataGenerator object for access to wide variety of data generation methods.

    datagen.LoadIldaFile("CanGoose.ild") # Call MTIDataGenerator's LoadIldaFile method to load the desired ILDA file
    print("\n\nLoaded ILDA file \"CanGoose.ild\" \n")
    
    spshold = mti.GetDeviceParam( MTIParam.SampleRate )	# Preserve sample rate from other examples to restore at the end of this demo.
    sps = 10000 
    mti.SetDeviceParam( MTIParam.SampleRate, sps )		# Now assert the desired sample rate for this demo, e.g. 10000

    fileType = 0 			# 0 = *.ild files | 1 = *.kpt files | 2 = *.smp files
    rr = 15
    oor = 2.0
    onFrac = 0.10
    animTime = 1.0
    animType = 0
    frameNum = 1.0
    camTheta = 0.0
    camPhi = np.pi/2		# ILDA file settings p3 = Camera Theta, p4 = CameraPhi
    theta = 0.0
    curveType = 3 			# 3 for Imported Files

    # Prepare data from the loaded ILDA file, only one frame (under frameNum).
    npts = datagen.CurvesDataSize( 
                curveType, sps, rr, animType, animTime, False,
                fileType, 0, 0, frameNum, camTheta, camPhi, "", 0)

    # Allocate memory for arrays based on estimated number of points * 2 for safety.
    xKey = np.ndarray(npts*2,dtype=np.float32)
    yKey = np.ndarray(npts*2,dtype=np.float32)
    mKey = np.ndarray(npts*2,dtype=np.uint8)
    xSample = np.ndarray(npts*2,dtype=np.float32)
    ySample = np.ndarray(npts*2,dtype=np.float32)
    mSample = np.ndarray(npts*2,dtype=np.uint8)

    # Function will provide key data (xKey...) and sample data (xSample...) based on previously loaded file (ILDA, kpt, smp) and various inputs.
    # It will return by reference the data to the pointers, as well as nkpts and npts so user knows how many points there really are.
    nkpts, npts = datagen.GenerateImportFileData( xKey, yKey, mKey, xSample, ySample, mSample, sps, rr, True, False, oor, onFrac, animType, animTime, 0, 
        fileType, 1.0, frameNum, camTheta, camPhi, theta, 0.0, 0.0, False, 0.0, 0.0 )
    
    print("Press any key to display the first frame of the ILDA file...\n")
    getch()
    mti.SendDataStream( xSample, ySample, mSample, npts, 0, False )

    print("Press any key to begin the ILDA file animation...\n")
    getch()
    # Prepare data from loaded ILDA file, animate all frames.
    animType = 4
    npts = datagen.CurvesDataSize( 
                curveType, sps, rr, animType, animTime, False,
                fileType, 0, 0, frameNum, camTheta, camPhi, "", 0)

    # Allocate memory for arrays based on the estimated number of points * 2 for safety.
    xKey = np.ndarray(npts*2,dtype=np.float32)
    yKey = np.ndarray(npts*2,dtype=np.float32)
    mKey = np.ndarray(npts*2,dtype=np.uint8)
    xSample = np.ndarray(npts*2,dtype=np.float32)
    ySample = np.ndarray(npts*2,dtype=np.float32)
    mSample = np.ndarray(npts*2,dtype=np.uint8)

    # Function will provide key data (xKey...) and sample data (xSample...) based on previously loaded file (ILDA, kpt, smp) and various inputs.
    # It will return by reference the data to the pointers, as well as the number of keypoints (nkpts) and number of points (npts) so the user knows how many points there really are.
    nkpts, npts = datagen.GenerateImportFileData( xKey, yKey, mKey, xSample, ySample, mSample, sps, rr, True, False, oor, onFrac, animType, animTime, 0, 
        fileType, 1.0, frameNum, camTheta, camPhi, theta, 0.0, 0.0, False, 0.0, 0.0 )
    mti.SendDataStream( xSample, ySample, mSample, npts, 0, False )

    print("Press any key to stop the waveform and device scanning...\n")
    getch()
    mti.ResetDevicePosition() 				# Stop then send analog outputs (and device) back to origin in 25ms
    mti.SetDeviceParam( MTIParam.SampleRate, spshold )	# Restore sample rate from other examples

def ImportTimeFileDemo(mti):
    """
    ImportTimeFileDemo demonstrates how to import and use a PointTimeFile
    
    PointTimeFile is similar to a PointFile, however it contains the time spent
    at a position as the fourth column (e.g. x/y/m/t)
    """
    stepTime = 10	# Step time (in milliseconds) to move from one point to the next point
    spshold = mti.GetDeviceParam( MTIParam.SampleRate )	# Preserve sample rate from other examples to restore at the end.
    datagen = mtidevice.MTIDataGenerator()
    nSample = 100000
    # Imported *.smp or *.txt file should have 3 columns.  First 2 columns are floating point numbers for X and Y, and 3rd column is unsigned char 0 (laserOff) or 1-255 (laserOn).
    npts = datagen.LoadPointTimeFile( "PointTimeExample.txt" )
    print("\n{} XYM points read from the \"PointTimeExample.txt\" file\n".format(npts))
    sps = datagen.GetPointTimeFileSps()
    
    npts, xKey, yKey, mKey, tKey = datagen.PointTimeFileDataStream()
    Limits = mti.GetDeviceParams ()
    sps_max = Limits.DeviceLimits.SampleRate_Max
    sps_min = Limits.DeviceLimits.SampleRate_Min
    if (sps<=0):
        print("Sample file did not specify SPS setting. SPS will be automatically calculated...\n")
    
    xSample = np.ndarray(nSample, dtype=np.float32)
    ySample = np.ndarray(nSample, dtype=np.float32)
    mSample = np.ndarray(nSample, dtype=np.uint8)
    
    # PointToPointPattern modifies the pre-allocated arrays xSample, ySample, and mSample in-place
    nSample, sps = datagen.PointToPointPattern(xSample, ySample, mSample, xKey, yKey, mKey, tKey, npts, stepTime, sps, sps_min, sps_max )

    mti.SetDeviceParam(MTIParam.SampleRate, sps )
    mti.SendDataStream( xSample, ySample, mSample, nSample, 0, False )
    print("\nPerforming Point to Point Pattern Scan with Step time of {}ms \nand Sample rate of {}\n\n".format(stepTime, sps))
    print("Press any key to stop the waveform and device scanning...\n")
    getch()
    mti.ResetDevicePosition() 				# Stop then send analog outputs (and device) back to origin in 25ms.
    mti.SetDeviceParam( MTIParam.SampleRate, spshold )	# Restore sample rate from other examples.

def RasterDemo(mti):
    """
    RasterDemo demonstrates how to quickly generate highly-customizable raster pattern
    using MTIDataGenerator.LinearRasterPattern
    """
    datagen = mtidevice.MTIDataGenerator()		# Handle for MTIDataGenerator class with a wide variety of content generation methods.

    # Following is a list of parameters user chooses to create a raster pattern of linearly spaced, uniform velocity or point-to-point rastering lines
    sps = 2000				# Sample rate (samples per second/sps) setting
    xAmp = 0.9				# Amplitude for X-Axis scan (0.0 to 1.0).
    yAmp = 0.75				# Amplitude for Y-Axis scan (0.0 to 1.0).
    numLines = 10			# Number of lines that are designed for uniform velocity or point-to-point motion. Does not include retrace lines if bidirF=0.
    numPixels = 10			# Number of stop-and-go locations along a line in ppRaster=1 mode, or number of synchronous digital output shots along a line.
    lineTime = 0.1			# Duration of time allocated to each raster line in seconds.
    ppMode = True			# False = uniform velocity line raster | True = Point-to-point raster with near stopping at each point.
    retrace = True			# False = Fast axis uniform scan in one direction, then fast retrace back | True = Fast axis uniform scan in both directions.
    triggerShift = 0		# Number of samples the synchronous digital output should be rotated with respect to the analog XY output (can be negative or positive).
    theta = np.pi / 2		# Rotation angle for the raster pattern (looking toward display surface). Default results in vertical lines which move from left to right on the display.

    params = mti.GetDeviceParams()
    spsMax = params.DeviceLimits.SampleRate_Max
    spsMin = params.DeviceLimits.SampleRate_Min
    npts = params.DeviceLimits.SamplesPerFrame_Max	# Number of points (samples) to send to the device - starting guess is maximum allowed by DeviceLimits
    
    xData = np.ndarray(npts, dtype=np.float32)
    yData = np.ndarray(npts, dtype=np.float32)
    mData = np.ndarray(npts, dtype=np.uint8)

    try:
        # This is the the primary function - LinearRasterPattern creates a full waveform for the raster based on the given parameters
        # LinearRasterPattern populates xData, yData, and mData in-place | These arrays must be pre-allocated
        npts, sps = datagen.LinearRasterPattern( xData, yData, mData, xAmp, yAmp, numLines, numPixels, lineTime, ppMode, retrace, triggerShift, theta, sps, spsMin, spsMax )
        # Modified sample rate returned from the function to give us desired lineTime and numPixels, now set that rate for the Controller
        spshold = mti.GetDeviceParam( MTIParam.SampleRate )		# Preserve the Controller's currrent sample rate to restore later.
        mti.SetDeviceParam( MTIParam.SampleRate, sps )

        print("\nPrepared a raster scan with {} lines x {} pixels = {} points.".format(numLines, numPixels, npts))
        print("\nScan is running with a {} samples per second rate.".format(sps))
        print("\nEach line has {}s active duration and overall scan repeats every {}s.\n\n".format(lineTime, (npts/sps)))

        mti.SendDataStream( xData, yData, mData, npts, 0, True )
        datagen.ExportFile("rasterexport.txt", xData, yData, mData, npts, sps)
        print("Press any key to stop the waveform and device scanning...\n")
        getch()
        mti.ResetDevicePosition()	# Stop then send analog outputs (and device) back to origin in 25ms
        mti.SetDeviceParam( MTIParam.SampleRate, spshold )	# Restore the sample rate that the Controller had before this function
    except:
        print("Error in Raster Demo - Press any key to return to the main menu.\n")
        getch()

def ArrowKeysDemo(mti):
    """
    ArrowKeysDemo uses the MTIParam OutputOffsets to quickly send the (x,y) coordinates
    received from user WASD-key input to the Controller and, in-turn, the MEMS
    """
    runFlag = True
    x = 0
    y = 0

    clear_screen()
    print("Use WASD Keys to Control Device Tip / Tilt Angle.")
    print("A and D Keys Control X-Axis.")
    print("W and S Keys Control Y-Axis.")
    print("Hit ESC to exit this mode\n")
    print("\n\n\t\tCurrent X and Y position [-1 to +1]: {}, {}".format( x, y ))
    # Ensure that digital output enable is on for the digital port to output "m" values
    mti.SetDeviceParam ( MTIParam.DigitalOutputEnable, 1 )
    mti.GoToDevicePosition( 0, 0, 255, 5 )	# Move to origin and output 255 on digital output port and HIGH on Sync

    while( runFlag ):
        if( kbhit() ):
            key = getch()
            if key == chr(97):		# chr(97): 'a'
                x -= 0.1 
                x = max( x, -1 ) 
            elif key == chr(100): 	# chr(100): 'd'
                x += 0.1
                x = min( x, 1 )
            elif key == chr(115):	# chr(115): 's'
                y -= 0.1 
                y = max( y, -1 )
            elif key == chr(119):	# chr(119): 'w'
                y += 0.1
                y = min( y, 1 ) 
            elif key == chr(27): 	# chr(27):  ESC
                runFlag = False
                continue
            x = round(x, 1)
            y = round(y, 1)
            mti.SetDeviceParam ( MTIParam.OutputOffsets, x, y )
            print("\n\n\t\tCurrent X and Y position [-1 to +1]: {}, {}".format( x, y ))
    mti.ResetDevicePosition() 				# Stop then send analog outputs (and device) back to origin in 25ms
    # Note that the reset also sent 0 to digital output port and LOW to sync (which may turn off connected laser)

def AnalogInputValues(mti):
    """
    AnalogInputValues demonstrates the use of GetAnalogValues to sample the Analog Input values
    at the Analog connectors at that point in time
    """
    clear_screen()
    print("Press any key to get analog input values\n")
    print("from the device for channels AI0 and AI1.\n")
    print("Hit ESC to exit this mode\n")

    runFlag = True
    while runFlag:
        key = getch() 
        AI0 = mti.GetAnalogInputValue( 0 )
        AI1 = mti.GetAnalogInputValue( 1 )
        print("\n\nAI0 Voltage = {:0.2f}   AI1 Voltage = {:0.2f}".format(AI0,AI1))
        if key == chr(27):
            runFlag = 0
    mti.ResetDevicePosition() 				# Stop then send analog outputs (and device) back to origin in 25ms

def AnalogInputBuffer(mti):
    """
    AnalogInputBuffer demonstrates the use of both the MTIDataMode Sample_And_Analog_Input_Buffer
    and GetAnalogInputBuffer

    GetAnalogInputBuffer will sample the Analog Input values to a depth of the specified npts
    """
    datagen = mtidevice.MTIDataGenerator()
    npts = 2500
    dt = np.pi * 4 / npts	# Create data for 2 sinusoidal cycles

    x = np.ndarray(npts, dtype=np.float32)
    y = np.ndarray(npts, dtype=np.float32)
    m = np.ndarray(npts, dtype=np.uint8)
    
    for i in range(npts):
        x[i] = 0.9 * np.sin( 2 * i * dt )	# X-Axis position follows a sin curve from -0.8 to +0.8 normalized position
        y[i] = 0.9 * np.cos( 3 * i * dt )	# Y-Axis position follows a cos curve from -0.5 to +0.5 normalized position at 4x freq of X axis
        m[i]= (i/2)%256						# Digital signals P0.0 to P0.7 doing 8-bit counter toggles every sample

    mti.SetDeviceParam( MTIParam.DataMode, MTIDataMode.Sample_And_Analog_Input_Buffer )

    clear_screen()
    print("Device will output a waveform and store analog inputs AI0 and AI1 in a buffer")
    print("during the output.\n")
    print("Press any key to execute a single run.  Analog input data will be stored into")
    print("an ASCII file Export.txt in first two columns.\n")
    getch()
    
    mti.SendDataStream( x, y, m, npts, 0, False ) 
    mti.StartDataStream ( 1 )
    ai0 = np.ndarray(npts, dtype=np.float32)
    ai1 = np.ndarray(npts, dtype=np.float32)
    ai0, ai1 = mti.GetAnalogInputBuffer( npts )
    
    # Store AI0 and AI1 values (float arrays) into an ASCII file in the first two columns.
    datagen.ExportFile( "Analog_Buffer_Export.txt", ai0, ai1, m, npts, int(mti.GetDeviceParam( MTIParam.SampleRate )))	# M values are stored in 3rd column. Not used here so we write dummy values.
    mti.SetDeviceParam( MTIParam.DataMode, MTIDataMode.Sample_Output )
    mti.ResetDevicePosition()

def AnalogInputToOutputDemo(mti):
    """
    AnalogInputToOutputDemo demonstrates how to switch between a Controller's various
    data modes (MTIDataMode).

    MTIDataMode.Analog_Input_To_Output allows the user to drive the MEMS via two analog
    input voltages to the Analog port of a USB Controller
    """
    # Set any MEMS Driver Settings for Analog Input to Output Mode
    sParams = mti.GetDeviceParams()	# Back up current params from controller
    tParams = mti.GetDeviceParams()	# Store new temporary params in tParams
    # Set the Driver Settings
    tParams.DataScale = 1.0
    tParams.Vbias = 70
    tParams.VdifferenceMax = 100
    tParams.HardwareFilterBw = 200
    tParams.SampleRate = 50000
    mti.SetDeviceParams( tParams )

    clear_screen()
    print("Device will read analog inputs and convert the signals to MEMS drive voltages\n")
    print("Press any key to set controller into Analog_Input_To_Output DataMode...\n")
    getch()

    mti.ResetDevicePosition()	# Reset the device position for safety
    mti.StopDataStream() 	 # Stop Data Stream BEFORE changing Data Modes
    mti.SetDeviceParam( MTIParam.DataMode, MTIDataMode.Analog_Input_To_Output )
    mti.StartDataStream()
    
    # Switch back to Sample_Output DataMode
    print("Press any key to return controller to Sample_Output DataMode...\n")
    getch()
    mti.StopDataStream() # Stop the Data Stream BEFORE changing Data Modes
    mti.SetDeviceParam( MTIParam.DataMode, MTIDataMode.Sample_Output )
    mti.StartDataStream()
    mti.ResetDevicePosition()
    mti.SetDeviceParams( sParams )  # Restore original params back to controller

def SinglePointTracking_Lissajous(mti):
    TrackingExamples.SinglePointTracking_Lissajous( mti )

def SinglePointTracking_Raster(mti):
    TrackingExamples.SinglePointTracking_Raster( mti )

def DataMode7Raster(mti):
    TrackingExamples.DataMode7Raster( mti )

def Imaging(mti):
    TrackingExamples.Imaging( mti )

def PointToPointDemoRGB(mti):
    RGBExamples.PointToPointDemoRGB( mti )

def ScannningDemoRGB(mti):
    RGBExamples.ScanningDemoRGB( mti )

def ImportILDAFileDemoRGB(mti):
    RGBExamples.ImportILDAFileDemoRGB( mti )

def ArrowKeysDemoRGB(mti):
    RGBExamples.ArrowKeysDemoRGB( mti )

if __name__ == '__main__':
    main()
