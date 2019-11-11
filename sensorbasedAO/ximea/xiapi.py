from os.path import dirname, join
import sys

from ctypes import *
from ctypes.wintypes import *

from .xidefs import * 

try:
    import numpy as np
except ImportError:
    pass

"""
API parameter groups: 
    Basic: Line 463
    Image Format (binning, decimation): Line 1392
    AE Setup (exposure, gain, output signal): Line 1757
    Performance (bandwidth, sensor line, bit depth): Line 1882
    Temperature: Line 2115
    Color Correction: Line 2426
    Device IO (trigger, acquisition): Line 3162
    GPIO Setup: Line 3318
    Lens Control: Line 3638
    Device info parameters: Line 3866
    Device acquisition settings (image data format, sensor clock 
        frequency, framerate, counter, trigger delay, time stamp): Line 4003
    Extended Device parameters: Line 4549
    Version info: Line 5143
    API features (debug level): Line 5197
    Camera FFS: Line 535
    APIContextControl: Line 5513
    Sensor Control: Line 5530
    Extended Features: Line 5595   
"""

#import platform; platform.architecture - not reliable on Mac OSX
if sys.maxsize > 2**32:
    LIB_PATH = join(dirname(__file__), 'libs', 'x64')

    #library for operations with image data - to increase speed with which
    #get_image_data_numpy() returns data
    c_arr_ops = CDLL(join(LIB_PATH, 'xiArrOps64.dll'))

    #library for communication with device
    _device = CDLL(join(LIB_PATH, 'xiapi64.dll'))
    
else:
    LIB_PATH = join(dirname(__file__), 'libs', 'x32')

    #library for operations with image data - to increase speed with which
    #get_image_data_numpy() returns data
    c_arr_ops = CDLL(join(LIB_PATH, 'xiArrOps32.dll'))

    #library for communication with device
    _device = CDLL(join(LIB_PATH, 'xiapi32.dll'))


class Xi_error(Exception):
    '''
    Camera error. Specified by return codes from camera c library.
    '''
    def __init__(self, status):
        if status in ERROR_CODES:
            self.status = status
            self.descr = ERROR_CODES[status]
        else:
            self.descr = 'Unknown error'

    def __str__(self):
        return 'ERROR %i: %s' %(self.status, self.descr)

class Image(XI_IMG):
    '''
    Camera image class. It inherits from ctypes.Structure XI_IMG (see xidefs.py).
    '''
    def __init__(self):
        '''
        Initialization of an image inst for image data and metadata.
        '''
        self.size = sizeof(self)


    def get_image_data_raw(self):
        '''
        Return data (of types bytes) from memory specified by Image.bp.
        
        NOTE: Call this function before closing the camera. After the camera
        is closed, the memory is deallocated and it is impossible to retrieve
        the data.
        '''
        output_length = self.get_bytes_per_pixel()*self.width*self.height+self.padding_x*self.height
        return string_at(self.bp, output_length) 


    def get_image_data_numpy(self, invert_rgb_order=False):
        '''
        Return data as a numpy.Array type with dimension Image.height x
        Image.width (in case imgdataformat is XI_MONO8, XI_MONO16, XI_RAW8 or
        XI_RAW16), Image.height x Image.width x 3 (in case imgdataformat is
        XI_RGB24) or Image.height x Image.width x 4 (in case imgdataformat is
        XI_RGB24)

        invert_rgb_order (bool) determines the order of bytes in case of
        RGB and RGBA settings (if the order is R-G-B or B-G-R).

        NOTE: Call this function before closing the camera. After the camera
        is closed, the memory is deallocated and it is impossible to retrieve
        the data.
        '''
        try:
            if self.get_bytes_per_pixel() == 1:
                c_array = c_ubyte*self.width*self.height
                data = c_array()
                c_arr_ops.arr8bit(
                    c_int(self.height),
                    c_int(self.width),
                    c_int(self.padding_x),
                    c_void_p(self.bp),
                    data
                    )
                numpy_data = np.array(data, copy=False, dtype=np.uint8)
                return numpy_data  
            
            elif self.get_bytes_per_pixel() == 2:
                c_array = c_ushort*self.width*self.height
                data = c_array()
                c_arr_ops.arr16bit(
                    c_int(self.height),
                    c_int(self.width),
                    c_int(self.padding_x),
                    c_void_p(self.bp),
                    data
                    )
                numpy_data = np.array(data, copy=False, dtype=np.uint16)
                return numpy_data            
            
            elif self.get_bytes_per_pixel() == 3:
                if invert_rgb_order: invRGB = 1
                else: invRGB = 0
                
                c_array = c_ubyte*3*self.width*self.height
                data = c_array()
                c_arr_ops.arrRGB(
                    c_int(self.height),
                    c_int(self.width),
                    c_int(self.padding_x),
                    c_void_p(self.bp),
                    data,
                    c_int(invRGB)
                    )
                numpy_data = np.array(data, copy=False, dtype=np.uint8)
                return numpy_data

            elif self.get_bytes_per_pixel() == 4:
                if invert_rgb_order: invRGB = 1
                else: invRGB = 0
                
                c_array = c_ubyte*4*self.width*self.height
                data = c_array()
                c_arr_ops.arrRGBA(
                    c_int(self.height),
                    c_int(self.width),
                    c_int(self.padding_x),
                    c_void_p(self.bp),
                    data,
                    c_int(invRGB)
                    )
                numpy_data = np.array(data, copy=False, dtype=np.uint8)
                return numpy_data
            
            else:
                raise Xi_error(108)     #"Data format not supported"
        
        except NameError:
            raise ImportError('Numpy module is not installed.')

        
    def get_bytes_per_pixel(self):
        '''
        Return number (int) of data image bytes per single pixel.
        '''
        if self.frm == XI_IMG_FORMAT["XI_MONO8"].value:
            return 1
        elif self.frm == XI_IMG_FORMAT["XI_RAW8"].value:
            return 1
        elif self.frm == XI_IMG_FORMAT["XI_MONO16"].value:
            return 2
        elif self.frm == XI_IMG_FORMAT["XI_RAW16"].value:
            return 2
        elif self.frm == XI_IMG_FORMAT["XI_RGB24"].value:
            return 3
        elif self.frm == XI_IMG_FORMAT["XI_RGB32"].value:
            return 4
        elif self.frm == XI_IMG_FORMAT["XI_RGB_PLANAR"].value:
            return 3
        else:
            raise Xi_error(108)     #"Data format not supported"
    

def _key_by_value(dictionary, val):
    # v.value == val.value because c_float(1) == c_float(1) returns False
    for k, v in dictionary.items():
        if v.value == val.value:
            return k
    raise ValueError('Value not found')

class Camera(object):
    '''
    Camera class. It wrapps xiApi c library and provides its functionality.
    '''
    def __init__(self, dev_id=0):
        '''
        Device initialization. For opening more connected cameras, create new
        instance with dev_id = 0, 1, 2, ...
        '''
        self.device = _device

        self.CAM_OPEN = False
        
        self.dev_id = dev_id
        
        self.handle = 0


    def open_device(self):
        '''
        Connect the camera specified by dev_id from __init__.
        '''
        if not self.CAM_OPEN:
            self.handle = HANDLE()
            stat = self.device.xiOpenDevice(self.dev_id, byref(self.handle))
            if not stat == 0:
                raise Xi_error(stat)
            
            self.CAM_OPEN = True
        else:
            raise RuntimeError('Camera already open. Create new instance to open next camera')

        
    def open_device_by(self, open_type, val):
        '''
        Connect the camera specified by open_type (string), see keys in
        dictionary xidefs.XI_OPEN_BY
        '''
        if not self.CAM_OPEN:
            self.get_number_devices()
            
            self.handle = HANDLE()

            if not open_type in XI_OPEN_BY:
                raise RuntimeError('invalid value')
                
            buf = create_string_buffer(bytes(val, 'UTF-8')) #only python3.x
            
            stat = self.device.xiOpenDeviceBy(
                XI_OPEN_BY[open_type],
                buf,
                byref(self.handle)
                )
            if not stat == 0:
                raise Xi_error(stat)
            self.CAM_OPEN = True
        else:
            raise RuntimeError('Camera already open. Create new instance to open next camera')


    def open_device_by_SN(self, serial_number):
        '''
        Connect the camera specified by its serial number (string).
        '''
        if not type(serial_number) == str:
            raise TypeError('serial_number must be a string')
        self.open_device_by('XI_OPEN_BY_SN', serial_number)


    def open_device_by_path(self, path):
        '''
        Connect the camera specified by its path (string).
        '''
        if not type(path) == str:
            raise TypeError('serial_number must be a string')
        self.open_device_by('XI_OPEN_BY_INST_PATH', path)
        
            
    def close_device(self):
        '''
        Close connection to the camera.
        '''
        stat = self.device.xiCloseDevice(self.handle)
        if not stat == 0:
            raise Xi_error(stat)
        self.CAM_OPEN = False

    
    def get_number_devices(self):
        '''
        Get number of cameras.

        NOTE: This function must be called before connection is established.
        '''
        count = DWORD()
        stat = self.device.xiGetNumberDevices(byref(count))
        if not stat == 0:
            raise Xi_error(stat)
        return count.value
        

    def start_acquisition(self):
        '''
        Start feeding data to the PC memory. Data can be retrieved with
        function get_image().
        '''
        stat = self.device.xiStartAcquisition(self.handle)
        if not stat == 0:
            raise Xi_error(stat)
        

    def stop_acquisition(self):
        '''
        Stop data acquisition.
        '''
        stat = self.device.xiStopAcquisition(self.handle)
        if not stat == 0:
            raise Xi_error(stat)
        

    def get_image(self, image, timeout=5000):
        '''
        Pass data from memory to Image instance image.
        Timeout is specified in microseconds.

        NOTE: Call this function before closing the camera. After the camera
        is closed, the memory is deallocated and it is impossible to retrieve
        the data.
        '''
        stat = self.device.xiGetImage(
            self.handle,
            DWORD(timeout),
            byref(image)
            )
        
        if not stat == 0:
            raise Xi_error(stat)


    def get_device_info_string(self, param):
        '''
        Return string with info specified by param (string). It is possible
        to call this function before establishing connection with the camera.

        param can be one of the following strings:
        "device_sn"
        "device_name"
        "device_inst_path"
        "device_loc_path"
        "device_type"
        '''
        
        prm = create_string_buffer(bytes(param, 'UTF-8')) #only python3.x
        
        val_len = 100
        val = create_string_buffer(val_len) 

        stat = self.device.xiGetDeviceInfoString(
            self.dev_id,
            prm,
            val,
            DWORD(val_len)
            )
        
        if not stat == 0:
            raise Xi_error(stat)
        return val.value        


    def set_param(self, param, val):
        '''
        Set value (data type depends on parameter) to a parameter
        (string, see parameters in xidefs.py).

        NOTE: Consider using function for specific parameter, e.g. if you want
        to set exposure, instead of using set_param('exposure', 10000), use
        set_exposure(10000).
        '''        
        prm = create_string_buffer(bytes(param, 'UTF-8')) #only python3.x
        
        if not param in VAL_TYPE:
            raise RuntimeError('invalid parameter')

        val_type = VAL_TYPE[param]
        
        if val_type == 'xiTypeString':
            val_len = DWORD(len(val))
            val = create_string_buffer(bytes(val, 'UTF-8')) #only python3.x
        elif val_type == 'xiTypeInteger':
            val_len = DWORD(4)
            val = pointer(c_int(val))
        elif val_type == 'xiTypeFloat':
            val_len = DWORD(4)
            val = pointer(FLOAT(val))
        elif val_type == 'xiTypeEnum':
            val_len = DWORD(4)
            val = pointer(ASSOC_ENUM[param][val])
        elif val_type == 'xiTypeBoolean':
            val_len = DWORD(4)
            val = pointer(c_int(val))       
        elif val_type == 'xiTypeCommand':
            val_len = DWORD(4)
            val = pointer(c_int(val))       
            
        stat = self.device.xiSetParam(
            self.handle,
            prm,
            val,
            val_len,
            XI_PRM_TYPE[val_type]
            )

        if not stat == 0:
            raise Xi_error(stat)

    
    def get_param(self, param, buffer_size=256):
        '''
        Get value (data type depends on parameter) of a parameter
        (string, see parameters in xidefs). buffer_size (int) determines the
        maximum size of output.

        NOTE: Consider using function for specific parameter, e.g. if you want
        to get exposure, instead of using get_param('exposure'), use
        get_exposure().
        ''' 
        prm = create_string_buffer(bytes(param, 'UTF-8')) #only python3.x

        if not param.split(':')[0] in VAL_TYPE:
            raise RuntimeError('invalid parameter')

        val_type = VAL_TYPE[param.split(':')[0]]

        if val_type == 'xiTypeString':
            val_len = DWORD(buffer_size)
            val = create_string_buffer(val_len.value) 
        elif val_type == 'xiTypeInteger' or \
             val_type == 'xiTypeEnum' or \
             val_type == 'xiTypeBoolean'or \
             val_type == 'xiTypeCommand' :
            val_len = DWORD(4)
            val = pointer(c_int())
        elif val_type == 'xiTypeFloat':
            val_len = DWORD(4)
            val = pointer(FLOAT())
        
        stat = self.device.xiGetParam(
            self.handle,
            prm,
            val,
            byref(val_len),
            byref(XI_PRM_TYPE[val_type])
            )

        if not stat == 0:
            raise Xi_error(stat)
        
        if val_type == 'xiTypeString':
            return val.value[:val_len.value]                 

        if val_type == 'xiTypeInteger' or val_type == 'xiTypeFloat':
            return val.contents.value

        if val_type == 'xiTypeEnum':
            return _key_by_value(ASSOC_ENUM[param], val.contents) 

        if val_type == 'xiTypeBoolean':
            return bool(val.contents.value)
#-------------------------------------------------------------------------------------------------------------------
	
# xiApi parameters
#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Basic
#-------------------------------------------------------------------------------------------------------------------

    def get_exposure(self):
        '''
        Exposure time in microsecondsXI_PRM_EXPOSURE
        '''
        return self.get_param('exposure')

    def get_exposure_maximum(self):
        '''
        Exposure time in microsecondsXI_PRM_EXPOSURE
        '''
        return self.get_param('exposure:max')

    def get_exposure_minimum(self):
        '''
        Exposure time in microsecondsXI_PRM_EXPOSURE
        '''
        return self.get_param('exposure:min')

    def get_exposure_increment(self):
        '''
        Exposure time in microsecondsXI_PRM_EXPOSURE
        '''
        return self.get_param('exposure:inc')

    def set_exposure(self, exposure):
        '''
        Exposure time in microsecondsXI_PRM_EXPOSURE
        '''
        self.set_param('exposure', exposure)

    def get_exposure_burst_count(self):
        '''
        Sets the number of times of exposure in one frame.XI_PRM_EXPOSURE_BURST_COUNT
        '''
        return self.get_param('exposure_burst_count')

    def get_exposure_burst_count_maximum(self):
        '''
        Sets the number of times of exposure in one frame.XI_PRM_EXPOSURE_BURST_COUNT
        '''
        return self.get_param('exposure_burst_count:max')

    def get_exposure_burst_count_minimum(self):
        '''
        Sets the number of times of exposure in one frame.XI_PRM_EXPOSURE_BURST_COUNT
        '''
        return self.get_param('exposure_burst_count:min')

    def get_exposure_burst_count_increment(self):
        '''
        Sets the number of times of exposure in one frame.XI_PRM_EXPOSURE_BURST_COUNT
        '''
        return self.get_param('exposure_burst_count:inc')

    def set_exposure_burst_count(self, exposure_burst_count):
        '''
        Sets the number of times of exposure in one frame.XI_PRM_EXPOSURE_BURST_COUNT
        '''
        self.set_param('exposure_burst_count', exposure_burst_count)

    def get_gain_selector(self):
        '''
        Gain selector for parameter Gain allows to select different type of gains.XI_PRM_GAIN_SELECTOR
        '''
        return self.get_param('gain_selector')

    def get_gain_selector_maximum(self):
        '''
        Gain selector for parameter Gain allows to select different type of gains.XI_PRM_GAIN_SELECTOR
        '''
        return self.get_param('gain_selector:max')

    def get_gain_selector_minimum(self):
        '''
        Gain selector for parameter Gain allows to select different type of gains.XI_PRM_GAIN_SELECTOR
        '''
        return self.get_param('gain_selector:min')

    def get_gain_selector_increment(self):
        '''
        Gain selector for parameter Gain allows to select different type of gains.XI_PRM_GAIN_SELECTOR
        '''
        return self.get_param('gain_selector:inc')

    def set_gain_selector(self, gain_selector):
        '''
        Gain selector for parameter Gain allows to select different type of gains.XI_PRM_GAIN_SELECTOR
        '''
        self.set_param('gain_selector', gain_selector)

    def get_gain(self):
        '''
        Gain in dBXI_PRM_GAIN
        '''
        return self.get_param('gain')

    def get_gain_maximum(self):
        '''
        Gain in dBXI_PRM_GAIN
        '''
        return self.get_param('gain:max')

    def get_gain_minimum(self):
        '''
        Gain in dBXI_PRM_GAIN
        '''
        return self.get_param('gain:min')

    def get_gain_increment(self):
        '''
        Gain in dBXI_PRM_GAIN
        '''
        return self.get_param('gain:inc')

    def set_gain(self, gain):
        '''
        Gain in dBXI_PRM_GAIN
        '''
        self.set_param('gain', gain)

    def get_downsampling(self):
        '''
        Change image resolution by binning or skipping.XI_PRM_DOWNSAMPLING
        '''
        return self.get_param('downsampling')

    def get_downsampling_maximum(self):
        '''
        Change image resolution by binning or skipping.XI_PRM_DOWNSAMPLING
        '''
        return self.get_param('downsampling:max')

    def get_downsampling_minimum(self):
        '''
        Change image resolution by binning or skipping.XI_PRM_DOWNSAMPLING
        '''
        return self.get_param('downsampling:min')

    def get_downsampling_increment(self):
        '''
        Change image resolution by binning or skipping.XI_PRM_DOWNSAMPLING
        '''
        return self.get_param('downsampling:inc')

    def set_downsampling(self, downsampling):
        '''
        Change image resolution by binning or skipping.XI_PRM_DOWNSAMPLING
        '''
        self.set_param('downsampling', downsampling)

    def get_downsampling_type(self):
        '''
        Change image downsampling type.XI_PRM_DOWNSAMPLING_TYPE
        '''
        return self.get_param('downsampling_type')

    def get_downsampling_type_maximum(self):
        '''
        Change image downsampling type.XI_PRM_DOWNSAMPLING_TYPE
        '''
        return self.get_param('downsampling_type:max')

    def get_downsampling_type_minimum(self):
        '''
        Change image downsampling type.XI_PRM_DOWNSAMPLING_TYPE
        '''
        return self.get_param('downsampling_type:min')

    def get_downsampling_type_increment(self):
        '''
        Change image downsampling type.XI_PRM_DOWNSAMPLING_TYPE
        '''
        return self.get_param('downsampling_type:inc')

    def set_downsampling_type(self, downsampling_type):
        '''
        Change image downsampling type.XI_PRM_DOWNSAMPLING_TYPE
        '''
        self.set_param('downsampling_type', downsampling_type)

    def get_test_pattern_generator_selector(self):
        '''
        Selects which test pattern generator is controlled by the TestPattern feature.XI_PRM_TEST_PATTERN_GENERATOR_SELECTOR
        '''
        return self.get_param('test_pattern_generator_selector')

    def get_test_pattern_generator_selector_maximum(self):
        '''
        Selects which test pattern generator is controlled by the TestPattern feature.XI_PRM_TEST_PATTERN_GENERATOR_SELECTOR
        '''
        return self.get_param('test_pattern_generator_selector:max')

    def get_test_pattern_generator_selector_minimum(self):
        '''
        Selects which test pattern generator is controlled by the TestPattern feature.XI_PRM_TEST_PATTERN_GENERATOR_SELECTOR
        '''
        return self.get_param('test_pattern_generator_selector:min')

    def get_test_pattern_generator_selector_increment(self):
        '''
        Selects which test pattern generator is controlled by the TestPattern feature.XI_PRM_TEST_PATTERN_GENERATOR_SELECTOR
        '''
        return self.get_param('test_pattern_generator_selector:inc')

    def set_test_pattern_generator_selector(self, test_pattern_generator_selector):
        '''
        Selects which test pattern generator is controlled by the TestPattern feature.XI_PRM_TEST_PATTERN_GENERATOR_SELECTOR
        '''
        self.set_param('test_pattern_generator_selector', test_pattern_generator_selector)

    def get_test_pattern(self):
        '''
        Selects which test pattern type is generated by the selected generator.XI_PRM_TEST_PATTERN
        '''
        return self.get_param('test_pattern')

    def get_test_pattern_maximum(self):
        '''
        Selects which test pattern type is generated by the selected generator.XI_PRM_TEST_PATTERN
        '''
        return self.get_param('test_pattern:max')

    def get_test_pattern_minimum(self):
        '''
        Selects which test pattern type is generated by the selected generator.XI_PRM_TEST_PATTERN
        '''
        return self.get_param('test_pattern:min')

    def get_test_pattern_increment(self):
        '''
        Selects which test pattern type is generated by the selected generator.XI_PRM_TEST_PATTERN
        '''
        return self.get_param('test_pattern:inc')

    def set_test_pattern(self, test_pattern):
        '''
        Selects which test pattern type is generated by the selected generator.XI_PRM_TEST_PATTERN
        '''
        self.set_param('test_pattern', test_pattern)

    def get_imgdataformat(self):
        '''
        Output data format.XI_PRM_IMAGE_DATA_FORMAT
        '''
        return self.get_param('imgdataformat')

    def get_imgdataformat_maximum(self):
        '''
        Output data format.XI_PRM_IMAGE_DATA_FORMAT
        '''
        return self.get_param('imgdataformat:max')

    def get_imgdataformat_minimum(self):
        '''
        Output data format.XI_PRM_IMAGE_DATA_FORMAT
        '''
        return self.get_param('imgdataformat:min')

    def get_imgdataformat_increment(self):
        '''
        Output data format.XI_PRM_IMAGE_DATA_FORMAT
        '''
        return self.get_param('imgdataformat:inc')

    def set_imgdataformat(self, imgdataformat):
        '''
        Output data format.XI_PRM_IMAGE_DATA_FORMAT
        '''
        self.set_param('imgdataformat', imgdataformat)

    def get_shutter_type(self):
        '''
        Change sensor shutter type(CMOS sensor).XI_PRM_SHUTTER_TYPE
        '''
        return self.get_param('shutter_type')

    def get_shutter_type_maximum(self):
        '''
        Change sensor shutter type(CMOS sensor).XI_PRM_SHUTTER_TYPE
        '''
        return self.get_param('shutter_type:max')

    def get_shutter_type_minimum(self):
        '''
        Change sensor shutter type(CMOS sensor).XI_PRM_SHUTTER_TYPE
        '''
        return self.get_param('shutter_type:min')

    def get_shutter_type_increment(self):
        '''
        Change sensor shutter type(CMOS sensor).XI_PRM_SHUTTER_TYPE
        '''
        return self.get_param('shutter_type:inc')

    def set_shutter_type(self, shutter_type):
        '''
        Change sensor shutter type(CMOS sensor).XI_PRM_SHUTTER_TYPE
        '''
        self.set_param('shutter_type', shutter_type)

    def get_sensor_taps(self):
        '''
        Number of tapsXI_PRM_SENSOR_TAPS
        '''
        return self.get_param('sensor_taps')

    def get_sensor_taps_maximum(self):
        '''
        Number of tapsXI_PRM_SENSOR_TAPS
        '''
        return self.get_param('sensor_taps:max')

    def get_sensor_taps_minimum(self):
        '''
        Number of tapsXI_PRM_SENSOR_TAPS
        '''
        return self.get_param('sensor_taps:min')

    def get_sensor_taps_increment(self):
        '''
        Number of tapsXI_PRM_SENSOR_TAPS
        '''
        return self.get_param('sensor_taps:inc')

    def set_sensor_taps(self, sensor_taps):
        '''
        Number of tapsXI_PRM_SENSOR_TAPS
        '''
        self.set_param('sensor_taps', sensor_taps)

    def is_aeag(self):
        '''
        Automatic exposure/gainXI_PRM_AEAG
        '''
        return self.get_param('aeag')

    def enable_aeag(self):
        '''
        Automatic exposure/gainXI_PRM_AEAG
        '''
        self.set_param('aeag', True)

    def disable_aeag(self):
        '''
        Automatic exposure/gainXI_PRM_AEAG
        '''
        self.set_param('aeag', False)

    def get_aeag_roi_offset_x(self):
        '''
        Automatic exposure/gain ROI offset XXI_PRM_AEAG_ROI_OFFSET_X
        '''
        return self.get_param('aeag_roi_offset_x')

    def get_aeag_roi_offset_x_maximum(self):
        '''
        Automatic exposure/gain ROI offset XXI_PRM_AEAG_ROI_OFFSET_X
        '''
        return self.get_param('aeag_roi_offset_x:max')

    def get_aeag_roi_offset_x_minimum(self):
        '''
        Automatic exposure/gain ROI offset XXI_PRM_AEAG_ROI_OFFSET_X
        '''
        return self.get_param('aeag_roi_offset_x:min')

    def get_aeag_roi_offset_x_increment(self):
        '''
        Automatic exposure/gain ROI offset XXI_PRM_AEAG_ROI_OFFSET_X
        '''
        return self.get_param('aeag_roi_offset_x:inc')

    def set_aeag_roi_offset_x(self, aeag_roi_offset_x):
        '''
        Automatic exposure/gain ROI offset XXI_PRM_AEAG_ROI_OFFSET_X
        '''
        self.set_param('aeag_roi_offset_x', aeag_roi_offset_x)

    def get_aeag_roi_offset_y(self):
        '''
        Automatic exposure/gain ROI offset YXI_PRM_AEAG_ROI_OFFSET_Y
        '''
        return self.get_param('aeag_roi_offset_y')

    def get_aeag_roi_offset_y_maximum(self):
        '''
        Automatic exposure/gain ROI offset YXI_PRM_AEAG_ROI_OFFSET_Y
        '''
        return self.get_param('aeag_roi_offset_y:max')

    def get_aeag_roi_offset_y_minimum(self):
        '''
        Automatic exposure/gain ROI offset YXI_PRM_AEAG_ROI_OFFSET_Y
        '''
        return self.get_param('aeag_roi_offset_y:min')

    def get_aeag_roi_offset_y_increment(self):
        '''
        Automatic exposure/gain ROI offset YXI_PRM_AEAG_ROI_OFFSET_Y
        '''
        return self.get_param('aeag_roi_offset_y:inc')

    def set_aeag_roi_offset_y(self, aeag_roi_offset_y):
        '''
        Automatic exposure/gain ROI offset YXI_PRM_AEAG_ROI_OFFSET_Y
        '''
        self.set_param('aeag_roi_offset_y', aeag_roi_offset_y)

    def get_aeag_roi_width(self):
        '''
        Automatic exposure/gain ROI WidthXI_PRM_AEAG_ROI_WIDTH
        '''
        return self.get_param('aeag_roi_width')

    def get_aeag_roi_width_maximum(self):
        '''
        Automatic exposure/gain ROI WidthXI_PRM_AEAG_ROI_WIDTH
        '''
        return self.get_param('aeag_roi_width:max')

    def get_aeag_roi_width_minimum(self):
        '''
        Automatic exposure/gain ROI WidthXI_PRM_AEAG_ROI_WIDTH
        '''
        return self.get_param('aeag_roi_width:min')

    def get_aeag_roi_width_increment(self):
        '''
        Automatic exposure/gain ROI WidthXI_PRM_AEAG_ROI_WIDTH
        '''
        return self.get_param('aeag_roi_width:inc')

    def set_aeag_roi_width(self, aeag_roi_width):
        '''
        Automatic exposure/gain ROI WidthXI_PRM_AEAG_ROI_WIDTH
        '''
        self.set_param('aeag_roi_width', aeag_roi_width)

    def get_aeag_roi_height(self):
        '''
        Automatic exposure/gain ROI HeightXI_PRM_AEAG_ROI_HEIGHT
        '''
        return self.get_param('aeag_roi_height')

    def get_aeag_roi_height_maximum(self):
        '''
        Automatic exposure/gain ROI HeightXI_PRM_AEAG_ROI_HEIGHT
        '''
        return self.get_param('aeag_roi_height:max')

    def get_aeag_roi_height_minimum(self):
        '''
        Automatic exposure/gain ROI HeightXI_PRM_AEAG_ROI_HEIGHT
        '''
        return self.get_param('aeag_roi_height:min')

    def get_aeag_roi_height_increment(self):
        '''
        Automatic exposure/gain ROI HeightXI_PRM_AEAG_ROI_HEIGHT
        '''
        return self.get_param('aeag_roi_height:inc')

    def set_aeag_roi_height(self, aeag_roi_height):
        '''
        Automatic exposure/gain ROI HeightXI_PRM_AEAG_ROI_HEIGHT
        '''
        self.set_param('aeag_roi_height', aeag_roi_height)

    def get_bpc_list_selector(self):
        '''
        Selector of list used by Sensor Defects Correction parameterXI_PRM_SENS_DEFECTS_CORR_LIST_SELECTOR
        '''
        return self.get_param('bpc_list_selector')

    def get_bpc_list_selector_maximum(self):
        '''
        Selector of list used by Sensor Defects Correction parameterXI_PRM_SENS_DEFECTS_CORR_LIST_SELECTOR
        '''
        return self.get_param('bpc_list_selector:max')

    def get_bpc_list_selector_minimum(self):
        '''
        Selector of list used by Sensor Defects Correction parameterXI_PRM_SENS_DEFECTS_CORR_LIST_SELECTOR
        '''
        return self.get_param('bpc_list_selector:min')

    def get_bpc_list_selector_increment(self):
        '''
        Selector of list used by Sensor Defects Correction parameterXI_PRM_SENS_DEFECTS_CORR_LIST_SELECTOR
        '''
        return self.get_param('bpc_list_selector:inc')

    def set_bpc_list_selector(self, bpc_list_selector):
        '''
        Selector of list used by Sensor Defects Correction parameterXI_PRM_SENS_DEFECTS_CORR_LIST_SELECTOR
        '''
        self.set_param('bpc_list_selector', bpc_list_selector)

    def get_sens_defects_corr_list_content(self,buffer_size=256):
        '''
        Sets/Gets sensor defects list in special text formatXI_PRM_SENS_DEFECTS_CORR_LIST_CONTENT
        '''
        return self.get_param('sens_defects_corr_list_content',buffer_size)

    def set_sens_defects_corr_list_content(self, sens_defects_corr_list_content):
        '''
        Sets/Gets sensor defects list in special text formatXI_PRM_SENS_DEFECTS_CORR_LIST_CONTENT
        '''
        self.set_param('sens_defects_corr_list_content', sens_defects_corr_list_content)

    def is_bpc(self):
        '''
        Correction of sensor defects (pixels, columns, rows) enable/disableXI_PRM_SENS_DEFECTS_CORR
        '''
        return self.get_param('bpc')

    def enable_bpc(self):
        '''
        Correction of sensor defects (pixels, columns, rows) enable/disableXI_PRM_SENS_DEFECTS_CORR
        '''
        self.set_param('bpc', True)

    def disable_bpc(self):
        '''
        Correction of sensor defects (pixels, columns, rows) enable/disableXI_PRM_SENS_DEFECTS_CORR
        '''
        self.set_param('bpc', False)

    def is_auto_wb(self):
        '''
        Automatic white balanceXI_PRM_AUTO_WB
        '''
        return self.get_param('auto_wb')

    def enable_auto_wb(self):
        '''
        Automatic white balanceXI_PRM_AUTO_WB
        '''
        self.set_param('auto_wb', True)

    def disable_auto_wb(self):
        '''
        Automatic white balanceXI_PRM_AUTO_WB
        '''
        self.set_param('auto_wb', False)

    def get_manual_wb(self):
        '''
        Calculates White Balance(xiGetImage function must be called)XI_PRM_MANUAL_WB
        '''
        return self.get_param('manual_wb')

    def get_manual_wb_maximum(self):
        '''
        Calculates White Balance(xiGetImage function must be called)XI_PRM_MANUAL_WB
        '''
        return self.get_param('manual_wb:max')

    def get_manual_wb_minimum(self):
        '''
        Calculates White Balance(xiGetImage function must be called)XI_PRM_MANUAL_WB
        '''
        return self.get_param('manual_wb:min')

    def get_manual_wb_increment(self):
        '''
        Calculates White Balance(xiGetImage function must be called)XI_PRM_MANUAL_WB
        '''
        return self.get_param('manual_wb:inc')

    def set_manual_wb(self, manual_wb):
        '''
        Calculates White Balance(xiGetImage function must be called)XI_PRM_MANUAL_WB
        '''
        self.set_param('manual_wb', manual_wb)

    def get_wb_kr(self):
        '''
        White balance red coefficientXI_PRM_WB_KR
        '''
        return self.get_param('wb_kr')

    def get_wb_kr_maximum(self):
        '''
        White balance red coefficientXI_PRM_WB_KR
        '''
        return self.get_param('wb_kr:max')

    def get_wb_kr_minimum(self):
        '''
        White balance red coefficientXI_PRM_WB_KR
        '''
        return self.get_param('wb_kr:min')

    def get_wb_kr_increment(self):
        '''
        White balance red coefficientXI_PRM_WB_KR
        '''
        return self.get_param('wb_kr:inc')

    def set_wb_kr(self, wb_kr):
        '''
        White balance red coefficientXI_PRM_WB_KR
        '''
        self.set_param('wb_kr', wb_kr)

    def get_wb_kg(self):
        '''
        White balance green coefficientXI_PRM_WB_KG
        '''
        return self.get_param('wb_kg')

    def get_wb_kg_maximum(self):
        '''
        White balance green coefficientXI_PRM_WB_KG
        '''
        return self.get_param('wb_kg:max')

    def get_wb_kg_minimum(self):
        '''
        White balance green coefficientXI_PRM_WB_KG
        '''
        return self.get_param('wb_kg:min')

    def get_wb_kg_increment(self):
        '''
        White balance green coefficientXI_PRM_WB_KG
        '''
        return self.get_param('wb_kg:inc')

    def set_wb_kg(self, wb_kg):
        '''
        White balance green coefficientXI_PRM_WB_KG
        '''
        self.set_param('wb_kg', wb_kg)

    def get_wb_kb(self):
        '''
        White balance blue coefficientXI_PRM_WB_KB
        '''
        return self.get_param('wb_kb')

    def get_wb_kb_maximum(self):
        '''
        White balance blue coefficientXI_PRM_WB_KB
        '''
        return self.get_param('wb_kb:max')

    def get_wb_kb_minimum(self):
        '''
        White balance blue coefficientXI_PRM_WB_KB
        '''
        return self.get_param('wb_kb:min')

    def get_wb_kb_increment(self):
        '''
        White balance blue coefficientXI_PRM_WB_KB
        '''
        return self.get_param('wb_kb:inc')

    def set_wb_kb(self, wb_kb):
        '''
        White balance blue coefficientXI_PRM_WB_KB
        '''
        self.set_param('wb_kb', wb_kb)

    def get_width(self):
        '''
        Width of the Image provided by the device (in pixels).XI_PRM_WIDTH
        '''
        return self.get_param('width')

    def get_width_maximum(self):
        '''
        Width of the Image provided by the device (in pixels).XI_PRM_WIDTH
        '''
        return self.get_param('width:max')

    def get_width_minimum(self):
        '''
        Width of the Image provided by the device (in pixels).XI_PRM_WIDTH
        '''
        return self.get_param('width:min')

    def get_width_increment(self):
        '''
        Width of the Image provided by the device (in pixels).XI_PRM_WIDTH
        '''
        return self.get_param('width:inc')

    def set_width(self, width):
        '''
        Width of the Image provided by the device (in pixels).XI_PRM_WIDTH
        '''
        self.set_param('width', width)

    def get_height(self):
        '''
        Height of the Image provided by the device (in pixels).XI_PRM_HEIGHT
        '''
        return self.get_param('height')

    def get_height_maximum(self):
        '''
        Height of the Image provided by the device (in pixels).XI_PRM_HEIGHT
        '''
        return self.get_param('height:max')

    def get_height_minimum(self):
        '''
        Height of the Image provided by the device (in pixels).XI_PRM_HEIGHT
        '''
        return self.get_param('height:min')

    def get_height_increment(self):
        '''
        Height of the Image provided by the device (in pixels).XI_PRM_HEIGHT
        '''
        return self.get_param('height:inc')

    def set_height(self, height):
        '''
        Height of the Image provided by the device (in pixels).XI_PRM_HEIGHT
        '''
        self.set_param('height', height)

    def get_offsetX(self):
        '''
        Horizontal offset from the origin to the area of interest (in pixels).XI_PRM_OFFSET_X
        '''
        return self.get_param('offsetX')

    def get_offsetX_maximum(self):
        '''
        Horizontal offset from the origin to the area of interest (in pixels).XI_PRM_OFFSET_X
        '''
        return self.get_param('offsetX:max')

    def get_offsetX_minimum(self):
        '''
        Horizontal offset from the origin to the area of interest (in pixels).XI_PRM_OFFSET_X
        '''
        return self.get_param('offsetX:min')

    def get_offsetX_increment(self):
        '''
        Horizontal offset from the origin to the area of interest (in pixels).XI_PRM_OFFSET_X
        '''
        return self.get_param('offsetX:inc')

    def set_offsetX(self, offsetX):
        '''
        Horizontal offset from the origin to the area of interest (in pixels).XI_PRM_OFFSET_X
        '''
        self.set_param('offsetX', offsetX)

    def get_offsetY(self):
        '''
        Vertical offset from the origin to the area of interest (in pixels).XI_PRM_OFFSET_Y
        '''
        return self.get_param('offsetY')

    def get_offsetY_maximum(self):
        '''
        Vertical offset from the origin to the area of interest (in pixels).XI_PRM_OFFSET_Y
        '''
        return self.get_param('offsetY:max')

    def get_offsetY_minimum(self):
        '''
        Vertical offset from the origin to the area of interest (in pixels).XI_PRM_OFFSET_Y
        '''
        return self.get_param('offsetY:min')

    def get_offsetY_increment(self):
        '''
        Vertical offset from the origin to the area of interest (in pixels).XI_PRM_OFFSET_Y
        '''
        return self.get_param('offsetY:inc')

    def set_offsetY(self, offsetY):
        '''
        Vertical offset from the origin to the area of interest (in pixels).XI_PRM_OFFSET_Y
        '''
        self.set_param('offsetY', offsetY)

    def get_region_selector(self):
        '''
        Selects Region in Multiple ROI which parameters are set by width, height, ... ,region modeXI_PRM_REGION_SELECTOR
        '''
        return self.get_param('region_selector')

    def get_region_selector_maximum(self):
        '''
        Selects Region in Multiple ROI which parameters are set by width, height, ... ,region modeXI_PRM_REGION_SELECTOR
        '''
        return self.get_param('region_selector:max')

    def get_region_selector_minimum(self):
        '''
        Selects Region in Multiple ROI which parameters are set by width, height, ... ,region modeXI_PRM_REGION_SELECTOR
        '''
        return self.get_param('region_selector:min')

    def get_region_selector_increment(self):
        '''
        Selects Region in Multiple ROI which parameters are set by width, height, ... ,region modeXI_PRM_REGION_SELECTOR
        '''
        return self.get_param('region_selector:inc')

    def set_region_selector(self, region_selector):
        '''
        Selects Region in Multiple ROI which parameters are set by width, height, ... ,region modeXI_PRM_REGION_SELECTOR
        '''
        self.set_param('region_selector', region_selector)

    def get_region_mode(self):
        '''
        Activates/deactivates Region selected by Region SelectorXI_PRM_REGION_MODE
        '''
        return self.get_param('region_mode')

    def get_region_mode_maximum(self):
        '''
        Activates/deactivates Region selected by Region SelectorXI_PRM_REGION_MODE
        '''
        return self.get_param('region_mode:max')

    def get_region_mode_minimum(self):
        '''
        Activates/deactivates Region selected by Region SelectorXI_PRM_REGION_MODE
        '''
        return self.get_param('region_mode:min')

    def get_region_mode_increment(self):
        '''
        Activates/deactivates Region selected by Region SelectorXI_PRM_REGION_MODE
        '''
        return self.get_param('region_mode:inc')

    def set_region_mode(self, region_mode):
        '''
        Activates/deactivates Region selected by Region SelectorXI_PRM_REGION_MODE
        '''
        self.set_param('region_mode', region_mode)

    def is_horizontal_flip(self):
        '''
        Horizontal flip enableXI_PRM_HORIZONTAL_FLIP
        '''
        return self.get_param('horizontal_flip')

    def enable_horizontal_flip(self):
        '''
        Horizontal flip enableXI_PRM_HORIZONTAL_FLIP
        '''
        self.set_param('horizontal_flip', True)

    def disable_horizontal_flip(self):
        '''
        Horizontal flip enableXI_PRM_HORIZONTAL_FLIP
        '''
        self.set_param('horizontal_flip', False)

    def is_vertical_flip(self):
        '''
        Vertical flip enableXI_PRM_VERTICAL_FLIP
        '''
        return self.get_param('vertical_flip')

    def enable_vertical_flip(self):
        '''
        Vertical flip enableXI_PRM_VERTICAL_FLIP
        '''
        self.set_param('vertical_flip', True)

    def disable_vertical_flip(self):
        '''
        Vertical flip enableXI_PRM_VERTICAL_FLIP
        '''
        self.set_param('vertical_flip', False)

    def is_ffc(self):
        '''
        Image flat field correctionXI_PRM_FFC
        '''
        return self.get_param('ffc')

    def enable_ffc(self):
        '''
        Image flat field correctionXI_PRM_FFC
        '''
        self.set_param('ffc', True)

    def disable_ffc(self):
        '''
        Image flat field correctionXI_PRM_FFC
        '''
        self.set_param('ffc', False)

    def get_ffc_flat_field_file_name(self,buffer_size=256):
        '''
        Set name of file to be applied for FFC processor.XI_PRM_FFC_FLAT_FIELD_FILE_NAME
        '''
        return self.get_param('ffc_flat_field_file_name',buffer_size)

    def set_ffc_flat_field_file_name(self, ffc_flat_field_file_name):
        '''
        Set name of file to be applied for FFC processor.XI_PRM_FFC_FLAT_FIELD_FILE_NAME
        '''
        self.set_param('ffc_flat_field_file_name', ffc_flat_field_file_name)

    def get_ffc_dark_field_file_name(self,buffer_size=256):
        '''
        Set name of file to be applied for FFC processor.XI_PRM_FFC_DARK_FIELD_FILE_NAME
        '''
        return self.get_param('ffc_dark_field_file_name',buffer_size)

    def set_ffc_dark_field_file_name(self, ffc_dark_field_file_name):
        '''
        Set name of file to be applied for FFC processor.XI_PRM_FFC_DARK_FIELD_FILE_NAME
        '''
        self.set_param('ffc_dark_field_file_name', ffc_dark_field_file_name)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Image Format
#-------------------------------------------------------------------------------------------------------------------

    def get_binning_selector(self):
        '''
        Binning engine selector.XI_PRM_BINNING_SELECTOR
        '''
        return self.get_param('binning_selector')

    def get_binning_selector_maximum(self):
        '''
        Binning engine selector.XI_PRM_BINNING_SELECTOR
        '''
        return self.get_param('binning_selector:max')

    def get_binning_selector_minimum(self):
        '''
        Binning engine selector.XI_PRM_BINNING_SELECTOR
        '''
        return self.get_param('binning_selector:min')

    def get_binning_selector_increment(self):
        '''
        Binning engine selector.XI_PRM_BINNING_SELECTOR
        '''
        return self.get_param('binning_selector:inc')

    def set_binning_selector(self, binning_selector):
        '''
        Binning engine selector.XI_PRM_BINNING_SELECTOR
        '''
        self.set_param('binning_selector', binning_selector)

    def get_binning_vertical_mode(self):
        '''
        Sets the mode to use to combine vertical pixel together.XI_PRM_BINNING_VERTICAL_MODE
        '''
        return self.get_param('binning_vertical_mode')

    def get_binning_vertical_mode_maximum(self):
        '''
        Sets the mode to use to combine vertical pixel together.XI_PRM_BINNING_VERTICAL_MODE
        '''
        return self.get_param('binning_vertical_mode:max')

    def get_binning_vertical_mode_minimum(self):
        '''
        Sets the mode to use to combine vertical pixel together.XI_PRM_BINNING_VERTICAL_MODE
        '''
        return self.get_param('binning_vertical_mode:min')

    def get_binning_vertical_mode_increment(self):
        '''
        Sets the mode to use to combine vertical pixel together.XI_PRM_BINNING_VERTICAL_MODE
        '''
        return self.get_param('binning_vertical_mode:inc')

    def set_binning_vertical_mode(self, binning_vertical_mode):
        '''
        Sets the mode to use to combine vertical pixel together.XI_PRM_BINNING_VERTICAL_MODE
        '''
        self.set_param('binning_vertical_mode', binning_vertical_mode)

    def get_binning_vertical(self):
        '''
        Vertical Binning - number of vertical photo-sensitive cells to combine together.XI_PRM_BINNING_VERTICAL
        '''
        return self.get_param('binning_vertical')

    def get_binning_vertical_maximum(self):
        '''
        Vertical Binning - number of vertical photo-sensitive cells to combine together.XI_PRM_BINNING_VERTICAL
        '''
        return self.get_param('binning_vertical:max')

    def get_binning_vertical_minimum(self):
        '''
        Vertical Binning - number of vertical photo-sensitive cells to combine together.XI_PRM_BINNING_VERTICAL
        '''
        return self.get_param('binning_vertical:min')

    def get_binning_vertical_increment(self):
        '''
        Vertical Binning - number of vertical photo-sensitive cells to combine together.XI_PRM_BINNING_VERTICAL
        '''
        return self.get_param('binning_vertical:inc')

    def set_binning_vertical(self, binning_vertical):
        '''
        Vertical Binning - number of vertical photo-sensitive cells to combine together.XI_PRM_BINNING_VERTICAL
        '''
        self.set_param('binning_vertical', binning_vertical)

    def get_binning_horizontal_mode(self):
        '''
        Sets the mode to use to combine horizontal pixel together.XI_PRM_BINNING_HORIZONTAL_MODE
        '''
        return self.get_param('binning_horizontal_mode')

    def get_binning_horizontal_mode_maximum(self):
        '''
        Sets the mode to use to combine horizontal pixel together.XI_PRM_BINNING_HORIZONTAL_MODE
        '''
        return self.get_param('binning_horizontal_mode:max')

    def get_binning_horizontal_mode_minimum(self):
        '''
        Sets the mode to use to combine horizontal pixel together.XI_PRM_BINNING_HORIZONTAL_MODE
        '''
        return self.get_param('binning_horizontal_mode:min')

    def get_binning_horizontal_mode_increment(self):
        '''
        Sets the mode to use to combine horizontal pixel together.XI_PRM_BINNING_HORIZONTAL_MODE
        '''
        return self.get_param('binning_horizontal_mode:inc')

    def set_binning_horizontal_mode(self, binning_horizontal_mode):
        '''
        Sets the mode to use to combine horizontal pixel together.XI_PRM_BINNING_HORIZONTAL_MODE
        '''
        self.set_param('binning_horizontal_mode', binning_horizontal_mode)

    def get_binning_horizontal(self):
        '''
        Horizontal Binning - number of horizontal photo-sensitive cells to combine together.XI_PRM_BINNING_HORIZONTAL
        '''
        return self.get_param('binning_horizontal')

    def get_binning_horizontal_maximum(self):
        '''
        Horizontal Binning - number of horizontal photo-sensitive cells to combine together.XI_PRM_BINNING_HORIZONTAL
        '''
        return self.get_param('binning_horizontal:max')

    def get_binning_horizontal_minimum(self):
        '''
        Horizontal Binning - number of horizontal photo-sensitive cells to combine together.XI_PRM_BINNING_HORIZONTAL
        '''
        return self.get_param('binning_horizontal:min')

    def get_binning_horizontal_increment(self):
        '''
        Horizontal Binning - number of horizontal photo-sensitive cells to combine together.XI_PRM_BINNING_HORIZONTAL
        '''
        return self.get_param('binning_horizontal:inc')

    def set_binning_horizontal(self, binning_horizontal):
        '''
        Horizontal Binning - number of horizontal photo-sensitive cells to combine together.XI_PRM_BINNING_HORIZONTAL
        '''
        self.set_param('binning_horizontal', binning_horizontal)

    def get_binning_horizontal_pattern(self):
        '''
        Binning horizontal pattern type.XI_PRM_BINNING_HORIZONTAL_PATTERN
        '''
        return self.get_param('binning_horizontal_pattern')

    def get_binning_horizontal_pattern_maximum(self):
        '''
        Binning horizontal pattern type.XI_PRM_BINNING_HORIZONTAL_PATTERN
        '''
        return self.get_param('binning_horizontal_pattern:max')

    def get_binning_horizontal_pattern_minimum(self):
        '''
        Binning horizontal pattern type.XI_PRM_BINNING_HORIZONTAL_PATTERN
        '''
        return self.get_param('binning_horizontal_pattern:min')

    def get_binning_horizontal_pattern_increment(self):
        '''
        Binning horizontal pattern type.XI_PRM_BINNING_HORIZONTAL_PATTERN
        '''
        return self.get_param('binning_horizontal_pattern:inc')

    def set_binning_horizontal_pattern(self, binning_horizontal_pattern):
        '''
        Binning horizontal pattern type.XI_PRM_BINNING_HORIZONTAL_PATTERN
        '''
        self.set_param('binning_horizontal_pattern', binning_horizontal_pattern)

    def get_binning_vertical_pattern(self):
        '''
        Binning vertical pattern type.XI_PRM_BINNING_VERTICAL_PATTERN
        '''
        return self.get_param('binning_vertical_pattern')

    def get_binning_vertical_pattern_maximum(self):
        '''
        Binning vertical pattern type.XI_PRM_BINNING_VERTICAL_PATTERN
        '''
        return self.get_param('binning_vertical_pattern:max')

    def get_binning_vertical_pattern_minimum(self):
        '''
        Binning vertical pattern type.XI_PRM_BINNING_VERTICAL_PATTERN
        '''
        return self.get_param('binning_vertical_pattern:min')

    def get_binning_vertical_pattern_increment(self):
        '''
        Binning vertical pattern type.XI_PRM_BINNING_VERTICAL_PATTERN
        '''
        return self.get_param('binning_vertical_pattern:inc')

    def set_binning_vertical_pattern(self, binning_vertical_pattern):
        '''
        Binning vertical pattern type.XI_PRM_BINNING_VERTICAL_PATTERN
        '''
        self.set_param('binning_vertical_pattern', binning_vertical_pattern)

    def get_decimation_selector(self):
        '''
        Decimation engine selector.XI_PRM_DECIMATION_SELECTOR
        '''
        return self.get_param('decimation_selector')

    def get_decimation_selector_maximum(self):
        '''
        Decimation engine selector.XI_PRM_DECIMATION_SELECTOR
        '''
        return self.get_param('decimation_selector:max')

    def get_decimation_selector_minimum(self):
        '''
        Decimation engine selector.XI_PRM_DECIMATION_SELECTOR
        '''
        return self.get_param('decimation_selector:min')

    def get_decimation_selector_increment(self):
        '''
        Decimation engine selector.XI_PRM_DECIMATION_SELECTOR
        '''
        return self.get_param('decimation_selector:inc')

    def set_decimation_selector(self, decimation_selector):
        '''
        Decimation engine selector.XI_PRM_DECIMATION_SELECTOR
        '''
        self.set_param('decimation_selector', decimation_selector)

    def get_decimation_vertical(self):
        '''
        Vertical Decimation - vertical sub-sampling of the image - reduces the vertical resolution of the image by the specified vertical decimation factor.XI_PRM_DECIMATION_VERTICAL
        '''
        return self.get_param('decimation_vertical')

    def get_decimation_vertical_maximum(self):
        '''
        Vertical Decimation - vertical sub-sampling of the image - reduces the vertical resolution of the image by the specified vertical decimation factor.XI_PRM_DECIMATION_VERTICAL
        '''
        return self.get_param('decimation_vertical:max')

    def get_decimation_vertical_minimum(self):
        '''
        Vertical Decimation - vertical sub-sampling of the image - reduces the vertical resolution of the image by the specified vertical decimation factor.XI_PRM_DECIMATION_VERTICAL
        '''
        return self.get_param('decimation_vertical:min')

    def get_decimation_vertical_increment(self):
        '''
        Vertical Decimation - vertical sub-sampling of the image - reduces the vertical resolution of the image by the specified vertical decimation factor.XI_PRM_DECIMATION_VERTICAL
        '''
        return self.get_param('decimation_vertical:inc')

    def set_decimation_vertical(self, decimation_vertical):
        '''
        Vertical Decimation - vertical sub-sampling of the image - reduces the vertical resolution of the image by the specified vertical decimation factor.XI_PRM_DECIMATION_VERTICAL
        '''
        self.set_param('decimation_vertical', decimation_vertical)

    def get_decimation_horizontal(self):
        '''
        Horizontal Decimation - horizontal sub-sampling of the image - reduces the horizontal resolution of the image by the specified vertical decimation factor.XI_PRM_DECIMATION_HORIZONTAL
        '''
        return self.get_param('decimation_horizontal')

    def get_decimation_horizontal_maximum(self):
        '''
        Horizontal Decimation - horizontal sub-sampling of the image - reduces the horizontal resolution of the image by the specified vertical decimation factor.XI_PRM_DECIMATION_HORIZONTAL
        '''
        return self.get_param('decimation_horizontal:max')

    def get_decimation_horizontal_minimum(self):
        '''
        Horizontal Decimation - horizontal sub-sampling of the image - reduces the horizontal resolution of the image by the specified vertical decimation factor.XI_PRM_DECIMATION_HORIZONTAL
        '''
        return self.get_param('decimation_horizontal:min')

    def get_decimation_horizontal_increment(self):
        '''
        Horizontal Decimation - horizontal sub-sampling of the image - reduces the horizontal resolution of the image by the specified vertical decimation factor.XI_PRM_DECIMATION_HORIZONTAL
        '''
        return self.get_param('decimation_horizontal:inc')

    def set_decimation_horizontal(self, decimation_horizontal):
        '''
        Horizontal Decimation - horizontal sub-sampling of the image - reduces the horizontal resolution of the image by the specified vertical decimation factor.XI_PRM_DECIMATION_HORIZONTAL
        '''
        self.set_param('decimation_horizontal', decimation_horizontal)

    def get_decimation_horizontal_pattern(self):
        '''
        Decimation horizontal pattern type.XI_PRM_DECIMATION_HORIZONTAL_PATTERN
        '''
        return self.get_param('decimation_horizontal_pattern')

    def get_decimation_horizontal_pattern_maximum(self):
        '''
        Decimation horizontal pattern type.XI_PRM_DECIMATION_HORIZONTAL_PATTERN
        '''
        return self.get_param('decimation_horizontal_pattern:max')

    def get_decimation_horizontal_pattern_minimum(self):
        '''
        Decimation horizontal pattern type.XI_PRM_DECIMATION_HORIZONTAL_PATTERN
        '''
        return self.get_param('decimation_horizontal_pattern:min')

    def get_decimation_horizontal_pattern_increment(self):
        '''
        Decimation horizontal pattern type.XI_PRM_DECIMATION_HORIZONTAL_PATTERN
        '''
        return self.get_param('decimation_horizontal_pattern:inc')

    def set_decimation_horizontal_pattern(self, decimation_horizontal_pattern):
        '''
        Decimation horizontal pattern type.XI_PRM_DECIMATION_HORIZONTAL_PATTERN
        '''
        self.set_param('decimation_horizontal_pattern', decimation_horizontal_pattern)

    def get_decimation_vertical_pattern(self):
        '''
        Decimation vertical pattern type.XI_PRM_DECIMATION_VERTICAL_PATTERN
        '''
        return self.get_param('decimation_vertical_pattern')

    def get_decimation_vertical_pattern_maximum(self):
        '''
        Decimation vertical pattern type.XI_PRM_DECIMATION_VERTICAL_PATTERN
        '''
        return self.get_param('decimation_vertical_pattern:max')

    def get_decimation_vertical_pattern_minimum(self):
        '''
        Decimation vertical pattern type.XI_PRM_DECIMATION_VERTICAL_PATTERN
        '''
        return self.get_param('decimation_vertical_pattern:min')

    def get_decimation_vertical_pattern_increment(self):
        '''
        Decimation vertical pattern type.XI_PRM_DECIMATION_VERTICAL_PATTERN
        '''
        return self.get_param('decimation_vertical_pattern:inc')

    def set_decimation_vertical_pattern(self, decimation_vertical_pattern):
        '''
        Decimation vertical pattern type.XI_PRM_DECIMATION_VERTICAL_PATTERN
        '''
        self.set_param('decimation_vertical_pattern', decimation_vertical_pattern)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: AE Setup
#-------------------------------------------------------------------------------------------------------------------

    def get_exp_priority(self):
        '''
        Exposure priority (0.8 - exposure 80%, gain 20%).XI_PRM_EXP_PRIORITY
        '''
        return self.get_param('exp_priority')

    def get_exp_priority_maximum(self):
        '''
        Exposure priority (0.8 - exposure 80%, gain 20%).XI_PRM_EXP_PRIORITY
        '''
        return self.get_param('exp_priority:max')

    def get_exp_priority_minimum(self):
        '''
        Exposure priority (0.8 - exposure 80%, gain 20%).XI_PRM_EXP_PRIORITY
        '''
        return self.get_param('exp_priority:min')

    def get_exp_priority_increment(self):
        '''
        Exposure priority (0.8 - exposure 80%, gain 20%).XI_PRM_EXP_PRIORITY
        '''
        return self.get_param('exp_priority:inc')

    def set_exp_priority(self, exp_priority):
        '''
        Exposure priority (0.8 - exposure 80%, gain 20%).XI_PRM_EXP_PRIORITY
        '''
        self.set_param('exp_priority', exp_priority)

    def get_ag_max_limit(self):
        '''
        Maximum limit of gain in AEAG procedureXI_PRM_AG_MAX_LIMIT
        '''
        return self.get_param('ag_max_limit')

    def get_ag_max_limit_maximum(self):
        '''
        Maximum limit of gain in AEAG procedureXI_PRM_AG_MAX_LIMIT
        '''
        return self.get_param('ag_max_limit:max')

    def get_ag_max_limit_minimum(self):
        '''
        Maximum limit of gain in AEAG procedureXI_PRM_AG_MAX_LIMIT
        '''
        return self.get_param('ag_max_limit:min')

    def get_ag_max_limit_increment(self):
        '''
        Maximum limit of gain in AEAG procedureXI_PRM_AG_MAX_LIMIT
        '''
        return self.get_param('ag_max_limit:inc')

    def set_ag_max_limit(self, ag_max_limit):
        '''
        Maximum limit of gain in AEAG procedureXI_PRM_AG_MAX_LIMIT
        '''
        self.set_param('ag_max_limit', ag_max_limit)

    def get_ae_max_limit(self):
        '''
        Maximum time (us) used for exposure in AEAG procedureXI_PRM_AE_MAX_LIMIT
        '''
        return self.get_param('ae_max_limit')

    def get_ae_max_limit_maximum(self):
        '''
        Maximum time (us) used for exposure in AEAG procedureXI_PRM_AE_MAX_LIMIT
        '''
        return self.get_param('ae_max_limit:max')

    def get_ae_max_limit_minimum(self):
        '''
        Maximum time (us) used for exposure in AEAG procedureXI_PRM_AE_MAX_LIMIT
        '''
        return self.get_param('ae_max_limit:min')

    def get_ae_max_limit_increment(self):
        '''
        Maximum time (us) used for exposure in AEAG procedureXI_PRM_AE_MAX_LIMIT
        '''
        return self.get_param('ae_max_limit:inc')

    def set_ae_max_limit(self, ae_max_limit):
        '''
        Maximum time (us) used for exposure in AEAG procedureXI_PRM_AE_MAX_LIMIT
        '''
        self.set_param('ae_max_limit', ae_max_limit)

    def get_aeag_level(self):
        '''
        Average intensity of output signal AEAG should achieve(in %)XI_PRM_AEAG_LEVEL
        '''
        return self.get_param('aeag_level')

    def get_aeag_level_maximum(self):
        '''
        Average intensity of output signal AEAG should achieve(in %)XI_PRM_AEAG_LEVEL
        '''
        return self.get_param('aeag_level:max')

    def get_aeag_level_minimum(self):
        '''
        Average intensity of output signal AEAG should achieve(in %)XI_PRM_AEAG_LEVEL
        '''
        return self.get_param('aeag_level:min')

    def get_aeag_level_increment(self):
        '''
        Average intensity of output signal AEAG should achieve(in %)XI_PRM_AEAG_LEVEL
        '''
        return self.get_param('aeag_level:inc')

    def set_aeag_level(self, aeag_level):
        '''
        Average intensity of output signal AEAG should achieve(in %)XI_PRM_AEAG_LEVEL
        '''
        self.set_param('aeag_level', aeag_level)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Performance
#-------------------------------------------------------------------------------------------------------------------

    def get_limit_bandwidth(self):
        '''
        Set/get bandwidth(datarate)(in Megabits)XI_PRM_LIMIT_BANDWIDTH
        '''
        return self.get_param('limit_bandwidth')

    def get_limit_bandwidth_maximum(self):
        '''
        Set/get bandwidth(datarate)(in Megabits)XI_PRM_LIMIT_BANDWIDTH
        '''
        return self.get_param('limit_bandwidth:max')

    def get_limit_bandwidth_minimum(self):
        '''
        Set/get bandwidth(datarate)(in Megabits)XI_PRM_LIMIT_BANDWIDTH
        '''
        return self.get_param('limit_bandwidth:min')

    def get_limit_bandwidth_increment(self):
        '''
        Set/get bandwidth(datarate)(in Megabits)XI_PRM_LIMIT_BANDWIDTH
        '''
        return self.get_param('limit_bandwidth:inc')

    def set_limit_bandwidth(self, limit_bandwidth):
        '''
        Set/get bandwidth(datarate)(in Megabits)XI_PRM_LIMIT_BANDWIDTH
        '''
        self.set_param('limit_bandwidth', limit_bandwidth)

    def get_limit_bandwidth_mode(self):
        '''
        Bandwidth limit enabledXI_PRM_LIMIT_BANDWIDTH_MODE
        '''
        return self.get_param('limit_bandwidth_mode')

    def get_limit_bandwidth_mode_maximum(self):
        '''
        Bandwidth limit enabledXI_PRM_LIMIT_BANDWIDTH_MODE
        '''
        return self.get_param('limit_bandwidth_mode:max')

    def get_limit_bandwidth_mode_minimum(self):
        '''
        Bandwidth limit enabledXI_PRM_LIMIT_BANDWIDTH_MODE
        '''
        return self.get_param('limit_bandwidth_mode:min')

    def get_limit_bandwidth_mode_increment(self):
        '''
        Bandwidth limit enabledXI_PRM_LIMIT_BANDWIDTH_MODE
        '''
        return self.get_param('limit_bandwidth_mode:inc')

    def set_limit_bandwidth_mode(self, limit_bandwidth_mode):
        '''
        Bandwidth limit enabledXI_PRM_LIMIT_BANDWIDTH_MODE
        '''
        self.set_param('limit_bandwidth_mode', limit_bandwidth_mode)

    def get_sensor_line_period(self):
        '''
        Image sensor line period in usXI_PRM_SENSOR_LINE_PERIOD
        '''
        return self.get_param('sensor_line_period')

    def get_sensor_line_period_maximum(self):
        '''
        Image sensor line period in usXI_PRM_SENSOR_LINE_PERIOD
        '''
        return self.get_param('sensor_line_period:max')

    def get_sensor_line_period_minimum(self):
        '''
        Image sensor line period in usXI_PRM_SENSOR_LINE_PERIOD
        '''
        return self.get_param('sensor_line_period:min')

    def get_sensor_line_period_increment(self):
        '''
        Image sensor line period in usXI_PRM_SENSOR_LINE_PERIOD
        '''
        return self.get_param('sensor_line_period:inc')

    def set_sensor_line_period(self, sensor_line_period):
        '''
        Image sensor line period in usXI_PRM_SENSOR_LINE_PERIOD
        '''
        self.set_param('sensor_line_period', sensor_line_period)

    def get_sensor_bit_depth(self):
        '''
        Sensor output data bit depth.XI_PRM_SENSOR_DATA_BIT_DEPTH
        '''
        return self.get_param('sensor_bit_depth')

    def get_sensor_bit_depth_maximum(self):
        '''
        Sensor output data bit depth.XI_PRM_SENSOR_DATA_BIT_DEPTH
        '''
        return self.get_param('sensor_bit_depth:max')

    def get_sensor_bit_depth_minimum(self):
        '''
        Sensor output data bit depth.XI_PRM_SENSOR_DATA_BIT_DEPTH
        '''
        return self.get_param('sensor_bit_depth:min')

    def get_sensor_bit_depth_increment(self):
        '''
        Sensor output data bit depth.XI_PRM_SENSOR_DATA_BIT_DEPTH
        '''
        return self.get_param('sensor_bit_depth:inc')

    def set_sensor_bit_depth(self, sensor_bit_depth):
        '''
        Sensor output data bit depth.XI_PRM_SENSOR_DATA_BIT_DEPTH
        '''
        self.set_param('sensor_bit_depth', sensor_bit_depth)

    def get_output_bit_depth(self):
        '''
        Device output data bit depth.XI_PRM_OUTPUT_DATA_BIT_DEPTH
        '''
        return self.get_param('output_bit_depth')

    def get_output_bit_depth_maximum(self):
        '''
        Device output data bit depth.XI_PRM_OUTPUT_DATA_BIT_DEPTH
        '''
        return self.get_param('output_bit_depth:max')

    def get_output_bit_depth_minimum(self):
        '''
        Device output data bit depth.XI_PRM_OUTPUT_DATA_BIT_DEPTH
        '''
        return self.get_param('output_bit_depth:min')

    def get_output_bit_depth_increment(self):
        '''
        Device output data bit depth.XI_PRM_OUTPUT_DATA_BIT_DEPTH
        '''
        return self.get_param('output_bit_depth:inc')

    def set_output_bit_depth(self, output_bit_depth):
        '''
        Device output data bit depth.XI_PRM_OUTPUT_DATA_BIT_DEPTH
        '''
        self.set_param('output_bit_depth', output_bit_depth)

    def get_image_data_bit_depth(self):
        '''
        bitdepth of data returned by function xiGetImageXI_PRM_IMAGE_DATA_BIT_DEPTH
        '''
        return self.get_param('image_data_bit_depth')

    def get_image_data_bit_depth_maximum(self):
        '''
        bitdepth of data returned by function xiGetImageXI_PRM_IMAGE_DATA_BIT_DEPTH
        '''
        return self.get_param('image_data_bit_depth:max')

    def get_image_data_bit_depth_minimum(self):
        '''
        bitdepth of data returned by function xiGetImageXI_PRM_IMAGE_DATA_BIT_DEPTH
        '''
        return self.get_param('image_data_bit_depth:min')

    def get_image_data_bit_depth_increment(self):
        '''
        bitdepth of data returned by function xiGetImageXI_PRM_IMAGE_DATA_BIT_DEPTH
        '''
        return self.get_param('image_data_bit_depth:inc')

    def set_image_data_bit_depth(self, image_data_bit_depth):
        '''
        bitdepth of data returned by function xiGetImageXI_PRM_IMAGE_DATA_BIT_DEPTH
        '''
        self.set_param('image_data_bit_depth', image_data_bit_depth)

    def is_output_bit_packing(self):
        '''
        Device output data packing (or grouping) enabled. Packing could be enabled if output_data_bit_depth > 8 and packing capability is available.XI_PRM_OUTPUT_DATA_PACKING
        '''
        return self.get_param('output_bit_packing')

    def enable_output_bit_packing(self):
        '''
        Device output data packing (or grouping) enabled. Packing could be enabled if output_data_bit_depth > 8 and packing capability is available.XI_PRM_OUTPUT_DATA_PACKING
        '''
        self.set_param('output_bit_packing', True)

    def disable_output_bit_packing(self):
        '''
        Device output data packing (or grouping) enabled. Packing could be enabled if output_data_bit_depth > 8 and packing capability is available.XI_PRM_OUTPUT_DATA_PACKING
        '''
        self.set_param('output_bit_packing', False)

    def get_output_bit_packing_type(self):
        '''
        Data packing type. Some cameras supports only specific packing type.XI_PRM_OUTPUT_DATA_PACKING_TYPE
        '''
        return self.get_param('output_bit_packing_type')

    def get_output_bit_packing_type_maximum(self):
        '''
        Data packing type. Some cameras supports only specific packing type.XI_PRM_OUTPUT_DATA_PACKING_TYPE
        '''
        return self.get_param('output_bit_packing_type:max')

    def get_output_bit_packing_type_minimum(self):
        '''
        Data packing type. Some cameras supports only specific packing type.XI_PRM_OUTPUT_DATA_PACKING_TYPE
        '''
        return self.get_param('output_bit_packing_type:min')

    def get_output_bit_packing_type_increment(self):
        '''
        Data packing type. Some cameras supports only specific packing type.XI_PRM_OUTPUT_DATA_PACKING_TYPE
        '''
        return self.get_param('output_bit_packing_type:inc')

    def set_output_bit_packing_type(self, output_bit_packing_type):
        '''
        Data packing type. Some cameras supports only specific packing type.XI_PRM_OUTPUT_DATA_PACKING_TYPE
        '''
        self.set_param('output_bit_packing_type', output_bit_packing_type)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Temperature
#-------------------------------------------------------------------------------------------------------------------

    def is_iscooled(self):
        '''
        Returns 1 for cameras that support cooling.XI_PRM_IS_COOLED
        '''
        return self.get_param('iscooled')

    def get_cooling(self):
        '''
        Temperature control mode.XI_PRM_COOLING
        '''
        return self.get_param('cooling')

    def get_cooling_maximum(self):
        '''
        Temperature control mode.XI_PRM_COOLING
        '''
        return self.get_param('cooling:max')

    def get_cooling_minimum(self):
        '''
        Temperature control mode.XI_PRM_COOLING
        '''
        return self.get_param('cooling:min')

    def get_cooling_increment(self):
        '''
        Temperature control mode.XI_PRM_COOLING
        '''
        return self.get_param('cooling:inc')

    def set_cooling(self, cooling):
        '''
        Temperature control mode.XI_PRM_COOLING
        '''
        self.set_param('cooling', cooling)

    def get_target_temp(self):
        '''
        Set sensor target temperature for cooling.XI_PRM_TARGET_TEMP
        '''
        return self.get_param('target_temp')

    def get_target_temp_maximum(self):
        '''
        Set sensor target temperature for cooling.XI_PRM_TARGET_TEMP
        '''
        return self.get_param('target_temp:max')

    def get_target_temp_minimum(self):
        '''
        Set sensor target temperature for cooling.XI_PRM_TARGET_TEMP
        '''
        return self.get_param('target_temp:min')

    def get_target_temp_increment(self):
        '''
        Set sensor target temperature for cooling.XI_PRM_TARGET_TEMP
        '''
        return self.get_param('target_temp:inc')

    def set_target_temp(self, target_temp):
        '''
        Set sensor target temperature for cooling.XI_PRM_TARGET_TEMP
        '''
        self.set_param('target_temp', target_temp)

    def get_temp_selector(self):
        '''
        Selector of mechanical point where thermometer is located.XI_PRM_TEMP_SELECTOR
        '''
        return self.get_param('temp_selector')

    def get_temp_selector_maximum(self):
        '''
        Selector of mechanical point where thermometer is located.XI_PRM_TEMP_SELECTOR
        '''
        return self.get_param('temp_selector:max')

    def get_temp_selector_minimum(self):
        '''
        Selector of mechanical point where thermometer is located.XI_PRM_TEMP_SELECTOR
        '''
        return self.get_param('temp_selector:min')

    def get_temp_selector_increment(self):
        '''
        Selector of mechanical point where thermometer is located.XI_PRM_TEMP_SELECTOR
        '''
        return self.get_param('temp_selector:inc')

    def set_temp_selector(self, temp_selector):
        '''
        Selector of mechanical point where thermometer is located.XI_PRM_TEMP_SELECTOR
        '''
        self.set_param('temp_selector', temp_selector)

    def get_temp(self):
        '''
        Camera temperature (selected by XI_PRM_TEMP_SELECTOR)XI_PRM_TEMP
        '''
        return self.get_param('temp')

    def get_temp_maximum(self):
        '''
        Camera temperature (selected by XI_PRM_TEMP_SELECTOR)XI_PRM_TEMP
        '''
        return self.get_param('temp:max')

    def get_temp_minimum(self):
        '''
        Camera temperature (selected by XI_PRM_TEMP_SELECTOR)XI_PRM_TEMP
        '''
        return self.get_param('temp:min')

    def get_temp_increment(self):
        '''
        Camera temperature (selected by XI_PRM_TEMP_SELECTOR)XI_PRM_TEMP
        '''
        return self.get_param('temp:inc')

    def get_device_temperature_ctrl_mode(self):
        '''
        Temperature control mode.XI_PRM_TEMP_CONTROL_MODE
        '''
        return self.get_param('device_temperature_ctrl_mode')

    def get_device_temperature_ctrl_mode_maximum(self):
        '''
        Temperature control mode.XI_PRM_TEMP_CONTROL_MODE
        '''
        return self.get_param('device_temperature_ctrl_mode:max')

    def get_device_temperature_ctrl_mode_minimum(self):
        '''
        Temperature control mode.XI_PRM_TEMP_CONTROL_MODE
        '''
        return self.get_param('device_temperature_ctrl_mode:min')

    def get_device_temperature_ctrl_mode_increment(self):
        '''
        Temperature control mode.XI_PRM_TEMP_CONTROL_MODE
        '''
        return self.get_param('device_temperature_ctrl_mode:inc')

    def set_device_temperature_ctrl_mode(self, device_temperature_ctrl_mode):
        '''
        Temperature control mode.XI_PRM_TEMP_CONTROL_MODE
        '''
        self.set_param('device_temperature_ctrl_mode', device_temperature_ctrl_mode)

    def get_chip_temp(self):
        '''
        Camera sensor temperatureXI_PRM_CHIP_TEMP
        '''
        return self.get_param('chip_temp')

    def get_chip_temp_maximum(self):
        '''
        Camera sensor temperatureXI_PRM_CHIP_TEMP
        '''
        return self.get_param('chip_temp:max')

    def get_chip_temp_minimum(self):
        '''
        Camera sensor temperatureXI_PRM_CHIP_TEMP
        '''
        return self.get_param('chip_temp:min')

    def get_chip_temp_increment(self):
        '''
        Camera sensor temperatureXI_PRM_CHIP_TEMP
        '''
        return self.get_param('chip_temp:inc')

    def get_hous_temp(self):
        '''
        Camera housing tepmeratureXI_PRM_HOUS_TEMP
        '''
        return self.get_param('hous_temp')

    def get_hous_temp_maximum(self):
        '''
        Camera housing tepmeratureXI_PRM_HOUS_TEMP
        '''
        return self.get_param('hous_temp:max')

    def get_hous_temp_minimum(self):
        '''
        Camera housing tepmeratureXI_PRM_HOUS_TEMP
        '''
        return self.get_param('hous_temp:min')

    def get_hous_temp_increment(self):
        '''
        Camera housing tepmeratureXI_PRM_HOUS_TEMP
        '''
        return self.get_param('hous_temp:inc')

    def get_hous_back_side_temp(self):
        '''
        Camera housing back side tepmeratureXI_PRM_HOUS_BACK_SIDE_TEMP
        '''
        return self.get_param('hous_back_side_temp')

    def get_hous_back_side_temp_maximum(self):
        '''
        Camera housing back side tepmeratureXI_PRM_HOUS_BACK_SIDE_TEMP
        '''
        return self.get_param('hous_back_side_temp:max')

    def get_hous_back_side_temp_minimum(self):
        '''
        Camera housing back side tepmeratureXI_PRM_HOUS_BACK_SIDE_TEMP
        '''
        return self.get_param('hous_back_side_temp:min')

    def get_hous_back_side_temp_increment(self):
        '''
        Camera housing back side tepmeratureXI_PRM_HOUS_BACK_SIDE_TEMP
        '''
        return self.get_param('hous_back_side_temp:inc')

    def get_sensor_board_temp(self):
        '''
        Camera sensor board temperatureXI_PRM_SENSOR_BOARD_TEMP
        '''
        return self.get_param('sensor_board_temp')

    def get_sensor_board_temp_maximum(self):
        '''
        Camera sensor board temperatureXI_PRM_SENSOR_BOARD_TEMP
        '''
        return self.get_param('sensor_board_temp:max')

    def get_sensor_board_temp_minimum(self):
        '''
        Camera sensor board temperatureXI_PRM_SENSOR_BOARD_TEMP
        '''
        return self.get_param('sensor_board_temp:min')

    def get_sensor_board_temp_increment(self):
        '''
        Camera sensor board temperatureXI_PRM_SENSOR_BOARD_TEMP
        '''
        return self.get_param('sensor_board_temp:inc')

    def get_device_temperature_element_sel(self):
        '''
        Temperature element selector (TEC(Peltier), Fan).XI_PRM_TEMP_ELEMENT_SEL
        '''
        return self.get_param('device_temperature_element_sel')

    def get_device_temperature_element_sel_maximum(self):
        '''
        Temperature element selector (TEC(Peltier), Fan).XI_PRM_TEMP_ELEMENT_SEL
        '''
        return self.get_param('device_temperature_element_sel:max')

    def get_device_temperature_element_sel_minimum(self):
        '''
        Temperature element selector (TEC(Peltier), Fan).XI_PRM_TEMP_ELEMENT_SEL
        '''
        return self.get_param('device_temperature_element_sel:min')

    def get_device_temperature_element_sel_increment(self):
        '''
        Temperature element selector (TEC(Peltier), Fan).XI_PRM_TEMP_ELEMENT_SEL
        '''
        return self.get_param('device_temperature_element_sel:inc')

    def set_device_temperature_element_sel(self, device_temperature_element_sel):
        '''
        Temperature element selector (TEC(Peltier), Fan).XI_PRM_TEMP_ELEMENT_SEL
        '''
        self.set_param('device_temperature_element_sel', device_temperature_element_sel)

    def get_device_temperature_element_val(self):
        '''
        Temperature element value in percents of full control rangeXI_PRM_TEMP_ELEMENT_VALUE
        '''
        return self.get_param('device_temperature_element_val')

    def get_device_temperature_element_val_maximum(self):
        '''
        Temperature element value in percents of full control rangeXI_PRM_TEMP_ELEMENT_VALUE
        '''
        return self.get_param('device_temperature_element_val:max')

    def get_device_temperature_element_val_minimum(self):
        '''
        Temperature element value in percents of full control rangeXI_PRM_TEMP_ELEMENT_VALUE
        '''
        return self.get_param('device_temperature_element_val:min')

    def get_device_temperature_element_val_increment(self):
        '''
        Temperature element value in percents of full control rangeXI_PRM_TEMP_ELEMENT_VALUE
        '''
        return self.get_param('device_temperature_element_val:inc')

    def set_device_temperature_element_val(self, device_temperature_element_val):
        '''
        Temperature element value in percents of full control rangeXI_PRM_TEMP_ELEMENT_VALUE
        '''
        self.set_param('device_temperature_element_val', device_temperature_element_val)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Color Correction
#-------------------------------------------------------------------------------------------------------------------

    def get_cms(self):
        '''
        Mode of color management system.XI_PRM_CMS
        '''
        return self.get_param('cms')

    def get_cms_maximum(self):
        '''
        Mode of color management system.XI_PRM_CMS
        '''
        return self.get_param('cms:max')

    def get_cms_minimum(self):
        '''
        Mode of color management system.XI_PRM_CMS
        '''
        return self.get_param('cms:min')

    def get_cms_increment(self):
        '''
        Mode of color management system.XI_PRM_CMS
        '''
        return self.get_param('cms:inc')

    def set_cms(self, cms):
        '''
        Mode of color management system.XI_PRM_CMS
        '''
        self.set_param('cms', cms)

    def get_cms_intent(self):
        '''
        Intent of color management system.XI_PRM_CMS_INTENT
        '''
        return self.get_param('cms_intent')

    def get_cms_intent_maximum(self):
        '''
        Intent of color management system.XI_PRM_CMS_INTENT
        '''
        return self.get_param('cms_intent:max')

    def get_cms_intent_minimum(self):
        '''
        Intent of color management system.XI_PRM_CMS_INTENT
        '''
        return self.get_param('cms_intent:min')

    def get_cms_intent_increment(self):
        '''
        Intent of color management system.XI_PRM_CMS_INTENT
        '''
        return self.get_param('cms_intent:inc')

    def set_cms_intent(self, cms_intent):
        '''
        Intent of color management system.XI_PRM_CMS_INTENT
        '''
        self.set_param('cms_intent', cms_intent)

    def is_apply_cms(self):
        '''
        Enable applying of CMS profiles to xiGetImage (see XI_PRM_INPUT_CMS_PROFILE, XI_PRM_OUTPUT_CMS_PROFILE).XI_PRM_APPLY_CMS
        '''
        return self.get_param('apply_cms')

    def enable_apply_cms(self):
        '''
        Enable applying of CMS profiles to xiGetImage (see XI_PRM_INPUT_CMS_PROFILE, XI_PRM_OUTPUT_CMS_PROFILE).XI_PRM_APPLY_CMS
        '''
        self.set_param('apply_cms', True)

    def disable_apply_cms(self):
        '''
        Enable applying of CMS profiles to xiGetImage (see XI_PRM_INPUT_CMS_PROFILE, XI_PRM_OUTPUT_CMS_PROFILE).XI_PRM_APPLY_CMS
        '''
        self.set_param('apply_cms', False)

    def get_input_cms_profile(self,buffer_size=256):
        '''
        Filename for input cms profile (e.g. input.icc)XI_PRM_INPUT_CMS_PROFILE
        '''
        return self.get_param('input_cms_profile',buffer_size)

    def set_input_cms_profile(self, input_cms_profile):
        '''
        Filename for input cms profile (e.g. input.icc)XI_PRM_INPUT_CMS_PROFILE
        '''
        self.set_param('input_cms_profile', input_cms_profile)

    def get_output_cms_profile(self,buffer_size=256):
        '''
        Filename for output cms profile (e.g. input.icc)XI_PRM_OUTPUT_CMS_PROFILE
        '''
        return self.get_param('output_cms_profile',buffer_size)

    def set_output_cms_profile(self, output_cms_profile):
        '''
        Filename for output cms profile (e.g. input.icc)XI_PRM_OUTPUT_CMS_PROFILE
        '''
        self.set_param('output_cms_profile', output_cms_profile)

    def is_iscolor(self):
        '''
        Returns 1 for color cameras.XI_PRM_IMAGE_IS_COLOR
        '''
        return self.get_param('iscolor')

    def get_cfa(self):
        '''
        Returns color filter array type of RAW data.XI_PRM_COLOR_FILTER_ARRAY
        '''
        return self.get_param('cfa')

    def get_cfa_maximum(self):
        '''
        Returns color filter array type of RAW data.XI_PRM_COLOR_FILTER_ARRAY
        '''
        return self.get_param('cfa:max')

    def get_cfa_minimum(self):
        '''
        Returns color filter array type of RAW data.XI_PRM_COLOR_FILTER_ARRAY
        '''
        return self.get_param('cfa:min')

    def get_cfa_increment(self):
        '''
        Returns color filter array type of RAW data.XI_PRM_COLOR_FILTER_ARRAY
        '''
        return self.get_param('cfa:inc')

    def get_gammaY(self):
        '''
        Luminosity gammaXI_PRM_GAMMAY
        '''
        return self.get_param('gammaY')

    def get_gammaY_maximum(self):
        '''
        Luminosity gammaXI_PRM_GAMMAY
        '''
        return self.get_param('gammaY:max')

    def get_gammaY_minimum(self):
        '''
        Luminosity gammaXI_PRM_GAMMAY
        '''
        return self.get_param('gammaY:min')

    def get_gammaY_increment(self):
        '''
        Luminosity gammaXI_PRM_GAMMAY
        '''
        return self.get_param('gammaY:inc')

    def set_gammaY(self, gammaY):
        '''
        Luminosity gammaXI_PRM_GAMMAY
        '''
        self.set_param('gammaY', gammaY)

    def get_gammaC(self):
        '''
        Chromaticity gammaXI_PRM_GAMMAC
        '''
        return self.get_param('gammaC')

    def get_gammaC_maximum(self):
        '''
        Chromaticity gammaXI_PRM_GAMMAC
        '''
        return self.get_param('gammaC:max')

    def get_gammaC_minimum(self):
        '''
        Chromaticity gammaXI_PRM_GAMMAC
        '''
        return self.get_param('gammaC:min')

    def get_gammaC_increment(self):
        '''
        Chromaticity gammaXI_PRM_GAMMAC
        '''
        return self.get_param('gammaC:inc')

    def set_gammaC(self, gammaC):
        '''
        Chromaticity gammaXI_PRM_GAMMAC
        '''
        self.set_param('gammaC', gammaC)

    def get_sharpness(self):
        '''
        Sharpness StrenghtXI_PRM_SHARPNESS
        '''
        return self.get_param('sharpness')

    def get_sharpness_maximum(self):
        '''
        Sharpness StrenghtXI_PRM_SHARPNESS
        '''
        return self.get_param('sharpness:max')

    def get_sharpness_minimum(self):
        '''
        Sharpness StrenghtXI_PRM_SHARPNESS
        '''
        return self.get_param('sharpness:min')

    def get_sharpness_increment(self):
        '''
        Sharpness StrenghtXI_PRM_SHARPNESS
        '''
        return self.get_param('sharpness:inc')

    def set_sharpness(self, sharpness):
        '''
        Sharpness StrenghtXI_PRM_SHARPNESS
        '''
        self.set_param('sharpness', sharpness)

    def get_ccMTX00(self):
        '''
        Color Correction Matrix element [0][0]XI_PRM_CC_MATRIX_00
        '''
        return self.get_param('ccMTX00')

    def get_ccMTX00_maximum(self):
        '''
        Color Correction Matrix element [0][0]XI_PRM_CC_MATRIX_00
        '''
        return self.get_param('ccMTX00:max')

    def get_ccMTX00_minimum(self):
        '''
        Color Correction Matrix element [0][0]XI_PRM_CC_MATRIX_00
        '''
        return self.get_param('ccMTX00:min')

    def get_ccMTX00_increment(self):
        '''
        Color Correction Matrix element [0][0]XI_PRM_CC_MATRIX_00
        '''
        return self.get_param('ccMTX00:inc')

    def set_ccMTX00(self, ccMTX00):
        '''
        Color Correction Matrix element [0][0]XI_PRM_CC_MATRIX_00
        '''
        self.set_param('ccMTX00', ccMTX00)

    def get_ccMTX01(self):
        '''
        Color Correction Matrix element [0][1]XI_PRM_CC_MATRIX_01
        '''
        return self.get_param('ccMTX01')

    def get_ccMTX01_maximum(self):
        '''
        Color Correction Matrix element [0][1]XI_PRM_CC_MATRIX_01
        '''
        return self.get_param('ccMTX01:max')

    def get_ccMTX01_minimum(self):
        '''
        Color Correction Matrix element [0][1]XI_PRM_CC_MATRIX_01
        '''
        return self.get_param('ccMTX01:min')

    def get_ccMTX01_increment(self):
        '''
        Color Correction Matrix element [0][1]XI_PRM_CC_MATRIX_01
        '''
        return self.get_param('ccMTX01:inc')

    def set_ccMTX01(self, ccMTX01):
        '''
        Color Correction Matrix element [0][1]XI_PRM_CC_MATRIX_01
        '''
        self.set_param('ccMTX01', ccMTX01)

    def get_ccMTX02(self):
        '''
        Color Correction Matrix element [0][2]XI_PRM_CC_MATRIX_02
        '''
        return self.get_param('ccMTX02')

    def get_ccMTX02_maximum(self):
        '''
        Color Correction Matrix element [0][2]XI_PRM_CC_MATRIX_02
        '''
        return self.get_param('ccMTX02:max')

    def get_ccMTX02_minimum(self):
        '''
        Color Correction Matrix element [0][2]XI_PRM_CC_MATRIX_02
        '''
        return self.get_param('ccMTX02:min')

    def get_ccMTX02_increment(self):
        '''
        Color Correction Matrix element [0][2]XI_PRM_CC_MATRIX_02
        '''
        return self.get_param('ccMTX02:inc')

    def set_ccMTX02(self, ccMTX02):
        '''
        Color Correction Matrix element [0][2]XI_PRM_CC_MATRIX_02
        '''
        self.set_param('ccMTX02', ccMTX02)

    def get_ccMTX03(self):
        '''
        Color Correction Matrix element [0][3]XI_PRM_CC_MATRIX_03
        '''
        return self.get_param('ccMTX03')

    def get_ccMTX03_maximum(self):
        '''
        Color Correction Matrix element [0][3]XI_PRM_CC_MATRIX_03
        '''
        return self.get_param('ccMTX03:max')

    def get_ccMTX03_minimum(self):
        '''
        Color Correction Matrix element [0][3]XI_PRM_CC_MATRIX_03
        '''
        return self.get_param('ccMTX03:min')

    def get_ccMTX03_increment(self):
        '''
        Color Correction Matrix element [0][3]XI_PRM_CC_MATRIX_03
        '''
        return self.get_param('ccMTX03:inc')

    def set_ccMTX03(self, ccMTX03):
        '''
        Color Correction Matrix element [0][3]XI_PRM_CC_MATRIX_03
        '''
        self.set_param('ccMTX03', ccMTX03)

    def get_ccMTX10(self):
        '''
        Color Correction Matrix element [1][0]XI_PRM_CC_MATRIX_10
        '''
        return self.get_param('ccMTX10')

    def get_ccMTX10_maximum(self):
        '''
        Color Correction Matrix element [1][0]XI_PRM_CC_MATRIX_10
        '''
        return self.get_param('ccMTX10:max')

    def get_ccMTX10_minimum(self):
        '''
        Color Correction Matrix element [1][0]XI_PRM_CC_MATRIX_10
        '''
        return self.get_param('ccMTX10:min')

    def get_ccMTX10_increment(self):
        '''
        Color Correction Matrix element [1][0]XI_PRM_CC_MATRIX_10
        '''
        return self.get_param('ccMTX10:inc')

    def set_ccMTX10(self, ccMTX10):
        '''
        Color Correction Matrix element [1][0]XI_PRM_CC_MATRIX_10
        '''
        self.set_param('ccMTX10', ccMTX10)

    def get_ccMTX11(self):
        '''
        Color Correction Matrix element [1][1]XI_PRM_CC_MATRIX_11
        '''
        return self.get_param('ccMTX11')

    def get_ccMTX11_maximum(self):
        '''
        Color Correction Matrix element [1][1]XI_PRM_CC_MATRIX_11
        '''
        return self.get_param('ccMTX11:max')

    def get_ccMTX11_minimum(self):
        '''
        Color Correction Matrix element [1][1]XI_PRM_CC_MATRIX_11
        '''
        return self.get_param('ccMTX11:min')

    def get_ccMTX11_increment(self):
        '''
        Color Correction Matrix element [1][1]XI_PRM_CC_MATRIX_11
        '''
        return self.get_param('ccMTX11:inc')

    def set_ccMTX11(self, ccMTX11):
        '''
        Color Correction Matrix element [1][1]XI_PRM_CC_MATRIX_11
        '''
        self.set_param('ccMTX11', ccMTX11)

    def get_ccMTX12(self):
        '''
        Color Correction Matrix element [1][2]XI_PRM_CC_MATRIX_12
        '''
        return self.get_param('ccMTX12')

    def get_ccMTX12_maximum(self):
        '''
        Color Correction Matrix element [1][2]XI_PRM_CC_MATRIX_12
        '''
        return self.get_param('ccMTX12:max')

    def get_ccMTX12_minimum(self):
        '''
        Color Correction Matrix element [1][2]XI_PRM_CC_MATRIX_12
        '''
        return self.get_param('ccMTX12:min')

    def get_ccMTX12_increment(self):
        '''
        Color Correction Matrix element [1][2]XI_PRM_CC_MATRIX_12
        '''
        return self.get_param('ccMTX12:inc')

    def set_ccMTX12(self, ccMTX12):
        '''
        Color Correction Matrix element [1][2]XI_PRM_CC_MATRIX_12
        '''
        self.set_param('ccMTX12', ccMTX12)

    def get_ccMTX13(self):
        '''
        Color Correction Matrix element [1][3]XI_PRM_CC_MATRIX_13
        '''
        return self.get_param('ccMTX13')

    def get_ccMTX13_maximum(self):
        '''
        Color Correction Matrix element [1][3]XI_PRM_CC_MATRIX_13
        '''
        return self.get_param('ccMTX13:max')

    def get_ccMTX13_minimum(self):
        '''
        Color Correction Matrix element [1][3]XI_PRM_CC_MATRIX_13
        '''
        return self.get_param('ccMTX13:min')

    def get_ccMTX13_increment(self):
        '''
        Color Correction Matrix element [1][3]XI_PRM_CC_MATRIX_13
        '''
        return self.get_param('ccMTX13:inc')

    def set_ccMTX13(self, ccMTX13):
        '''
        Color Correction Matrix element [1][3]XI_PRM_CC_MATRIX_13
        '''
        self.set_param('ccMTX13', ccMTX13)

    def get_ccMTX20(self):
        '''
        Color Correction Matrix element [2][0]XI_PRM_CC_MATRIX_20
        '''
        return self.get_param('ccMTX20')

    def get_ccMTX20_maximum(self):
        '''
        Color Correction Matrix element [2][0]XI_PRM_CC_MATRIX_20
        '''
        return self.get_param('ccMTX20:max')

    def get_ccMTX20_minimum(self):
        '''
        Color Correction Matrix element [2][0]XI_PRM_CC_MATRIX_20
        '''
        return self.get_param('ccMTX20:min')

    def get_ccMTX20_increment(self):
        '''
        Color Correction Matrix element [2][0]XI_PRM_CC_MATRIX_20
        '''
        return self.get_param('ccMTX20:inc')

    def set_ccMTX20(self, ccMTX20):
        '''
        Color Correction Matrix element [2][0]XI_PRM_CC_MATRIX_20
        '''
        self.set_param('ccMTX20', ccMTX20)

    def get_ccMTX21(self):
        '''
        Color Correction Matrix element [2][1]XI_PRM_CC_MATRIX_21
        '''
        return self.get_param('ccMTX21')

    def get_ccMTX21_maximum(self):
        '''
        Color Correction Matrix element [2][1]XI_PRM_CC_MATRIX_21
        '''
        return self.get_param('ccMTX21:max')

    def get_ccMTX21_minimum(self):
        '''
        Color Correction Matrix element [2][1]XI_PRM_CC_MATRIX_21
        '''
        return self.get_param('ccMTX21:min')

    def get_ccMTX21_increment(self):
        '''
        Color Correction Matrix element [2][1]XI_PRM_CC_MATRIX_21
        '''
        return self.get_param('ccMTX21:inc')

    def set_ccMTX21(self, ccMTX21):
        '''
        Color Correction Matrix element [2][1]XI_PRM_CC_MATRIX_21
        '''
        self.set_param('ccMTX21', ccMTX21)

    def get_ccMTX22(self):
        '''
        Color Correction Matrix element [2][2]XI_PRM_CC_MATRIX_22
        '''
        return self.get_param('ccMTX22')

    def get_ccMTX22_maximum(self):
        '''
        Color Correction Matrix element [2][2]XI_PRM_CC_MATRIX_22
        '''
        return self.get_param('ccMTX22:max')

    def get_ccMTX22_minimum(self):
        '''
        Color Correction Matrix element [2][2]XI_PRM_CC_MATRIX_22
        '''
        return self.get_param('ccMTX22:min')

    def get_ccMTX22_increment(self):
        '''
        Color Correction Matrix element [2][2]XI_PRM_CC_MATRIX_22
        '''
        return self.get_param('ccMTX22:inc')

    def set_ccMTX22(self, ccMTX22):
        '''
        Color Correction Matrix element [2][2]XI_PRM_CC_MATRIX_22
        '''
        self.set_param('ccMTX22', ccMTX22)

    def get_ccMTX23(self):
        '''
        Color Correction Matrix element [2][3]XI_PRM_CC_MATRIX_23
        '''
        return self.get_param('ccMTX23')

    def get_ccMTX23_maximum(self):
        '''
        Color Correction Matrix element [2][3]XI_PRM_CC_MATRIX_23
        '''
        return self.get_param('ccMTX23:max')

    def get_ccMTX23_minimum(self):
        '''
        Color Correction Matrix element [2][3]XI_PRM_CC_MATRIX_23
        '''
        return self.get_param('ccMTX23:min')

    def get_ccMTX23_increment(self):
        '''
        Color Correction Matrix element [2][3]XI_PRM_CC_MATRIX_23
        '''
        return self.get_param('ccMTX23:inc')

    def set_ccMTX23(self, ccMTX23):
        '''
        Color Correction Matrix element [2][3]XI_PRM_CC_MATRIX_23
        '''
        self.set_param('ccMTX23', ccMTX23)

    def get_ccMTX30(self):
        '''
        Color Correction Matrix element [3][0]XI_PRM_CC_MATRIX_30
        '''
        return self.get_param('ccMTX30')

    def get_ccMTX30_maximum(self):
        '''
        Color Correction Matrix element [3][0]XI_PRM_CC_MATRIX_30
        '''
        return self.get_param('ccMTX30:max')

    def get_ccMTX30_minimum(self):
        '''
        Color Correction Matrix element [3][0]XI_PRM_CC_MATRIX_30
        '''
        return self.get_param('ccMTX30:min')

    def get_ccMTX30_increment(self):
        '''
        Color Correction Matrix element [3][0]XI_PRM_CC_MATRIX_30
        '''
        return self.get_param('ccMTX30:inc')

    def set_ccMTX30(self, ccMTX30):
        '''
        Color Correction Matrix element [3][0]XI_PRM_CC_MATRIX_30
        '''
        self.set_param('ccMTX30', ccMTX30)

    def get_ccMTX31(self):
        '''
        Color Correction Matrix element [3][1]XI_PRM_CC_MATRIX_31
        '''
        return self.get_param('ccMTX31')

    def get_ccMTX31_maximum(self):
        '''
        Color Correction Matrix element [3][1]XI_PRM_CC_MATRIX_31
        '''
        return self.get_param('ccMTX31:max')

    def get_ccMTX31_minimum(self):
        '''
        Color Correction Matrix element [3][1]XI_PRM_CC_MATRIX_31
        '''
        return self.get_param('ccMTX31:min')

    def get_ccMTX31_increment(self):
        '''
        Color Correction Matrix element [3][1]XI_PRM_CC_MATRIX_31
        '''
        return self.get_param('ccMTX31:inc')

    def set_ccMTX31(self, ccMTX31):
        '''
        Color Correction Matrix element [3][1]XI_PRM_CC_MATRIX_31
        '''
        self.set_param('ccMTX31', ccMTX31)

    def get_ccMTX32(self):
        '''
        Color Correction Matrix element [3][2]XI_PRM_CC_MATRIX_32
        '''
        return self.get_param('ccMTX32')

    def get_ccMTX32_maximum(self):
        '''
        Color Correction Matrix element [3][2]XI_PRM_CC_MATRIX_32
        '''
        return self.get_param('ccMTX32:max')

    def get_ccMTX32_minimum(self):
        '''
        Color Correction Matrix element [3][2]XI_PRM_CC_MATRIX_32
        '''
        return self.get_param('ccMTX32:min')

    def get_ccMTX32_increment(self):
        '''
        Color Correction Matrix element [3][2]XI_PRM_CC_MATRIX_32
        '''
        return self.get_param('ccMTX32:inc')

    def set_ccMTX32(self, ccMTX32):
        '''
        Color Correction Matrix element [3][2]XI_PRM_CC_MATRIX_32
        '''
        self.set_param('ccMTX32', ccMTX32)

    def get_ccMTX33(self):
        '''
        Color Correction Matrix element [3][3]XI_PRM_CC_MATRIX_33
        '''
        return self.get_param('ccMTX33')

    def get_ccMTX33_maximum(self):
        '''
        Color Correction Matrix element [3][3]XI_PRM_CC_MATRIX_33
        '''
        return self.get_param('ccMTX33:max')

    def get_ccMTX33_minimum(self):
        '''
        Color Correction Matrix element [3][3]XI_PRM_CC_MATRIX_33
        '''
        return self.get_param('ccMTX33:min')

    def get_ccMTX33_increment(self):
        '''
        Color Correction Matrix element [3][3]XI_PRM_CC_MATRIX_33
        '''
        return self.get_param('ccMTX33:inc')

    def set_ccMTX33(self, ccMTX33):
        '''
        Color Correction Matrix element [3][3]XI_PRM_CC_MATRIX_33
        '''
        self.set_param('ccMTX33', ccMTX33)

    def get_defccMTX(self):
        '''
        Set default Color Correction MatrixXI_PRM_DEFAULT_CC_MATRIX
        '''
        return self.get_param('defccMTX')

    def get_defccMTX_maximum(self):
        '''
        Set default Color Correction MatrixXI_PRM_DEFAULT_CC_MATRIX
        '''
        return self.get_param('defccMTX:max')

    def get_defccMTX_minimum(self):
        '''
        Set default Color Correction MatrixXI_PRM_DEFAULT_CC_MATRIX
        '''
        return self.get_param('defccMTX:min')

    def get_defccMTX_increment(self):
        '''
        Set default Color Correction MatrixXI_PRM_DEFAULT_CC_MATRIX
        '''
        return self.get_param('defccMTX:inc')

    def set_defccMTX(self, defccMTX):
        '''
        Set default Color Correction MatrixXI_PRM_DEFAULT_CC_MATRIX
        '''
        self.set_param('defccMTX', defccMTX)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Device IO
#-------------------------------------------------------------------------------------------------------------------

    def get_trigger_source(self):
        '''
        Defines source of trigger.XI_PRM_TRG_SOURCE
        '''
        return self.get_param('trigger_source')

    def get_trigger_source_maximum(self):
        '''
        Defines source of trigger.XI_PRM_TRG_SOURCE
        '''
        return self.get_param('trigger_source:max')

    def get_trigger_source_minimum(self):
        '''
        Defines source of trigger.XI_PRM_TRG_SOURCE
        '''
        return self.get_param('trigger_source:min')

    def get_trigger_source_increment(self):
        '''
        Defines source of trigger.XI_PRM_TRG_SOURCE
        '''
        return self.get_param('trigger_source:inc')

    def set_trigger_source(self, trigger_source):
        '''
        Defines source of trigger.XI_PRM_TRG_SOURCE
        '''
        self.set_param('trigger_source', trigger_source)

    def get_trigger_software(self):
        '''
        Generates an internal trigger. XI_PRM_TRG_SOURCE must be set to TRG_SOFTWARE.XI_PRM_TRG_SOFTWARE
        '''
        return self.get_param('trigger_software')

    def get_trigger_software_maximum(self):
        '''
        Generates an internal trigger. XI_PRM_TRG_SOURCE must be set to TRG_SOFTWARE.XI_PRM_TRG_SOFTWARE
        '''
        return self.get_param('trigger_software:max')

    def get_trigger_software_minimum(self):
        '''
        Generates an internal trigger. XI_PRM_TRG_SOURCE must be set to TRG_SOFTWARE.XI_PRM_TRG_SOFTWARE
        '''
        return self.get_param('trigger_software:min')

    def get_trigger_software_increment(self):
        '''
        Generates an internal trigger. XI_PRM_TRG_SOURCE must be set to TRG_SOFTWARE.XI_PRM_TRG_SOFTWARE
        '''
        return self.get_param('trigger_software:inc')

    def set_trigger_software(self, trigger_software):
        '''
        Generates an internal trigger. XI_PRM_TRG_SOURCE must be set to TRG_SOFTWARE.XI_PRM_TRG_SOFTWARE
        '''
        self.set_param('trigger_software', trigger_software)

    def get_trigger_selector(self):
        '''
        Selects the type of trigger.XI_PRM_TRG_SELECTOR
        '''
        return self.get_param('trigger_selector')

    def get_trigger_selector_maximum(self):
        '''
        Selects the type of trigger.XI_PRM_TRG_SELECTOR
        '''
        return self.get_param('trigger_selector:max')

    def get_trigger_selector_minimum(self):
        '''
        Selects the type of trigger.XI_PRM_TRG_SELECTOR
        '''
        return self.get_param('trigger_selector:min')

    def get_trigger_selector_increment(self):
        '''
        Selects the type of trigger.XI_PRM_TRG_SELECTOR
        '''
        return self.get_param('trigger_selector:inc')

    def set_trigger_selector(self, trigger_selector):
        '''
        Selects the type of trigger.XI_PRM_TRG_SELECTOR
        '''
        self.set_param('trigger_selector', trigger_selector)

    def get_trigger_overlap(self):
        '''
        The mode of Trigger Overlap. This influences of trigger acception/rejection policyXI_PRM_TRG_OVERLAP
        '''
        return self.get_param('trigger_overlap')

    def get_trigger_overlap_maximum(self):
        '''
        The mode of Trigger Overlap. This influences of trigger acception/rejection policyXI_PRM_TRG_OVERLAP
        '''
        return self.get_param('trigger_overlap:max')

    def get_trigger_overlap_minimum(self):
        '''
        The mode of Trigger Overlap. This influences of trigger acception/rejection policyXI_PRM_TRG_OVERLAP
        '''
        return self.get_param('trigger_overlap:min')

    def get_trigger_overlap_increment(self):
        '''
        The mode of Trigger Overlap. This influences of trigger acception/rejection policyXI_PRM_TRG_OVERLAP
        '''
        return self.get_param('trigger_overlap:inc')

    def set_trigger_overlap(self, trigger_overlap):
        '''
        The mode of Trigger Overlap. This influences of trigger acception/rejection policyXI_PRM_TRG_OVERLAP
        '''
        self.set_param('trigger_overlap', trigger_overlap)

    def get_acq_frame_burst_count(self):
        '''
        Sets number of frames acquired by burst. This burst is used only if trigger is set to FrameBurstStartXI_PRM_ACQ_FRAME_BURST_COUNT
        '''
        return self.get_param('acq_frame_burst_count')

    def get_acq_frame_burst_count_maximum(self):
        '''
        Sets number of frames acquired by burst. This burst is used only if trigger is set to FrameBurstStartXI_PRM_ACQ_FRAME_BURST_COUNT
        '''
        return self.get_param('acq_frame_burst_count:max')

    def get_acq_frame_burst_count_minimum(self):
        '''
        Sets number of frames acquired by burst. This burst is used only if trigger is set to FrameBurstStartXI_PRM_ACQ_FRAME_BURST_COUNT
        '''
        return self.get_param('acq_frame_burst_count:min')

    def get_acq_frame_burst_count_increment(self):
        '''
        Sets number of frames acquired by burst. This burst is used only if trigger is set to FrameBurstStartXI_PRM_ACQ_FRAME_BURST_COUNT
        '''
        return self.get_param('acq_frame_burst_count:inc')

    def set_acq_frame_burst_count(self, acq_frame_burst_count):
        '''
        Sets number of frames acquired by burst. This burst is used only if trigger is set to FrameBurstStartXI_PRM_ACQ_FRAME_BURST_COUNT
        '''
        self.set_param('acq_frame_burst_count', acq_frame_burst_count)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: GPIO Setup
#-------------------------------------------------------------------------------------------------------------------

    def get_gpi_selector(self):
        '''
        Selects GPIXI_PRM_GPI_SELECTOR
        '''
        return self.get_param('gpi_selector')

    def get_gpi_selector_maximum(self):
        '''
        Selects GPIXI_PRM_GPI_SELECTOR
        '''
        return self.get_param('gpi_selector:max')

    def get_gpi_selector_minimum(self):
        '''
        Selects GPIXI_PRM_GPI_SELECTOR
        '''
        return self.get_param('gpi_selector:min')

    def get_gpi_selector_increment(self):
        '''
        Selects GPIXI_PRM_GPI_SELECTOR
        '''
        return self.get_param('gpi_selector:inc')

    def set_gpi_selector(self, gpi_selector):
        '''
        Selects GPIXI_PRM_GPI_SELECTOR
        '''
        self.set_param('gpi_selector', gpi_selector)

    def get_gpi_mode(self):
        '''
        Defines GPI functionalityXI_PRM_GPI_MODE
        '''
        return self.get_param('gpi_mode')

    def get_gpi_mode_maximum(self):
        '''
        Defines GPI functionalityXI_PRM_GPI_MODE
        '''
        return self.get_param('gpi_mode:max')

    def get_gpi_mode_minimum(self):
        '''
        Defines GPI functionalityXI_PRM_GPI_MODE
        '''
        return self.get_param('gpi_mode:min')

    def get_gpi_mode_increment(self):
        '''
        Defines GPI functionalityXI_PRM_GPI_MODE
        '''
        return self.get_param('gpi_mode:inc')

    def set_gpi_mode(self, gpi_mode):
        '''
        Defines GPI functionalityXI_PRM_GPI_MODE
        '''
        self.set_param('gpi_mode', gpi_mode)

    def get_gpi_level(self):
        '''
        GPI levelXI_PRM_GPI_LEVEL
        '''
        return self.get_param('gpi_level')

    def get_gpi_level_maximum(self):
        '''
        GPI levelXI_PRM_GPI_LEVEL
        '''
        return self.get_param('gpi_level:max')

    def get_gpi_level_minimum(self):
        '''
        GPI levelXI_PRM_GPI_LEVEL
        '''
        return self.get_param('gpi_level:min')

    def get_gpi_level_increment(self):
        '''
        GPI levelXI_PRM_GPI_LEVEL
        '''
        return self.get_param('gpi_level:inc')

    def get_gpo_selector(self):
        '''
        Selects GPOXI_PRM_GPO_SELECTOR
        '''
        return self.get_param('gpo_selector')

    def get_gpo_selector_maximum(self):
        '''
        Selects GPOXI_PRM_GPO_SELECTOR
        '''
        return self.get_param('gpo_selector:max')

    def get_gpo_selector_minimum(self):
        '''
        Selects GPOXI_PRM_GPO_SELECTOR
        '''
        return self.get_param('gpo_selector:min')

    def get_gpo_selector_increment(self):
        '''
        Selects GPOXI_PRM_GPO_SELECTOR
        '''
        return self.get_param('gpo_selector:inc')

    def set_gpo_selector(self, gpo_selector):
        '''
        Selects GPOXI_PRM_GPO_SELECTOR
        '''
        self.set_param('gpo_selector', gpo_selector)

    def get_gpo_mode(self):
        '''
        Defines GPO functionalityXI_PRM_GPO_MODE
        '''
        return self.get_param('gpo_mode')

    def get_gpo_mode_maximum(self):
        '''
        Defines GPO functionalityXI_PRM_GPO_MODE
        '''
        return self.get_param('gpo_mode:max')

    def get_gpo_mode_minimum(self):
        '''
        Defines GPO functionalityXI_PRM_GPO_MODE
        '''
        return self.get_param('gpo_mode:min')

    def get_gpo_mode_increment(self):
        '''
        Defines GPO functionalityXI_PRM_GPO_MODE
        '''
        return self.get_param('gpo_mode:inc')

    def set_gpo_mode(self, gpo_mode):
        '''
        Defines GPO functionalityXI_PRM_GPO_MODE
        '''
        self.set_param('gpo_mode', gpo_mode)

    def get_led_selector(self):
        '''
        Selects LEDXI_PRM_LED_SELECTOR
        '''
        return self.get_param('led_selector')

    def get_led_selector_maximum(self):
        '''
        Selects LEDXI_PRM_LED_SELECTOR
        '''
        return self.get_param('led_selector:max')

    def get_led_selector_minimum(self):
        '''
        Selects LEDXI_PRM_LED_SELECTOR
        '''
        return self.get_param('led_selector:min')

    def get_led_selector_increment(self):
        '''
        Selects LEDXI_PRM_LED_SELECTOR
        '''
        return self.get_param('led_selector:inc')

    def set_led_selector(self, led_selector):
        '''
        Selects LEDXI_PRM_LED_SELECTOR
        '''
        self.set_param('led_selector', led_selector)

    def get_led_mode(self):
        '''
        Defines LED functionalityXI_PRM_LED_MODE
        '''
        return self.get_param('led_mode')

    def get_led_mode_maximum(self):
        '''
        Defines LED functionalityXI_PRM_LED_MODE
        '''
        return self.get_param('led_mode:max')

    def get_led_mode_minimum(self):
        '''
        Defines LED functionalityXI_PRM_LED_MODE
        '''
        return self.get_param('led_mode:min')

    def get_led_mode_increment(self):
        '''
        Defines LED functionalityXI_PRM_LED_MODE
        '''
        return self.get_param('led_mode:inc')

    def set_led_mode(self, led_mode):
        '''
        Defines LED functionalityXI_PRM_LED_MODE
        '''
        self.set_param('led_mode', led_mode)

    def is_dbnc_en(self):
        '''
        Enable/Disable debounce to selected GPIXI_PRM_DEBOUNCE_EN
        '''
        return self.get_param('dbnc_en')

    def enable_dbnc_en(self):
        '''
        Enable/Disable debounce to selected GPIXI_PRM_DEBOUNCE_EN
        '''
        self.set_param('dbnc_en', True)

    def disable_dbnc_en(self):
        '''
        Enable/Disable debounce to selected GPIXI_PRM_DEBOUNCE_EN
        '''
        self.set_param('dbnc_en', False)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Debounce Setup
#-------------------------------------------------------------------------------------------------------------------

    def get_dbnc_t0(self):
        '''
        Debounce time (x * 10us)XI_PRM_DEBOUNCE_T0
        '''
        return self.get_param('dbnc_t0')

    def get_dbnc_t0_maximum(self):
        '''
        Debounce time (x * 10us)XI_PRM_DEBOUNCE_T0
        '''
        return self.get_param('dbnc_t0:max')

    def get_dbnc_t0_minimum(self):
        '''
        Debounce time (x * 10us)XI_PRM_DEBOUNCE_T0
        '''
        return self.get_param('dbnc_t0:min')

    def get_dbnc_t0_increment(self):
        '''
        Debounce time (x * 10us)XI_PRM_DEBOUNCE_T0
        '''
        return self.get_param('dbnc_t0:inc')

    def set_dbnc_t0(self, dbnc_t0):
        '''
        Debounce time (x * 10us)XI_PRM_DEBOUNCE_T0
        '''
        self.set_param('dbnc_t0', dbnc_t0)

    def get_dbnc_t1(self):
        '''
        Debounce time (x * 10us)XI_PRM_DEBOUNCE_T1
        '''
        return self.get_param('dbnc_t1')

    def get_dbnc_t1_maximum(self):
        '''
        Debounce time (x * 10us)XI_PRM_DEBOUNCE_T1
        '''
        return self.get_param('dbnc_t1:max')

    def get_dbnc_t1_minimum(self):
        '''
        Debounce time (x * 10us)XI_PRM_DEBOUNCE_T1
        '''
        return self.get_param('dbnc_t1:min')

    def get_dbnc_t1_increment(self):
        '''
        Debounce time (x * 10us)XI_PRM_DEBOUNCE_T1
        '''
        return self.get_param('dbnc_t1:inc')

    def set_dbnc_t1(self, dbnc_t1):
        '''
        Debounce time (x * 10us)XI_PRM_DEBOUNCE_T1
        '''
        self.set_param('dbnc_t1', dbnc_t1)

    def get_dbnc_pol(self):
        '''
        Debounce polarity (pol = 1 t0 - falling edge, t1 - rising edge)XI_PRM_DEBOUNCE_POL
        '''
        return self.get_param('dbnc_pol')

    def get_dbnc_pol_maximum(self):
        '''
        Debounce polarity (pol = 1 t0 - falling edge, t1 - rising edge)XI_PRM_DEBOUNCE_POL
        '''
        return self.get_param('dbnc_pol:max')

    def get_dbnc_pol_minimum(self):
        '''
        Debounce polarity (pol = 1 t0 - falling edge, t1 - rising edge)XI_PRM_DEBOUNCE_POL
        '''
        return self.get_param('dbnc_pol:min')

    def get_dbnc_pol_increment(self):
        '''
        Debounce polarity (pol = 1 t0 - falling edge, t1 - rising edge)XI_PRM_DEBOUNCE_POL
        '''
        return self.get_param('dbnc_pol:inc')

    def set_dbnc_pol(self, dbnc_pol):
        '''
        Debounce polarity (pol = 1 t0 - falling edge, t1 - rising edge)XI_PRM_DEBOUNCE_POL
        '''
        self.set_param('dbnc_pol', dbnc_pol)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Lens Control
#-------------------------------------------------------------------------------------------------------------------

    def is_lens_mode(self):
        '''
        Status of lens control interface. This shall be set to XI_ON before any Lens operations.XI_PRM_LENS_MODE
        '''
        return self.get_param('lens_mode')

    def enable_lens_mode(self):
        '''
        Status of lens control interface. This shall be set to XI_ON before any Lens operations.XI_PRM_LENS_MODE
        '''
        self.set_param('lens_mode', True)

    def disable_lens_mode(self):
        '''
        Status of lens control interface. This shall be set to XI_ON before any Lens operations.XI_PRM_LENS_MODE
        '''
        self.set_param('lens_mode', False)

    def get_lens_aperture_value(self):
        '''
        Current lens aperture value in stops. Examples: 2.8, 4, 5.6, 8, 11XI_PRM_LENS_APERTURE_VALUE
        '''
        return self.get_param('lens_aperture_value')

    def get_lens_aperture_value_maximum(self):
        '''
        Current lens aperture value in stops. Examples: 2.8, 4, 5.6, 8, 11XI_PRM_LENS_APERTURE_VALUE
        '''
        return self.get_param('lens_aperture_value:max')

    def get_lens_aperture_value_minimum(self):
        '''
        Current lens aperture value in stops. Examples: 2.8, 4, 5.6, 8, 11XI_PRM_LENS_APERTURE_VALUE
        '''
        return self.get_param('lens_aperture_value:min')

    def get_lens_aperture_value_increment(self):
        '''
        Current lens aperture value in stops. Examples: 2.8, 4, 5.6, 8, 11XI_PRM_LENS_APERTURE_VALUE
        '''
        return self.get_param('lens_aperture_value:inc')

    def set_lens_aperture_value(self, lens_aperture_value):
        '''
        Current lens aperture value in stops. Examples: 2.8, 4, 5.6, 8, 11XI_PRM_LENS_APERTURE_VALUE
        '''
        self.set_param('lens_aperture_value', lens_aperture_value)

    def get_lens_focus_movement_value(self):
        '''
        Lens current focus movement value to be used by XI_PRM_LENS_FOCUS_MOVE in motor steps.XI_PRM_LENS_FOCUS_MOVEMENT_VALUE
        '''
        return self.get_param('lens_focus_movement_value')

    def get_lens_focus_movement_value_maximum(self):
        '''
        Lens current focus movement value to be used by XI_PRM_LENS_FOCUS_MOVE in motor steps.XI_PRM_LENS_FOCUS_MOVEMENT_VALUE
        '''
        return self.get_param('lens_focus_movement_value:max')

    def get_lens_focus_movement_value_minimum(self):
        '''
        Lens current focus movement value to be used by XI_PRM_LENS_FOCUS_MOVE in motor steps.XI_PRM_LENS_FOCUS_MOVEMENT_VALUE
        '''
        return self.get_param('lens_focus_movement_value:min')

    def get_lens_focus_movement_value_increment(self):
        '''
        Lens current focus movement value to be used by XI_PRM_LENS_FOCUS_MOVE in motor steps.XI_PRM_LENS_FOCUS_MOVEMENT_VALUE
        '''
        return self.get_param('lens_focus_movement_value:inc')

    def set_lens_focus_movement_value(self, lens_focus_movement_value):
        '''
        Lens current focus movement value to be used by XI_PRM_LENS_FOCUS_MOVE in motor steps.XI_PRM_LENS_FOCUS_MOVEMENT_VALUE
        '''
        self.set_param('lens_focus_movement_value', lens_focus_movement_value)

    def get_lens_focus_move(self):
        '''
        Moves lens focus motor by steps set in XI_PRM_LENS_FOCUS_MOVEMENT_VALUE.XI_PRM_LENS_FOCUS_MOVE
        '''
        return self.get_param('lens_focus_move')

    def get_lens_focus_move_maximum(self):
        '''
        Moves lens focus motor by steps set in XI_PRM_LENS_FOCUS_MOVEMENT_VALUE.XI_PRM_LENS_FOCUS_MOVE
        '''
        return self.get_param('lens_focus_move:max')

    def get_lens_focus_move_minimum(self):
        '''
        Moves lens focus motor by steps set in XI_PRM_LENS_FOCUS_MOVEMENT_VALUE.XI_PRM_LENS_FOCUS_MOVE
        '''
        return self.get_param('lens_focus_move:min')

    def get_lens_focus_move_increment(self):
        '''
        Moves lens focus motor by steps set in XI_PRM_LENS_FOCUS_MOVEMENT_VALUE.XI_PRM_LENS_FOCUS_MOVE
        '''
        return self.get_param('lens_focus_move:inc')

    def set_lens_focus_move(self, lens_focus_move):
        '''
        Moves lens focus motor by steps set in XI_PRM_LENS_FOCUS_MOVEMENT_VALUE.XI_PRM_LENS_FOCUS_MOVE
        '''
        self.set_param('lens_focus_move', lens_focus_move)

    def get_lens_focus_distance(self):
        '''
        Lens focus distance in cm.XI_PRM_LENS_FOCUS_DISTANCE
        '''
        return self.get_param('lens_focus_distance')

    def get_lens_focus_distance_maximum(self):
        '''
        Lens focus distance in cm.XI_PRM_LENS_FOCUS_DISTANCE
        '''
        return self.get_param('lens_focus_distance:max')

    def get_lens_focus_distance_minimum(self):
        '''
        Lens focus distance in cm.XI_PRM_LENS_FOCUS_DISTANCE
        '''
        return self.get_param('lens_focus_distance:min')

    def get_lens_focus_distance_increment(self):
        '''
        Lens focus distance in cm.XI_PRM_LENS_FOCUS_DISTANCE
        '''
        return self.get_param('lens_focus_distance:inc')

    def get_lens_focal_length(self):
        '''
        Lens focal distance in mm.XI_PRM_LENS_FOCAL_LENGTH
        '''
        return self.get_param('lens_focal_length')

    def get_lens_focal_length_maximum(self):
        '''
        Lens focal distance in mm.XI_PRM_LENS_FOCAL_LENGTH
        '''
        return self.get_param('lens_focal_length:max')

    def get_lens_focal_length_minimum(self):
        '''
        Lens focal distance in mm.XI_PRM_LENS_FOCAL_LENGTH
        '''
        return self.get_param('lens_focal_length:min')

    def get_lens_focal_length_increment(self):
        '''
        Lens focal distance in mm.XI_PRM_LENS_FOCAL_LENGTH
        '''
        return self.get_param('lens_focal_length:inc')

    def get_lens_feature_selector(self):
        '''
        Selects the current feature which is accessible by XI_PRM_LENS_FEATURE.XI_PRM_LENS_FEATURE_SELECTOR
        '''
        return self.get_param('lens_feature_selector')

    def get_lens_feature_selector_maximum(self):
        '''
        Selects the current feature which is accessible by XI_PRM_LENS_FEATURE.XI_PRM_LENS_FEATURE_SELECTOR
        '''
        return self.get_param('lens_feature_selector:max')

    def get_lens_feature_selector_minimum(self):
        '''
        Selects the current feature which is accessible by XI_PRM_LENS_FEATURE.XI_PRM_LENS_FEATURE_SELECTOR
        '''
        return self.get_param('lens_feature_selector:min')

    def get_lens_feature_selector_increment(self):
        '''
        Selects the current feature which is accessible by XI_PRM_LENS_FEATURE.XI_PRM_LENS_FEATURE_SELECTOR
        '''
        return self.get_param('lens_feature_selector:inc')

    def set_lens_feature_selector(self, lens_feature_selector):
        '''
        Selects the current feature which is accessible by XI_PRM_LENS_FEATURE.XI_PRM_LENS_FEATURE_SELECTOR
        '''
        self.set_param('lens_feature_selector', lens_feature_selector)

    def get_lens_feature(self):
        '''
        Allows access to lens feature value currently selected by XI_PRM_LENS_FEATURE_SELECTOR.XI_PRM_LENS_FEATURE
        '''
        return self.get_param('lens_feature')

    def get_lens_feature_maximum(self):
        '''
        Allows access to lens feature value currently selected by XI_PRM_LENS_FEATURE_SELECTOR.XI_PRM_LENS_FEATURE
        '''
        return self.get_param('lens_feature:max')

    def get_lens_feature_minimum(self):
        '''
        Allows access to lens feature value currently selected by XI_PRM_LENS_FEATURE_SELECTOR.XI_PRM_LENS_FEATURE
        '''
        return self.get_param('lens_feature:min')

    def get_lens_feature_increment(self):
        '''
        Allows access to lens feature value currently selected by XI_PRM_LENS_FEATURE_SELECTOR.XI_PRM_LENS_FEATURE
        '''
        return self.get_param('lens_feature:inc')

    def set_lens_feature(self, lens_feature):
        '''
        Allows access to lens feature value currently selected by XI_PRM_LENS_FEATURE_SELECTOR.XI_PRM_LENS_FEATURE
        '''
        self.set_param('lens_feature', lens_feature)

    def get_lens_comm_data(self,buffer_size=256):
        '''
        Write/Read data sequences to/from lensXI_PRM_LENS_COMM_DATA
        '''
        return self.get_param('lens_comm_data',buffer_size)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Device info parameters
#-------------------------------------------------------------------------------------------------------------------

    def get_device_name(self,buffer_size=256):
        '''
        Return device nameXI_PRM_DEVICE_NAME
        '''
        return self.get_param('device_name',buffer_size)

    def get_device_type(self,buffer_size=256):
        '''
        Return device typeXI_PRM_DEVICE_TYPE
        '''
        return self.get_param('device_type',buffer_size)

    def get_device_model_id(self):
        '''
        Return device model idXI_PRM_DEVICE_MODEL_ID
        '''
        return self.get_param('device_model_id')

    def get_device_model_id_maximum(self):
        '''
        Return device model idXI_PRM_DEVICE_MODEL_ID
        '''
        return self.get_param('device_model_id:max')

    def get_device_model_id_minimum(self):
        '''
        Return device model idXI_PRM_DEVICE_MODEL_ID
        '''
        return self.get_param('device_model_id:min')

    def get_device_model_id_increment(self):
        '''
        Return device model idXI_PRM_DEVICE_MODEL_ID
        '''
        return self.get_param('device_model_id:inc')

    def get_sensor_model_id(self):
        '''
        Return device sensor model idXI_PRM_SENSOR_MODEL_ID
        '''
        return self.get_param('sensor_model_id')

    def get_sensor_model_id_maximum(self):
        '''
        Return device sensor model idXI_PRM_SENSOR_MODEL_ID
        '''
        return self.get_param('sensor_model_id:max')

    def get_sensor_model_id_minimum(self):
        '''
        Return device sensor model idXI_PRM_SENSOR_MODEL_ID
        '''
        return self.get_param('sensor_model_id:min')

    def get_sensor_model_id_increment(self):
        '''
        Return device sensor model idXI_PRM_SENSOR_MODEL_ID
        '''
        return self.get_param('sensor_model_id:inc')

    def get_device_sn(self,buffer_size=256):
        '''
        Return device serial numberXI_PRM_DEVICE_SN
        '''
        return self.get_param('device_sn',buffer_size)

    def get_device_sens_sn(self,buffer_size=256):
        '''
        Return sensor serial numberXI_PRM_DEVICE_SENS_SN
        '''
        return self.get_param('device_sens_sn',buffer_size)

    def get_device_id(self,buffer_size=256):
        '''
        Return unique device IDXI_PRM_DEVICE_ID
        '''
        return self.get_param('device_id',buffer_size)

    def get_device_inst_path(self,buffer_size=256):
        '''
        Return device system instance path.XI_PRM_DEVICE_INSTANCE_PATH
        '''
        return self.get_param('device_inst_path',buffer_size)

    def get_device_loc_path(self,buffer_size=256):
        '''
        Represents the location of the device in the device tree.XI_PRM_DEVICE_LOCATION_PATH
        '''
        return self.get_param('device_loc_path',buffer_size)

    def get_device_user_id(self,buffer_size=256):
        '''
        Return custom ID of camera.XI_PRM_DEVICE_USER_ID
        '''
        return self.get_param('device_user_id',buffer_size)

    def get_device_manifest(self,buffer_size=256):
        '''
        Return device capability description XML.XI_PRM_DEVICE_MANIFEST
        '''
        return self.get_param('device_manifest',buffer_size)

    def get_image_user_data(self):
        '''
        User image data at image header to track parameters synchronization.XI_PRM_IMAGE_USER_DATA
        '''
        return self.get_param('image_user_data')

    def get_image_user_data_maximum(self):
        '''
        User image data at image header to track parameters synchronization.XI_PRM_IMAGE_USER_DATA
        '''
        return self.get_param('image_user_data:max')

    def get_image_user_data_minimum(self):
        '''
        User image data at image header to track parameters synchronization.XI_PRM_IMAGE_USER_DATA
        '''
        return self.get_param('image_user_data:min')

    def get_image_user_data_increment(self):
        '''
        User image data at image header to track parameters synchronization.XI_PRM_IMAGE_USER_DATA
        '''
        return self.get_param('image_user_data:inc')

    def set_image_user_data(self, image_user_data):
        '''
        User image data at image header to track parameters synchronization.XI_PRM_IMAGE_USER_DATA
        '''
        self.set_param('image_user_data', image_user_data)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Device acquisition settings
#-------------------------------------------------------------------------------------------------------------------

    def get_imgdataformatrgb32alpha(self):
        '''
        The alpha channel of RGB32 output image format.XI_PRM_IMAGE_DATA_FORMAT_RGB32_ALPHA
        '''
        return self.get_param('imgdataformatrgb32alpha')

    def get_imgdataformatrgb32alpha_maximum(self):
        '''
        The alpha channel of RGB32 output image format.XI_PRM_IMAGE_DATA_FORMAT_RGB32_ALPHA
        '''
        return self.get_param('imgdataformatrgb32alpha:max')

    def get_imgdataformatrgb32alpha_minimum(self):
        '''
        The alpha channel of RGB32 output image format.XI_PRM_IMAGE_DATA_FORMAT_RGB32_ALPHA
        '''
        return self.get_param('imgdataformatrgb32alpha:min')

    def get_imgdataformatrgb32alpha_increment(self):
        '''
        The alpha channel of RGB32 output image format.XI_PRM_IMAGE_DATA_FORMAT_RGB32_ALPHA
        '''
        return self.get_param('imgdataformatrgb32alpha:inc')

    def set_imgdataformatrgb32alpha(self, imgdataformatrgb32alpha):
        '''
        The alpha channel of RGB32 output image format.XI_PRM_IMAGE_DATA_FORMAT_RGB32_ALPHA
        '''
        self.set_param('imgdataformatrgb32alpha', imgdataformatrgb32alpha)

    def get_imgpayloadsize(self):
        '''
        Buffer size in bytes sufficient for output image returned by xiGetImageXI_PRM_IMAGE_PAYLOAD_SIZE
        '''
        return self.get_param('imgpayloadsize')

    def get_imgpayloadsize_maximum(self):
        '''
        Buffer size in bytes sufficient for output image returned by xiGetImageXI_PRM_IMAGE_PAYLOAD_SIZE
        '''
        return self.get_param('imgpayloadsize:max')

    def get_imgpayloadsize_minimum(self):
        '''
        Buffer size in bytes sufficient for output image returned by xiGetImageXI_PRM_IMAGE_PAYLOAD_SIZE
        '''
        return self.get_param('imgpayloadsize:min')

    def get_imgpayloadsize_increment(self):
        '''
        Buffer size in bytes sufficient for output image returned by xiGetImageXI_PRM_IMAGE_PAYLOAD_SIZE
        '''
        return self.get_param('imgpayloadsize:inc')

    def get_transport_pixel_format(self):
        '''
        Current format of pixels on transport layer.XI_PRM_TRANSPORT_PIXEL_FORMAT
        '''
        return self.get_param('transport_pixel_format')

    def get_transport_pixel_format_maximum(self):
        '''
        Current format of pixels on transport layer.XI_PRM_TRANSPORT_PIXEL_FORMAT
        '''
        return self.get_param('transport_pixel_format:max')

    def get_transport_pixel_format_minimum(self):
        '''
        Current format of pixels on transport layer.XI_PRM_TRANSPORT_PIXEL_FORMAT
        '''
        return self.get_param('transport_pixel_format:min')

    def get_transport_pixel_format_increment(self):
        '''
        Current format of pixels on transport layer.XI_PRM_TRANSPORT_PIXEL_FORMAT
        '''
        return self.get_param('transport_pixel_format:inc')

    def set_transport_pixel_format(self, transport_pixel_format):
        '''
        Current format of pixels on transport layer.XI_PRM_TRANSPORT_PIXEL_FORMAT
        '''
        self.set_param('transport_pixel_format', transport_pixel_format)

    def get_transport_data_target(self):
        '''
        Target selector for data - CPU RAM or GPU RAMXI_PRM_TRANSPORT_DATA_TARGET
        '''
        return self.get_param('transport_data_target')

    def get_transport_data_target_maximum(self):
        '''
        Target selector for data - CPU RAM or GPU RAMXI_PRM_TRANSPORT_DATA_TARGET
        '''
        return self.get_param('transport_data_target:max')

    def get_transport_data_target_minimum(self):
        '''
        Target selector for data - CPU RAM or GPU RAMXI_PRM_TRANSPORT_DATA_TARGET
        '''
        return self.get_param('transport_data_target:min')

    def get_transport_data_target_increment(self):
        '''
        Target selector for data - CPU RAM or GPU RAMXI_PRM_TRANSPORT_DATA_TARGET
        '''
        return self.get_param('transport_data_target:inc')

    def set_transport_data_target(self, transport_data_target):
        '''
        Target selector for data - CPU RAM or GPU RAMXI_PRM_TRANSPORT_DATA_TARGET
        '''
        self.set_param('transport_data_target', transport_data_target)

    def get_sensor_clock_freq_hz(self):
        '''
        Sensor clock frequency in Hz.XI_PRM_SENSOR_CLOCK_FREQ_HZ
        '''
        return self.get_param('sensor_clock_freq_hz')

    def get_sensor_clock_freq_hz_maximum(self):
        '''
        Sensor clock frequency in Hz.XI_PRM_SENSOR_CLOCK_FREQ_HZ
        '''
        return self.get_param('sensor_clock_freq_hz:max')

    def get_sensor_clock_freq_hz_minimum(self):
        '''
        Sensor clock frequency in Hz.XI_PRM_SENSOR_CLOCK_FREQ_HZ
        '''
        return self.get_param('sensor_clock_freq_hz:min')

    def get_sensor_clock_freq_hz_increment(self):
        '''
        Sensor clock frequency in Hz.XI_PRM_SENSOR_CLOCK_FREQ_HZ
        '''
        return self.get_param('sensor_clock_freq_hz:inc')

    def set_sensor_clock_freq_hz(self, sensor_clock_freq_hz):
        '''
        Sensor clock frequency in Hz.XI_PRM_SENSOR_CLOCK_FREQ_HZ
        '''
        self.set_param('sensor_clock_freq_hz', sensor_clock_freq_hz)

    def get_sensor_clock_freq_index(self):
        '''
        Sensor clock frequency index. Sensor with selected frequencies have possibility to set the frequency only by this index.XI_PRM_SENSOR_CLOCK_FREQ_INDEX
        '''
        return self.get_param('sensor_clock_freq_index')

    def get_sensor_clock_freq_index_maximum(self):
        '''
        Sensor clock frequency index. Sensor with selected frequencies have possibility to set the frequency only by this index.XI_PRM_SENSOR_CLOCK_FREQ_INDEX
        '''
        return self.get_param('sensor_clock_freq_index:max')

    def get_sensor_clock_freq_index_minimum(self):
        '''
        Sensor clock frequency index. Sensor with selected frequencies have possibility to set the frequency only by this index.XI_PRM_SENSOR_CLOCK_FREQ_INDEX
        '''
        return self.get_param('sensor_clock_freq_index:min')

    def get_sensor_clock_freq_index_increment(self):
        '''
        Sensor clock frequency index. Sensor with selected frequencies have possibility to set the frequency only by this index.XI_PRM_SENSOR_CLOCK_FREQ_INDEX
        '''
        return self.get_param('sensor_clock_freq_index:inc')

    def set_sensor_clock_freq_index(self, sensor_clock_freq_index):
        '''
        Sensor clock frequency index. Sensor with selected frequencies have possibility to set the frequency only by this index.XI_PRM_SENSOR_CLOCK_FREQ_INDEX
        '''
        self.set_param('sensor_clock_freq_index', sensor_clock_freq_index)

    def get_sensor_output_channel_count(self):
        '''
        Number of output channels from sensor used for data transfer.XI_PRM_SENSOR_OUTPUT_CHANNEL_COUNT
        '''
        return self.get_param('sensor_output_channel_count')

    def get_sensor_output_channel_count_maximum(self):
        '''
        Number of output channels from sensor used for data transfer.XI_PRM_SENSOR_OUTPUT_CHANNEL_COUNT
        '''
        return self.get_param('sensor_output_channel_count:max')

    def get_sensor_output_channel_count_minimum(self):
        '''
        Number of output channels from sensor used for data transfer.XI_PRM_SENSOR_OUTPUT_CHANNEL_COUNT
        '''
        return self.get_param('sensor_output_channel_count:min')

    def get_sensor_output_channel_count_increment(self):
        '''
        Number of output channels from sensor used for data transfer.XI_PRM_SENSOR_OUTPUT_CHANNEL_COUNT
        '''
        return self.get_param('sensor_output_channel_count:inc')

    def set_sensor_output_channel_count(self, sensor_output_channel_count):
        '''
        Number of output channels from sensor used for data transfer.XI_PRM_SENSOR_OUTPUT_CHANNEL_COUNT
        '''
        self.set_param('sensor_output_channel_count', sensor_output_channel_count)

    def get_framerate(self):
        '''
        Define framerate in HzXI_PRM_FRAMERATE
        '''
        return self.get_param('framerate')

    def get_framerate_maximum(self):
        '''
        Define framerate in HzXI_PRM_FRAMERATE
        '''
        return self.get_param('framerate:max')

    def get_framerate_minimum(self):
        '''
        Define framerate in HzXI_PRM_FRAMERATE
        '''
        return self.get_param('framerate:min')

    def get_framerate_increment(self):
        '''
        Define framerate in HzXI_PRM_FRAMERATE
        '''
        return self.get_param('framerate:inc')

    def set_framerate(self, framerate):
        '''
        Define framerate in HzXI_PRM_FRAMERATE
        '''
        self.set_param('framerate', framerate)

    def get_counter_selector(self):
        '''
        Select counterXI_PRM_COUNTER_SELECTOR
        '''
        return self.get_param('counter_selector')

    def get_counter_selector_maximum(self):
        '''
        Select counterXI_PRM_COUNTER_SELECTOR
        '''
        return self.get_param('counter_selector:max')

    def get_counter_selector_minimum(self):
        '''
        Select counterXI_PRM_COUNTER_SELECTOR
        '''
        return self.get_param('counter_selector:min')

    def get_counter_selector_increment(self):
        '''
        Select counterXI_PRM_COUNTER_SELECTOR
        '''
        return self.get_param('counter_selector:inc')

    def set_counter_selector(self, counter_selector):
        '''
        Select counterXI_PRM_COUNTER_SELECTOR
        '''
        self.set_param('counter_selector', counter_selector)

    def get_counter_value(self):
        '''
        Counter statusXI_PRM_COUNTER_VALUE
        '''
        return self.get_param('counter_value')

    def get_counter_value_maximum(self):
        '''
        Counter statusXI_PRM_COUNTER_VALUE
        '''
        return self.get_param('counter_value:max')

    def get_counter_value_minimum(self):
        '''
        Counter statusXI_PRM_COUNTER_VALUE
        '''
        return self.get_param('counter_value:min')

    def get_counter_value_increment(self):
        '''
        Counter statusXI_PRM_COUNTER_VALUE
        '''
        return self.get_param('counter_value:inc')

    def get_acq_timing_mode(self):
        '''
        Type of sensor frames timing.XI_PRM_ACQ_TIMING_MODE
        '''
        return self.get_param('acq_timing_mode')

    def get_acq_timing_mode_maximum(self):
        '''
        Type of sensor frames timing.XI_PRM_ACQ_TIMING_MODE
        '''
        return self.get_param('acq_timing_mode:max')

    def get_acq_timing_mode_minimum(self):
        '''
        Type of sensor frames timing.XI_PRM_ACQ_TIMING_MODE
        '''
        return self.get_param('acq_timing_mode:min')

    def get_acq_timing_mode_increment(self):
        '''
        Type of sensor frames timing.XI_PRM_ACQ_TIMING_MODE
        '''
        return self.get_param('acq_timing_mode:inc')

    def set_acq_timing_mode(self, acq_timing_mode):
        '''
        Type of sensor frames timing.XI_PRM_ACQ_TIMING_MODE
        '''
        self.set_param('acq_timing_mode', acq_timing_mode)

    def get_available_bandwidth(self):
        '''
        Measure and return available interface bandwidth(int Megabits)XI_PRM_AVAILABLE_BANDWIDTH
        '''
        return self.get_param('available_bandwidth')

    def get_available_bandwidth_maximum(self):
        '''
        Measure and return available interface bandwidth(int Megabits)XI_PRM_AVAILABLE_BANDWIDTH
        '''
        return self.get_param('available_bandwidth:max')

    def get_available_bandwidth_minimum(self):
        '''
        Measure and return available interface bandwidth(int Megabits)XI_PRM_AVAILABLE_BANDWIDTH
        '''
        return self.get_param('available_bandwidth:min')

    def get_available_bandwidth_increment(self):
        '''
        Measure and return available interface bandwidth(int Megabits)XI_PRM_AVAILABLE_BANDWIDTH
        '''
        return self.get_param('available_bandwidth:inc')

    def get_buffer_policy(self):
        '''
        Data move policyXI_PRM_BUFFER_POLICY
        '''
        return self.get_param('buffer_policy')

    def get_buffer_policy_maximum(self):
        '''
        Data move policyXI_PRM_BUFFER_POLICY
        '''
        return self.get_param('buffer_policy:max')

    def get_buffer_policy_minimum(self):
        '''
        Data move policyXI_PRM_BUFFER_POLICY
        '''
        return self.get_param('buffer_policy:min')

    def get_buffer_policy_increment(self):
        '''
        Data move policyXI_PRM_BUFFER_POLICY
        '''
        return self.get_param('buffer_policy:inc')

    def set_buffer_policy(self, buffer_policy):
        '''
        Data move policyXI_PRM_BUFFER_POLICY
        '''
        self.set_param('buffer_policy', buffer_policy)

    def is_LUTEnable(self):
        '''
        Activates LUT.XI_PRM_LUT_EN
        '''
        return self.get_param('LUTEnable')

    def enable_LUTEnable(self):
        '''
        Activates LUT.XI_PRM_LUT_EN
        '''
        self.set_param('LUTEnable', True)

    def disable_LUTEnable(self):
        '''
        Activates LUT.XI_PRM_LUT_EN
        '''
        self.set_param('LUTEnable', False)

    def get_LUTIndex(self):
        '''
        Control the index (offset) of the coefficient to access in the LUT.XI_PRM_LUT_INDEX
        '''
        return self.get_param('LUTIndex')

    def get_LUTIndex_maximum(self):
        '''
        Control the index (offset) of the coefficient to access in the LUT.XI_PRM_LUT_INDEX
        '''
        return self.get_param('LUTIndex:max')

    def get_LUTIndex_minimum(self):
        '''
        Control the index (offset) of the coefficient to access in the LUT.XI_PRM_LUT_INDEX
        '''
        return self.get_param('LUTIndex:min')

    def get_LUTIndex_increment(self):
        '''
        Control the index (offset) of the coefficient to access in the LUT.XI_PRM_LUT_INDEX
        '''
        return self.get_param('LUTIndex:inc')

    def set_LUTIndex(self, LUTIndex):
        '''
        Control the index (offset) of the coefficient to access in the LUT.XI_PRM_LUT_INDEX
        '''
        self.set_param('LUTIndex', LUTIndex)

    def get_LUTValue(self):
        '''
        Value at entry LUTIndex of the LUTXI_PRM_LUT_VALUE
        '''
        return self.get_param('LUTValue')

    def get_LUTValue_maximum(self):
        '''
        Value at entry LUTIndex of the LUTXI_PRM_LUT_VALUE
        '''
        return self.get_param('LUTValue:max')

    def get_LUTValue_minimum(self):
        '''
        Value at entry LUTIndex of the LUTXI_PRM_LUT_VALUE
        '''
        return self.get_param('LUTValue:min')

    def get_LUTValue_increment(self):
        '''
        Value at entry LUTIndex of the LUTXI_PRM_LUT_VALUE
        '''
        return self.get_param('LUTValue:inc')

    def set_LUTValue(self, LUTValue):
        '''
        Value at entry LUTIndex of the LUTXI_PRM_LUT_VALUE
        '''
        self.set_param('LUTValue', LUTValue)

    def get_trigger_delay(self):
        '''
        Specifies the delay in microseconds (us) to apply after the trigger reception before activating it.XI_PRM_TRG_DELAY
        '''
        return self.get_param('trigger_delay')

    def get_trigger_delay_maximum(self):
        '''
        Specifies the delay in microseconds (us) to apply after the trigger reception before activating it.XI_PRM_TRG_DELAY
        '''
        return self.get_param('trigger_delay:max')

    def get_trigger_delay_minimum(self):
        '''
        Specifies the delay in microseconds (us) to apply after the trigger reception before activating it.XI_PRM_TRG_DELAY
        '''
        return self.get_param('trigger_delay:min')

    def get_trigger_delay_increment(self):
        '''
        Specifies the delay in microseconds (us) to apply after the trigger reception before activating it.XI_PRM_TRG_DELAY
        '''
        return self.get_param('trigger_delay:inc')

    def set_trigger_delay(self, trigger_delay):
        '''
        Specifies the delay in microseconds (us) to apply after the trigger reception before activating it.XI_PRM_TRG_DELAY
        '''
        self.set_param('trigger_delay', trigger_delay)

    def get_ts_rst_mode(self):
        '''
        Defines how time stamp reset engine will be armedXI_PRM_TS_RST_MODE
        '''
        return self.get_param('ts_rst_mode')

    def get_ts_rst_mode_maximum(self):
        '''
        Defines how time stamp reset engine will be armedXI_PRM_TS_RST_MODE
        '''
        return self.get_param('ts_rst_mode:max')

    def get_ts_rst_mode_minimum(self):
        '''
        Defines how time stamp reset engine will be armedXI_PRM_TS_RST_MODE
        '''
        return self.get_param('ts_rst_mode:min')

    def get_ts_rst_mode_increment(self):
        '''
        Defines how time stamp reset engine will be armedXI_PRM_TS_RST_MODE
        '''
        return self.get_param('ts_rst_mode:inc')

    def set_ts_rst_mode(self, ts_rst_mode):
        '''
        Defines how time stamp reset engine will be armedXI_PRM_TS_RST_MODE
        '''
        self.set_param('ts_rst_mode', ts_rst_mode)

    def get_ts_rst_source(self):
        '''
        Defines which source will be used for timestamp reset. Writing this parameter will trigger settings of engine (arming)XI_PRM_TS_RST_SOURCE
        '''
        return self.get_param('ts_rst_source')

    def get_ts_rst_source_maximum(self):
        '''
        Defines which source will be used for timestamp reset. Writing this parameter will trigger settings of engine (arming)XI_PRM_TS_RST_SOURCE
        '''
        return self.get_param('ts_rst_source:max')

    def get_ts_rst_source_minimum(self):
        '''
        Defines which source will be used for timestamp reset. Writing this parameter will trigger settings of engine (arming)XI_PRM_TS_RST_SOURCE
        '''
        return self.get_param('ts_rst_source:min')

    def get_ts_rst_source_increment(self):
        '''
        Defines which source will be used for timestamp reset. Writing this parameter will trigger settings of engine (arming)XI_PRM_TS_RST_SOURCE
        '''
        return self.get_param('ts_rst_source:inc')

    def set_ts_rst_source(self, ts_rst_source):
        '''
        Defines which source will be used for timestamp reset. Writing this parameter will trigger settings of engine (arming)XI_PRM_TS_RST_SOURCE
        '''
        self.set_param('ts_rst_source', ts_rst_source)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Extended Device parameters
#-------------------------------------------------------------------------------------------------------------------

    def is_isexist(self):
        '''
        Returns 1 if camera connected and works properly.XI_PRM_IS_DEVICE_EXIST
        '''
        return self.get_param('isexist')

    def get_acq_buffer_size(self):
        '''
        Acquisition buffer size in buffer_size_unit. Default bytes.XI_PRM_ACQ_BUFFER_SIZE
        '''
        return self.get_param('acq_buffer_size')

    def get_acq_buffer_size_maximum(self):
        '''
        Acquisition buffer size in buffer_size_unit. Default bytes.XI_PRM_ACQ_BUFFER_SIZE
        '''
        return self.get_param('acq_buffer_size:max')

    def get_acq_buffer_size_minimum(self):
        '''
        Acquisition buffer size in buffer_size_unit. Default bytes.XI_PRM_ACQ_BUFFER_SIZE
        '''
        return self.get_param('acq_buffer_size:min')

    def get_acq_buffer_size_increment(self):
        '''
        Acquisition buffer size in buffer_size_unit. Default bytes.XI_PRM_ACQ_BUFFER_SIZE
        '''
        return self.get_param('acq_buffer_size:inc')

    def set_acq_buffer_size(self, acq_buffer_size):
        '''
        Acquisition buffer size in buffer_size_unit. Default bytes.XI_PRM_ACQ_BUFFER_SIZE
        '''
        self.set_param('acq_buffer_size', acq_buffer_size)

    def get_acq_buffer_size_unit(self):
        '''
        Acquisition buffer size unit in bytes. Default 1. E.g. Value 1024 means that buffer_size is in KiBytesXI_PRM_ACQ_BUFFER_SIZE_UNIT
        '''
        return self.get_param('acq_buffer_size_unit')

    def get_acq_buffer_size_unit_maximum(self):
        '''
        Acquisition buffer size unit in bytes. Default 1. E.g. Value 1024 means that buffer_size is in KiBytesXI_PRM_ACQ_BUFFER_SIZE_UNIT
        '''
        return self.get_param('acq_buffer_size_unit:max')

    def get_acq_buffer_size_unit_minimum(self):
        '''
        Acquisition buffer size unit in bytes. Default 1. E.g. Value 1024 means that buffer_size is in KiBytesXI_PRM_ACQ_BUFFER_SIZE_UNIT
        '''
        return self.get_param('acq_buffer_size_unit:min')

    def get_acq_buffer_size_unit_increment(self):
        '''
        Acquisition buffer size unit in bytes. Default 1. E.g. Value 1024 means that buffer_size is in KiBytesXI_PRM_ACQ_BUFFER_SIZE_UNIT
        '''
        return self.get_param('acq_buffer_size_unit:inc')

    def set_acq_buffer_size_unit(self, acq_buffer_size_unit):
        '''
        Acquisition buffer size unit in bytes. Default 1. E.g. Value 1024 means that buffer_size is in KiBytesXI_PRM_ACQ_BUFFER_SIZE_UNIT
        '''
        self.set_param('acq_buffer_size_unit', acq_buffer_size_unit)

    def get_acq_transport_buffer_size(self):
        '''
        Acquisition transport buffer size in bytesXI_PRM_ACQ_TRANSPORT_BUFFER_SIZE
        '''
        return self.get_param('acq_transport_buffer_size')

    def get_acq_transport_buffer_size_maximum(self):
        '''
        Acquisition transport buffer size in bytesXI_PRM_ACQ_TRANSPORT_BUFFER_SIZE
        '''
        return self.get_param('acq_transport_buffer_size:max')

    def get_acq_transport_buffer_size_minimum(self):
        '''
        Acquisition transport buffer size in bytesXI_PRM_ACQ_TRANSPORT_BUFFER_SIZE
        '''
        return self.get_param('acq_transport_buffer_size:min')

    def get_acq_transport_buffer_size_increment(self):
        '''
        Acquisition transport buffer size in bytesXI_PRM_ACQ_TRANSPORT_BUFFER_SIZE
        '''
        return self.get_param('acq_transport_buffer_size:inc')

    def set_acq_transport_buffer_size(self, acq_transport_buffer_size):
        '''
        Acquisition transport buffer size in bytesXI_PRM_ACQ_TRANSPORT_BUFFER_SIZE
        '''
        self.set_param('acq_transport_buffer_size', acq_transport_buffer_size)

    def get_acq_transport_packet_size(self):
        '''
        Acquisition transport packet size in bytesXI_PRM_ACQ_TRANSPORT_PACKET_SIZE
        '''
        return self.get_param('acq_transport_packet_size')

    def get_acq_transport_packet_size_maximum(self):
        '''
        Acquisition transport packet size in bytesXI_PRM_ACQ_TRANSPORT_PACKET_SIZE
        '''
        return self.get_param('acq_transport_packet_size:max')

    def get_acq_transport_packet_size_minimum(self):
        '''
        Acquisition transport packet size in bytesXI_PRM_ACQ_TRANSPORT_PACKET_SIZE
        '''
        return self.get_param('acq_transport_packet_size:min')

    def get_acq_transport_packet_size_increment(self):
        '''
        Acquisition transport packet size in bytesXI_PRM_ACQ_TRANSPORT_PACKET_SIZE
        '''
        return self.get_param('acq_transport_packet_size:inc')

    def set_acq_transport_packet_size(self, acq_transport_packet_size):
        '''
        Acquisition transport packet size in bytesXI_PRM_ACQ_TRANSPORT_PACKET_SIZE
        '''
        self.set_param('acq_transport_packet_size', acq_transport_packet_size)

    def get_buffers_queue_size(self):
        '''
        Queue of field/frame buffersXI_PRM_BUFFERS_QUEUE_SIZE
        '''
        return self.get_param('buffers_queue_size')

    def get_buffers_queue_size_maximum(self):
        '''
        Queue of field/frame buffersXI_PRM_BUFFERS_QUEUE_SIZE
        '''
        return self.get_param('buffers_queue_size:max')

    def get_buffers_queue_size_minimum(self):
        '''
        Queue of field/frame buffersXI_PRM_BUFFERS_QUEUE_SIZE
        '''
        return self.get_param('buffers_queue_size:min')

    def get_buffers_queue_size_increment(self):
        '''
        Queue of field/frame buffersXI_PRM_BUFFERS_QUEUE_SIZE
        '''
        return self.get_param('buffers_queue_size:inc')

    def set_buffers_queue_size(self, buffers_queue_size):
        '''
        Queue of field/frame buffersXI_PRM_BUFFERS_QUEUE_SIZE
        '''
        self.set_param('buffers_queue_size', buffers_queue_size)

    def get_acq_transport_buffer_commit(self):
        '''
        Number of buffers to commit to low levelXI_PRM_ACQ_TRANSPORT_BUFFER_COMMIT
        '''
        return self.get_param('acq_transport_buffer_commit')

    def get_acq_transport_buffer_commit_maximum(self):
        '''
        Number of buffers to commit to low levelXI_PRM_ACQ_TRANSPORT_BUFFER_COMMIT
        '''
        return self.get_param('acq_transport_buffer_commit:max')

    def get_acq_transport_buffer_commit_minimum(self):
        '''
        Number of buffers to commit to low levelXI_PRM_ACQ_TRANSPORT_BUFFER_COMMIT
        '''
        return self.get_param('acq_transport_buffer_commit:min')

    def get_acq_transport_buffer_commit_increment(self):
        '''
        Number of buffers to commit to low levelXI_PRM_ACQ_TRANSPORT_BUFFER_COMMIT
        '''
        return self.get_param('acq_transport_buffer_commit:inc')

    def set_acq_transport_buffer_commit(self, acq_transport_buffer_commit):
        '''
        Number of buffers to commit to low levelXI_PRM_ACQ_TRANSPORT_BUFFER_COMMIT
        '''
        self.set_param('acq_transport_buffer_commit', acq_transport_buffer_commit)

    def is_recent_frame(self):
        '''
        GetImage returns most recent frameXI_PRM_RECENT_FRAME
        '''
        return self.get_param('recent_frame')

    def enable_recent_frame(self):
        '''
        GetImage returns most recent frameXI_PRM_RECENT_FRAME
        '''
        self.set_param('recent_frame', True)

    def disable_recent_frame(self):
        '''
        GetImage returns most recent frameXI_PRM_RECENT_FRAME
        '''
        self.set_param('recent_frame', False)

    def get_device_reset(self):
        '''
        Resets the camera to default state.XI_PRM_DEVICE_RESET
        '''
        return self.get_param('device_reset')

    def get_device_reset_maximum(self):
        '''
        Resets the camera to default state.XI_PRM_DEVICE_RESET
        '''
        return self.get_param('device_reset:max')

    def get_device_reset_minimum(self):
        '''
        Resets the camera to default state.XI_PRM_DEVICE_RESET
        '''
        return self.get_param('device_reset:min')

    def get_device_reset_increment(self):
        '''
        Resets the camera to default state.XI_PRM_DEVICE_RESET
        '''
        return self.get_param('device_reset:inc')

    def set_device_reset(self, device_reset):
        '''
        Resets the camera to default state.XI_PRM_DEVICE_RESET
        '''
        self.set_param('device_reset', device_reset)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Sensor Defects Correction
#-------------------------------------------------------------------------------------------------------------------

    def get_column_fpn_correction(self):
        '''
        Correction of column FPNXI_PRM_COLUMN_FPN_CORRECTION
        '''
        return self.get_param('column_fpn_correction')

    def get_column_fpn_correction_maximum(self):
        '''
        Correction of column FPNXI_PRM_COLUMN_FPN_CORRECTION
        '''
        return self.get_param('column_fpn_correction:max')

    def get_column_fpn_correction_minimum(self):
        '''
        Correction of column FPNXI_PRM_COLUMN_FPN_CORRECTION
        '''
        return self.get_param('column_fpn_correction:min')

    def get_column_fpn_correction_increment(self):
        '''
        Correction of column FPNXI_PRM_COLUMN_FPN_CORRECTION
        '''
        return self.get_param('column_fpn_correction:inc')

    def set_column_fpn_correction(self, column_fpn_correction):
        '''
        Correction of column FPNXI_PRM_COLUMN_FPN_CORRECTION
        '''
        self.set_param('column_fpn_correction', column_fpn_correction)

    def get_row_fpn_correction(self):
        '''
        Correction of row FPNXI_PRM_ROW_FPN_CORRECTION
        '''
        return self.get_param('row_fpn_correction')

    def get_row_fpn_correction_maximum(self):
        '''
        Correction of row FPNXI_PRM_ROW_FPN_CORRECTION
        '''
        return self.get_param('row_fpn_correction:max')

    def get_row_fpn_correction_minimum(self):
        '''
        Correction of row FPNXI_PRM_ROW_FPN_CORRECTION
        '''
        return self.get_param('row_fpn_correction:min')

    def get_row_fpn_correction_increment(self):
        '''
        Correction of row FPNXI_PRM_ROW_FPN_CORRECTION
        '''
        return self.get_param('row_fpn_correction:inc')

    def set_row_fpn_correction(self, row_fpn_correction):
        '''
        Correction of row FPNXI_PRM_ROW_FPN_CORRECTION
        '''
        self.set_param('row_fpn_correction', row_fpn_correction)

    def get_image_correction_selector(self):
        '''
        Select image correction functionXI_PRM_IMAGE_CORRECTION_SELECTOR
        '''
        return self.get_param('image_correction_selector')

    def get_image_correction_selector_maximum(self):
        '''
        Select image correction functionXI_PRM_IMAGE_CORRECTION_SELECTOR
        '''
        return self.get_param('image_correction_selector:max')

    def get_image_correction_selector_minimum(self):
        '''
        Select image correction functionXI_PRM_IMAGE_CORRECTION_SELECTOR
        '''
        return self.get_param('image_correction_selector:min')

    def get_image_correction_selector_increment(self):
        '''
        Select image correction functionXI_PRM_IMAGE_CORRECTION_SELECTOR
        '''
        return self.get_param('image_correction_selector:inc')

    def set_image_correction_selector(self, image_correction_selector):
        '''
        Select image correction functionXI_PRM_IMAGE_CORRECTION_SELECTOR
        '''
        self.set_param('image_correction_selector', image_correction_selector)

    def get_image_correction_value(self):
        '''
        Select image correction selected function valueXI_PRM_IMAGE_CORRECTION_VALUE
        '''
        return self.get_param('image_correction_value')

    def get_image_correction_value_maximum(self):
        '''
        Select image correction selected function valueXI_PRM_IMAGE_CORRECTION_VALUE
        '''
        return self.get_param('image_correction_value:max')

    def get_image_correction_value_minimum(self):
        '''
        Select image correction selected function valueXI_PRM_IMAGE_CORRECTION_VALUE
        '''
        return self.get_param('image_correction_value:min')

    def get_image_correction_value_increment(self):
        '''
        Select image correction selected function valueXI_PRM_IMAGE_CORRECTION_VALUE
        '''
        return self.get_param('image_correction_value:inc')

    def set_image_correction_value(self, image_correction_value):
        '''
        Select image correction selected function valueXI_PRM_IMAGE_CORRECTION_VALUE
        '''
        self.set_param('image_correction_value', image_correction_value)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Sensor features
#-------------------------------------------------------------------------------------------------------------------

    def get_sensor_mode(self):
        '''
        Current sensor mode. Allows to select sensor mode by one integer. Setting of this parameter affects: image dimensions and downsampling.XI_PRM_SENSOR_MODE
        '''
        return self.get_param('sensor_mode')

    def get_sensor_mode_maximum(self):
        '''
        Current sensor mode. Allows to select sensor mode by one integer. Setting of this parameter affects: image dimensions and downsampling.XI_PRM_SENSOR_MODE
        '''
        return self.get_param('sensor_mode:max')

    def get_sensor_mode_minimum(self):
        '''
        Current sensor mode. Allows to select sensor mode by one integer. Setting of this parameter affects: image dimensions and downsampling.XI_PRM_SENSOR_MODE
        '''
        return self.get_param('sensor_mode:min')

    def get_sensor_mode_increment(self):
        '''
        Current sensor mode. Allows to select sensor mode by one integer. Setting of this parameter affects: image dimensions and downsampling.XI_PRM_SENSOR_MODE
        '''
        return self.get_param('sensor_mode:inc')

    def set_sensor_mode(self, sensor_mode):
        '''
        Current sensor mode. Allows to select sensor mode by one integer. Setting of this parameter affects: image dimensions and downsampling.XI_PRM_SENSOR_MODE
        '''
        self.set_param('sensor_mode', sensor_mode)

    def is_hdr(self):
        '''
        Enable High Dynamic Range feature.XI_PRM_HDR
        '''
        return self.get_param('hdr')

    def enable_hdr(self):
        '''
        Enable High Dynamic Range feature.XI_PRM_HDR
        '''
        self.set_param('hdr', True)

    def disable_hdr(self):
        '''
        Enable High Dynamic Range feature.XI_PRM_HDR
        '''
        self.set_param('hdr', False)

    def get_hdr_kneepoint_count(self):
        '''
        The number of kneepoints in the PWLR.XI_PRM_HDR_KNEEPOINT_COUNT
        '''
        return self.get_param('hdr_kneepoint_count')

    def get_hdr_kneepoint_count_maximum(self):
        '''
        The number of kneepoints in the PWLR.XI_PRM_HDR_KNEEPOINT_COUNT
        '''
        return self.get_param('hdr_kneepoint_count:max')

    def get_hdr_kneepoint_count_minimum(self):
        '''
        The number of kneepoints in the PWLR.XI_PRM_HDR_KNEEPOINT_COUNT
        '''
        return self.get_param('hdr_kneepoint_count:min')

    def get_hdr_kneepoint_count_increment(self):
        '''
        The number of kneepoints in the PWLR.XI_PRM_HDR_KNEEPOINT_COUNT
        '''
        return self.get_param('hdr_kneepoint_count:inc')

    def set_hdr_kneepoint_count(self, hdr_kneepoint_count):
        '''
        The number of kneepoints in the PWLR.XI_PRM_HDR_KNEEPOINT_COUNT
        '''
        self.set_param('hdr_kneepoint_count', hdr_kneepoint_count)

    def get_hdr_t1(self):
        '''
        position of first kneepoint(in % of XI_PRM_EXPOSURE)XI_PRM_HDR_T1
        '''
        return self.get_param('hdr_t1')

    def get_hdr_t1_maximum(self):
        '''
        position of first kneepoint(in % of XI_PRM_EXPOSURE)XI_PRM_HDR_T1
        '''
        return self.get_param('hdr_t1:max')

    def get_hdr_t1_minimum(self):
        '''
        position of first kneepoint(in % of XI_PRM_EXPOSURE)XI_PRM_HDR_T1
        '''
        return self.get_param('hdr_t1:min')

    def get_hdr_t1_increment(self):
        '''
        position of first kneepoint(in % of XI_PRM_EXPOSURE)XI_PRM_HDR_T1
        '''
        return self.get_param('hdr_t1:inc')

    def set_hdr_t1(self, hdr_t1):
        '''
        position of first kneepoint(in % of XI_PRM_EXPOSURE)XI_PRM_HDR_T1
        '''
        self.set_param('hdr_t1', hdr_t1)

    def get_hdr_t2(self):
        '''
        position of second kneepoint (in % of XI_PRM_EXPOSURE)XI_PRM_HDR_T2
        '''
        return self.get_param('hdr_t2')

    def get_hdr_t2_maximum(self):
        '''
        position of second kneepoint (in % of XI_PRM_EXPOSURE)XI_PRM_HDR_T2
        '''
        return self.get_param('hdr_t2:max')

    def get_hdr_t2_minimum(self):
        '''
        position of second kneepoint (in % of XI_PRM_EXPOSURE)XI_PRM_HDR_T2
        '''
        return self.get_param('hdr_t2:min')

    def get_hdr_t2_increment(self):
        '''
        position of second kneepoint (in % of XI_PRM_EXPOSURE)XI_PRM_HDR_T2
        '''
        return self.get_param('hdr_t2:inc')

    def set_hdr_t2(self, hdr_t2):
        '''
        position of second kneepoint (in % of XI_PRM_EXPOSURE)XI_PRM_HDR_T2
        '''
        self.set_param('hdr_t2', hdr_t2)

    def get_hdr_kneepoint1(self):
        '''
        value of first kneepoint (% of sensor saturation)XI_PRM_KNEEPOINT1
        '''
        return self.get_param('hdr_kneepoint1')

    def get_hdr_kneepoint1_maximum(self):
        '''
        value of first kneepoint (% of sensor saturation)XI_PRM_KNEEPOINT1
        '''
        return self.get_param('hdr_kneepoint1:max')

    def get_hdr_kneepoint1_minimum(self):
        '''
        value of first kneepoint (% of sensor saturation)XI_PRM_KNEEPOINT1
        '''
        return self.get_param('hdr_kneepoint1:min')

    def get_hdr_kneepoint1_increment(self):
        '''
        value of first kneepoint (% of sensor saturation)XI_PRM_KNEEPOINT1
        '''
        return self.get_param('hdr_kneepoint1:inc')

    def set_hdr_kneepoint1(self, hdr_kneepoint1):
        '''
        value of first kneepoint (% of sensor saturation)XI_PRM_KNEEPOINT1
        '''
        self.set_param('hdr_kneepoint1', hdr_kneepoint1)

    def get_hdr_kneepoint2(self):
        '''
        value of second kneepoint (% of sensor saturation)XI_PRM_KNEEPOINT2
        '''
        return self.get_param('hdr_kneepoint2')

    def get_hdr_kneepoint2_maximum(self):
        '''
        value of second kneepoint (% of sensor saturation)XI_PRM_KNEEPOINT2
        '''
        return self.get_param('hdr_kneepoint2:max')

    def get_hdr_kneepoint2_minimum(self):
        '''
        value of second kneepoint (% of sensor saturation)XI_PRM_KNEEPOINT2
        '''
        return self.get_param('hdr_kneepoint2:min')

    def get_hdr_kneepoint2_increment(self):
        '''
        value of second kneepoint (% of sensor saturation)XI_PRM_KNEEPOINT2
        '''
        return self.get_param('hdr_kneepoint2:inc')

    def set_hdr_kneepoint2(self, hdr_kneepoint2):
        '''
        value of second kneepoint (% of sensor saturation)XI_PRM_KNEEPOINT2
        '''
        self.set_param('hdr_kneepoint2', hdr_kneepoint2)

    def get_image_black_level(self):
        '''
        Last image black level counts. Can be used for Offline processing to recall it.XI_PRM_IMAGE_BLACK_LEVEL
        '''
        return self.get_param('image_black_level')

    def get_image_black_level_maximum(self):
        '''
        Last image black level counts. Can be used for Offline processing to recall it.XI_PRM_IMAGE_BLACK_LEVEL
        '''
        return self.get_param('image_black_level:max')

    def get_image_black_level_minimum(self):
        '''
        Last image black level counts. Can be used for Offline processing to recall it.XI_PRM_IMAGE_BLACK_LEVEL
        '''
        return self.get_param('image_black_level:min')

    def get_image_black_level_increment(self):
        '''
        Last image black level counts. Can be used for Offline processing to recall it.XI_PRM_IMAGE_BLACK_LEVEL
        '''
        return self.get_param('image_black_level:inc')

    def set_image_black_level(self, image_black_level):
        '''
        Last image black level counts. Can be used for Offline processing to recall it.XI_PRM_IMAGE_BLACK_LEVEL
        '''
        self.set_param('image_black_level', image_black_level)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Version info
#-------------------------------------------------------------------------------------------------------------------

    def get_api_version(self,buffer_size=256):
        '''
        Returns version of API.XI_PRM_API_VERSION
        '''
        return self.get_param('api_version',buffer_size)

    def get_drv_version(self,buffer_size=256):
        '''
        Returns version of current device driver.XI_PRM_DRV_VERSION
        '''
        return self.get_param('drv_version',buffer_size)

    def get_version_mcu1(self,buffer_size=256):
        '''
        Returns version of MCU1 firmware.XI_PRM_MCU1_VERSION
        '''
        return self.get_param('version_mcu1',buffer_size)

    def get_version_mcu2(self,buffer_size=256):
        '''
        Returns version of MCU2 firmware.XI_PRM_MCU2_VERSION
        '''
        return self.get_param('version_mcu2',buffer_size)

    def get_version_mcu3(self,buffer_size=256):
        '''
        Returns version of MCU3 firmware.XI_PRM_MCU3_VERSION
        '''
        return self.get_param('version_mcu3',buffer_size)

    def get_version_fpga1(self,buffer_size=256):
        '''
        Returns version of FPGA1 firmware.XI_PRM_FPGA1_VERSION
        '''
        return self.get_param('version_fpga1',buffer_size)

    def get_version_xmlman(self,buffer_size=256):
        '''
        Returns version of XML manifest.XI_PRM_XMLMAN_VERSION
        '''
        return self.get_param('version_xmlman',buffer_size)

    def get_hw_revision(self,buffer_size=256):
        '''
        Returns hardware revision number.XI_PRM_HW_REVISION
        '''
        return self.get_param('hw_revision',buffer_size)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: API features
#-------------------------------------------------------------------------------------------------------------------

    def get_debug_level(self):
        '''
        Set debug levelXI_PRM_DEBUG_LEVEL
        '''
        return self.get_param('debug_level')

    def get_debug_level_maximum(self):
        '''
        Set debug levelXI_PRM_DEBUG_LEVEL
        '''
        return self.get_param('debug_level:max')

    def get_debug_level_minimum(self):
        '''
        Set debug levelXI_PRM_DEBUG_LEVEL
        '''
        return self.get_param('debug_level:min')

    def get_debug_level_increment(self):
        '''
        Set debug levelXI_PRM_DEBUG_LEVEL
        '''
        return self.get_param('debug_level:inc')

    def set_debug_level(self, debug_level):
        '''
        Set debug levelXI_PRM_DEBUG_LEVEL
        '''
        self.set_param('debug_level', debug_level)

    def is_auto_bandwidth_calculation(self):
        '''
        Automatic bandwidth calculation,XI_PRM_AUTO_BANDWIDTH_CALCULATION
        '''
        return self.get_param('auto_bandwidth_calculation')

    def enable_auto_bandwidth_calculation(self):
        '''
        Automatic bandwidth calculation,XI_PRM_AUTO_BANDWIDTH_CALCULATION
        '''
        self.set_param('auto_bandwidth_calculation', True)

    def disable_auto_bandwidth_calculation(self):
        '''
        Automatic bandwidth calculation,XI_PRM_AUTO_BANDWIDTH_CALCULATION
        '''
        self.set_param('auto_bandwidth_calculation', False)

    def is_new_process_chain_enable(self):
        '''
        Enables (2015/FAPI) processing chain for MQ MU camerasXI_PRM_NEW_PROCESS_CHAIN_ENABLE
        '''
        return self.get_param('new_process_chain_enable')

    def enable_new_process_chain_enable(self):
        '''
        Enables (2015/FAPI) processing chain for MQ MU camerasXI_PRM_NEW_PROCESS_CHAIN_ENABLE
        '''
        self.set_param('new_process_chain_enable', True)

    def disable_new_process_chain_enable(self):
        '''
        Enables (2015/FAPI) processing chain for MQ MU camerasXI_PRM_NEW_PROCESS_CHAIN_ENABLE
        '''
        self.set_param('new_process_chain_enable', False)

    def is_cam_enum_golden_enabled(self):
        '''
        Enable enumeration of golden devicesXI_PRM_CAM_ENUM_GOLDEN_ENABLED
        '''
        return self.get_param('cam_enum_golden_enabled')

    def enable_cam_enum_golden_enabled(self):
        '''
        Enable enumeration of golden devicesXI_PRM_CAM_ENUM_GOLDEN_ENABLED
        '''
        self.set_param('cam_enum_golden_enabled', True)

    def disable_cam_enum_golden_enabled(self):
        '''
        Enable enumeration of golden devicesXI_PRM_CAM_ENUM_GOLDEN_ENABLED
        '''
        self.set_param('cam_enum_golden_enabled', False)

    def is_reset_usb_if_bootloader(self):
        '''
        Resets USB device if started as bootloaderXI_PRM_RESET_USB_IF_BOOTLOADER
        '''
        return self.get_param('reset_usb_if_bootloader')

    def enable_reset_usb_if_bootloader(self):
        '''
        Resets USB device if started as bootloaderXI_PRM_RESET_USB_IF_BOOTLOADER
        '''
        self.set_param('reset_usb_if_bootloader', True)

    def disable_reset_usb_if_bootloader(self):
        '''
        Resets USB device if started as bootloaderXI_PRM_RESET_USB_IF_BOOTLOADER
        '''
        self.set_param('reset_usb_if_bootloader', False)

    def get_cam_simulators_count(self):
        '''
        Number of camera simulators to be available.XI_PRM_CAM_SIMULATORS_COUNT
        '''
        return self.get_param('cam_simulators_count')

    def get_cam_simulators_count_maximum(self):
        '''
        Number of camera simulators to be available.XI_PRM_CAM_SIMULATORS_COUNT
        '''
        return self.get_param('cam_simulators_count:max')

    def get_cam_simulators_count_minimum(self):
        '''
        Number of camera simulators to be available.XI_PRM_CAM_SIMULATORS_COUNT
        '''
        return self.get_param('cam_simulators_count:min')

    def get_cam_simulators_count_increment(self):
        '''
        Number of camera simulators to be available.XI_PRM_CAM_SIMULATORS_COUNT
        '''
        return self.get_param('cam_simulators_count:inc')

    def set_cam_simulators_count(self, cam_simulators_count):
        '''
        Number of camera simulators to be available.XI_PRM_CAM_SIMULATORS_COUNT
        '''
        self.set_param('cam_simulators_count', cam_simulators_count)

    def is_cam_sensor_init_disabled(self):
        '''
        Camera sensor will not be initialized when 1=XI_ON is set.XI_PRM_CAM_SENSOR_INIT_DISABLED
        '''
        return self.get_param('cam_sensor_init_disabled')

    def enable_cam_sensor_init_disabled(self):
        '''
        Camera sensor will not be initialized when 1=XI_ON is set.XI_PRM_CAM_SENSOR_INIT_DISABLED
        '''
        self.set_param('cam_sensor_init_disabled', True)

    def disable_cam_sensor_init_disabled(self):
        '''
        Camera sensor will not be initialized when 1=XI_ON is set.XI_PRM_CAM_SENSOR_INIT_DISABLED
        '''
        self.set_param('cam_sensor_init_disabled', False)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Camera FFS
#-------------------------------------------------------------------------------------------------------------------

    def get_read_file_ffs(self,buffer_size=256):
        '''
        Read file from camera flash filesystem.XI_PRM_READ_FILE_FFS
        '''
        return self.get_param('read_file_ffs',buffer_size)

    def get_write_file_ffs(self,buffer_size=256):
        '''
        Write file to camera flash filesystem.XI_PRM_WRITE_FILE_FFS
        '''
        return self.get_param('write_file_ffs',buffer_size)

    def set_write_file_ffs(self, write_file_ffs):
        '''
        Write file to camera flash filesystem.XI_PRM_WRITE_FILE_FFS
        '''
        self.set_param('write_file_ffs', write_file_ffs)

    def get_ffs_file_name(self,buffer_size=256):
        '''
        Set name of file to be written/read from camera FFS.XI_PRM_FFS_FILE_NAME
        '''
        return self.get_param('ffs_file_name',buffer_size)

    def set_ffs_file_name(self, ffs_file_name):
        '''
        Set name of file to be written/read from camera FFS.XI_PRM_FFS_FILE_NAME
        '''
        self.set_param('ffs_file_name', ffs_file_name)

    def get_ffs_file_id(self):
        '''
        File number.XI_PRM_FFS_FILE_ID
        '''
        return self.get_param('ffs_file_id')

    def get_ffs_file_id_maximum(self):
        '''
        File number.XI_PRM_FFS_FILE_ID
        '''
        return self.get_param('ffs_file_id:max')

    def get_ffs_file_id_minimum(self):
        '''
        File number.XI_PRM_FFS_FILE_ID
        '''
        return self.get_param('ffs_file_id:min')

    def get_ffs_file_id_increment(self):
        '''
        File number.XI_PRM_FFS_FILE_ID
        '''
        return self.get_param('ffs_file_id:inc')

    def get_ffs_file_size(self):
        '''
        Size of file.XI_PRM_FFS_FILE_SIZE
        '''
        return self.get_param('ffs_file_size')

    def get_ffs_file_size_maximum(self):
        '''
        Size of file.XI_PRM_FFS_FILE_SIZE
        '''
        return self.get_param('ffs_file_size:max')

    def get_ffs_file_size_minimum(self):
        '''
        Size of file.XI_PRM_FFS_FILE_SIZE
        '''
        return self.get_param('ffs_file_size:min')

    def get_ffs_file_size_increment(self):
        '''
        Size of file.XI_PRM_FFS_FILE_SIZE
        '''
        return self.get_param('ffs_file_size:inc')

    def get_free_ffs_size(self):
        '''
        Size of free camera FFS.XI_PRM_FREE_FFS_SIZE
        '''
        return self.get_param('free_ffs_size')

    def get_free_ffs_size_maximum(self):
        '''
        Size of free camera FFS.XI_PRM_FREE_FFS_SIZE
        '''
        return self.get_param('free_ffs_size:max')

    def get_free_ffs_size_minimum(self):
        '''
        Size of free camera FFS.XI_PRM_FREE_FFS_SIZE
        '''
        return self.get_param('free_ffs_size:min')

    def get_free_ffs_size_increment(self):
        '''
        Size of free camera FFS.XI_PRM_FREE_FFS_SIZE
        '''
        return self.get_param('free_ffs_size:inc')

    def get_used_ffs_size(self):
        '''
        Size of used camera FFS.XI_PRM_USED_FFS_SIZE
        '''
        return self.get_param('used_ffs_size')

    def get_used_ffs_size_maximum(self):
        '''
        Size of used camera FFS.XI_PRM_USED_FFS_SIZE
        '''
        return self.get_param('used_ffs_size:max')

    def get_used_ffs_size_minimum(self):
        '''
        Size of used camera FFS.XI_PRM_USED_FFS_SIZE
        '''
        return self.get_param('used_ffs_size:min')

    def get_used_ffs_size_increment(self):
        '''
        Size of used camera FFS.XI_PRM_USED_FFS_SIZE
        '''
        return self.get_param('used_ffs_size:inc')

    def get_ffs_access_key(self):
        '''
        Setting of key enables file operations on some cameras.XI_PRM_FFS_ACCESS_KEY
        '''
        return self.get_param('ffs_access_key')

    def get_ffs_access_key_maximum(self):
        '''
        Setting of key enables file operations on some cameras.XI_PRM_FFS_ACCESS_KEY
        '''
        return self.get_param('ffs_access_key:max')

    def get_ffs_access_key_minimum(self):
        '''
        Setting of key enables file operations on some cameras.XI_PRM_FFS_ACCESS_KEY
        '''
        return self.get_param('ffs_access_key:min')

    def get_ffs_access_key_increment(self):
        '''
        Setting of key enables file operations on some cameras.XI_PRM_FFS_ACCESS_KEY
        '''
        return self.get_param('ffs_access_key:inc')

    def set_ffs_access_key(self, ffs_access_key):
        '''
        Setting of key enables file operations on some cameras.XI_PRM_FFS_ACCESS_KEY
        '''
        self.set_param('ffs_access_key', ffs_access_key)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: APIContextControl
#-------------------------------------------------------------------------------------------------------------------

    def get_xiapi_context_list(self,buffer_size=256):
        '''
        List of current parameters settings context - parameters with values. Used for offline processing.XI_PRM_API_CONTEXT_LIST
        '''
        return self.get_param('xiapi_context_list',buffer_size)

    def set_xiapi_context_list(self, xiapi_context_list):
        '''
        List of current parameters settings context - parameters with values. Used for offline processing.XI_PRM_API_CONTEXT_LIST
        '''
        self.set_param('xiapi_context_list', xiapi_context_list)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Sensor Control
#-------------------------------------------------------------------------------------------------------------------

    def get_sensor_feature_selector(self):
        '''
        Selects the current feature which is accessible by XI_PRM_SENSOR_FEATURE_VALUE.XI_PRM_SENSOR_FEATURE_SELECTOR
        '''
        return self.get_param('sensor_feature_selector')

    def get_sensor_feature_selector_maximum(self):
        '''
        Selects the current feature which is accessible by XI_PRM_SENSOR_FEATURE_VALUE.XI_PRM_SENSOR_FEATURE_SELECTOR
        '''
        return self.get_param('sensor_feature_selector:max')

    def get_sensor_feature_selector_minimum(self):
        '''
        Selects the current feature which is accessible by XI_PRM_SENSOR_FEATURE_VALUE.XI_PRM_SENSOR_FEATURE_SELECTOR
        '''
        return self.get_param('sensor_feature_selector:min')

    def get_sensor_feature_selector_increment(self):
        '''
        Selects the current feature which is accessible by XI_PRM_SENSOR_FEATURE_VALUE.XI_PRM_SENSOR_FEATURE_SELECTOR
        '''
        return self.get_param('sensor_feature_selector:inc')

    def set_sensor_feature_selector(self, sensor_feature_selector):
        '''
        Selects the current feature which is accessible by XI_PRM_SENSOR_FEATURE_VALUE.XI_PRM_SENSOR_FEATURE_SELECTOR
        '''
        self.set_param('sensor_feature_selector', sensor_feature_selector)

    def get_sensor_feature_value(self):
        '''
        Allows access to sensor feature value currently selected by XI_PRM_SENSOR_FEATURE_SELECTOR.XI_PRM_SENSOR_FEATURE_VALUE
        '''
        return self.get_param('sensor_feature_value')

    def get_sensor_feature_value_maximum(self):
        '''
        Allows access to sensor feature value currently selected by XI_PRM_SENSOR_FEATURE_SELECTOR.XI_PRM_SENSOR_FEATURE_VALUE
        '''
        return self.get_param('sensor_feature_value:max')

    def get_sensor_feature_value_minimum(self):
        '''
        Allows access to sensor feature value currently selected by XI_PRM_SENSOR_FEATURE_SELECTOR.XI_PRM_SENSOR_FEATURE_VALUE
        '''
        return self.get_param('sensor_feature_value:min')

    def get_sensor_feature_value_increment(self):
        '''
        Allows access to sensor feature value currently selected by XI_PRM_SENSOR_FEATURE_SELECTOR.XI_PRM_SENSOR_FEATURE_VALUE
        '''
        return self.get_param('sensor_feature_value:inc')

    def set_sensor_feature_value(self, sensor_feature_value):
        '''
        Allows access to sensor feature value currently selected by XI_PRM_SENSOR_FEATURE_SELECTOR.XI_PRM_SENSOR_FEATURE_VALUE
        '''
        self.set_param('sensor_feature_value', sensor_feature_value)

#-------------------------------------------------------------------------------------------------------------------
# ---- Parameter Group: Extended Features
#-------------------------------------------------------------------------------------------------------------------

    def get_ext_feature_selector(self):
        '''
        Selection of extended feature.XI_PRM_EXTENDED_FEATURE_SELECTOR
        '''
        return self.get_param('ext_feature_selector')

    def get_ext_feature_selector_maximum(self):
        '''
        Selection of extended feature.XI_PRM_EXTENDED_FEATURE_SELECTOR
        '''
        return self.get_param('ext_feature_selector:max')

    def get_ext_feature_selector_minimum(self):
        '''
        Selection of extended feature.XI_PRM_EXTENDED_FEATURE_SELECTOR
        '''
        return self.get_param('ext_feature_selector:min')

    def get_ext_feature_selector_increment(self):
        '''
        Selection of extended feature.XI_PRM_EXTENDED_FEATURE_SELECTOR
        '''
        return self.get_param('ext_feature_selector:inc')

    def set_ext_feature_selector(self, ext_feature_selector):
        '''
        Selection of extended feature.XI_PRM_EXTENDED_FEATURE_SELECTOR
        '''
        self.set_param('ext_feature_selector', ext_feature_selector)

    def get_ext_feature(self):
        '''
        Extended feature value.XI_PRM_EXTENDED_FEATURE
        '''
        return self.get_param('ext_feature')

    def get_ext_feature_maximum(self):
        '''
        Extended feature value.XI_PRM_EXTENDED_FEATURE
        '''
        return self.get_param('ext_feature:max')

    def get_ext_feature_minimum(self):
        '''
        Extended feature value.XI_PRM_EXTENDED_FEATURE
        '''
        return self.get_param('ext_feature:min')

    def get_ext_feature_increment(self):
        '''
        Extended feature value.XI_PRM_EXTENDED_FEATURE
        '''
        return self.get_param('ext_feature:inc')

    def set_ext_feature(self, ext_feature):
        '''
        Extended feature value.XI_PRM_EXTENDED_FEATURE
        '''
        self.set_param('ext_feature', ext_feature)

    def get_device_unit_selector(self):
        '''
        Selects device unit.XI_PRM_DEVICE_UNIT_SELECTOR
        '''
        return self.get_param('device_unit_selector')

    def get_device_unit_selector_maximum(self):
        '''
        Selects device unit.XI_PRM_DEVICE_UNIT_SELECTOR
        '''
        return self.get_param('device_unit_selector:max')

    def get_device_unit_selector_minimum(self):
        '''
        Selects device unit.XI_PRM_DEVICE_UNIT_SELECTOR
        '''
        return self.get_param('device_unit_selector:min')

    def get_device_unit_selector_increment(self):
        '''
        Selects device unit.XI_PRM_DEVICE_UNIT_SELECTOR
        '''
        return self.get_param('device_unit_selector:inc')

    def set_device_unit_selector(self, device_unit_selector):
        '''
        Selects device unit.XI_PRM_DEVICE_UNIT_SELECTOR
        '''
        self.set_param('device_unit_selector', device_unit_selector)

    def get_device_unit_register_selector(self):
        '''
        Selects register of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).XI_PRM_DEVICE_UNIT_REGISTER_SELECTOR
        '''
        return self.get_param('device_unit_register_selector')

    def get_device_unit_register_selector_maximum(self):
        '''
        Selects register of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).XI_PRM_DEVICE_UNIT_REGISTER_SELECTOR
        '''
        return self.get_param('device_unit_register_selector:max')

    def get_device_unit_register_selector_minimum(self):
        '''
        Selects register of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).XI_PRM_DEVICE_UNIT_REGISTER_SELECTOR
        '''
        return self.get_param('device_unit_register_selector:min')

    def get_device_unit_register_selector_increment(self):
        '''
        Selects register of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).XI_PRM_DEVICE_UNIT_REGISTER_SELECTOR
        '''
        return self.get_param('device_unit_register_selector:inc')

    def set_device_unit_register_selector(self, device_unit_register_selector):
        '''
        Selects register of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).XI_PRM_DEVICE_UNIT_REGISTER_SELECTOR
        '''
        self.set_param('device_unit_register_selector', device_unit_register_selector)

    def get_device_unit_register_value(self):
        '''
        Sets/gets register value of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).XI_PRM_DEVICE_UNIT_REGISTER_VALUE
        '''
        return self.get_param('device_unit_register_value')

    def get_device_unit_register_value_maximum(self):
        '''
        Sets/gets register value of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).XI_PRM_DEVICE_UNIT_REGISTER_VALUE
        '''
        return self.get_param('device_unit_register_value:max')

    def get_device_unit_register_value_minimum(self):
        '''
        Sets/gets register value of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).XI_PRM_DEVICE_UNIT_REGISTER_VALUE
        '''
        return self.get_param('device_unit_register_value:min')

    def get_device_unit_register_value_increment(self):
        '''
        Sets/gets register value of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).XI_PRM_DEVICE_UNIT_REGISTER_VALUE
        '''
        return self.get_param('device_unit_register_value:inc')

    def set_device_unit_register_value(self, device_unit_register_value):
        '''
        Sets/gets register value of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).XI_PRM_DEVICE_UNIT_REGISTER_VALUE
        '''
        self.set_param('device_unit_register_value', device_unit_register_value)

    def get_api_progress_callback(self,buffer_size=256):
        '''
        Callback address of pointer that is called upon long tasks (e.g. XI_PRM_WRITE_FILE_FFS).XI_PRM_API_PROGRESS_CALLBACK
        '''
        return self.get_param('api_progress_callback',buffer_size)

    def set_api_progress_callback(self, api_progress_callback):
        '''
        Callback address of pointer that is called upon long tasks (e.g. XI_PRM_WRITE_FILE_FFS).XI_PRM_API_PROGRESS_CALLBACK
        '''
        self.set_param('api_progress_callback', api_progress_callback)

    def get_acquisition_status_selector(self):
        '''
        Selects the internal acquisition signal to read using XI_PRM_ACQUISITION_STATUS.XI_PRM_ACQUISITION_STATUS_SELECTOR
        '''
        return self.get_param('acquisition_status_selector')

    def get_acquisition_status_selector_maximum(self):
        '''
        Selects the internal acquisition signal to read using XI_PRM_ACQUISITION_STATUS.XI_PRM_ACQUISITION_STATUS_SELECTOR
        '''
        return self.get_param('acquisition_status_selector:max')

    def get_acquisition_status_selector_minimum(self):
        '''
        Selects the internal acquisition signal to read using XI_PRM_ACQUISITION_STATUS.XI_PRM_ACQUISITION_STATUS_SELECTOR
        '''
        return self.get_param('acquisition_status_selector:min')

    def get_acquisition_status_selector_increment(self):
        '''
        Selects the internal acquisition signal to read using XI_PRM_ACQUISITION_STATUS.XI_PRM_ACQUISITION_STATUS_SELECTOR
        '''
        return self.get_param('acquisition_status_selector:inc')

    def set_acquisition_status_selector(self, acquisition_status_selector):
        '''
        Selects the internal acquisition signal to read using XI_PRM_ACQUISITION_STATUS.XI_PRM_ACQUISITION_STATUS_SELECTOR
        '''
        self.set_param('acquisition_status_selector', acquisition_status_selector)

    def get_acquisition_status(self):
        '''
        Acquisition status(True/False)XI_PRM_ACQUISITION_STATUS
        '''
        return self.get_param('acquisition_status')

    def get_acquisition_status_maximum(self):
        '''
        Acquisition status(True/False)XI_PRM_ACQUISITION_STATUS
        '''
        return self.get_param('acquisition_status:max')

    def get_acquisition_status_minimum(self):
        '''
        Acquisition status(True/False)XI_PRM_ACQUISITION_STATUS
        '''
        return self.get_param('acquisition_status:min')

    def get_acquisition_status_increment(self):
        '''
        Acquisition status(True/False)XI_PRM_ACQUISITION_STATUS
        '''
        return self.get_param('acquisition_status:inc')

