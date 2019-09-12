from ctypes import *
from ctypes.wintypes import *

ERROR_CODES = {    
     0: "Function call succeeded",
     1: "Invalid handle",
     2: "Register read error",
     3: "Register write error",
     4: "Freeing resources error",
     5: "Freeing channel error",
     6: "Freeing bandwith error",
     7: "Read block error",
     8: "Write block error",
     9: "No image",
    10: "Timeout",
    11: "Invalid arguments supplied",
    12: "Not supported",
    13: "Attach buffers error",
    14: "Overlapped result",
    15: "Memory allocation error",
    16: "DLL context is NULL",
    17: "DLL context is non zero",
    18: "DLL context exists",
    19: "Too many devices connected",
    20: "Camera context error",
    21: "Unknown hardware",
    22: "Invalid TM file",
    23: "Invalid TM tag",
    24: "Incomplete TM",
    25: "Bus reset error",
    26: "Not implemented",
    27: "Shading is too bright",
    28: "Shading is too dark",
    29: "Gain is too low",
    30: "Invalid sensor defect correction list",
    31: "Error while sensor defect correction list reallocation",
    32: "Invalid pixel list",
    33: "Invalid Flash File System",
    34: "Invalid profile",
    35: "Invalid calibration",
    36: "Invalid buffer",
    38: "Invalid data",
    39: "Timing generator is busy",
    40: "Wrong operation open/write/read/close",
    41: "Acquisition already started",
    42: "Old version of device driver installed to the system.",
    43: "To get error code please call GetLastError function.",
    44: "Data cannot be processed",
    45: "Acquisition is stopped. It needs to be started to perform operation.",
    46: "Acquisition has been stopped with an error.",
    47: "Input ICC profile missing or corrupted",
    48: "Output ICC profile missing or corrupted",
    49: "Device not ready to operate",
    50: "Shading is too contrast",
    51: "Module already initialized",
    52: "Application does not have enough privileges (one or more app)",
    53: "Installed driver is not compatible with current software",
    54: "TM file was not loaded successfully from resources",
    55: "Device has been reset, abnormal initial state",
    56: "No Devices Found",
    57: "Resource (device) or function locked by mutex",
    58: "Buffer provided by user is too small",
    59: "Couldnt initialize processor.",
    60: "The object/module/procedure/process being referred to has not been started.",
    61: "Resource not found(could be processor, file, item...).",
    0: "Function call succeeded",
    100: "Unknown parameter",
    101: "Wrong parameter value",
    103: "Wrong parameter type",
    104: "Wrong parameter size",
    105: "Input buffer is too small",
    106: "Parameter is not supported",
    107: "Parameter info not supported",
    108: "Data format is not supported",
    109: "Read only parameter",
    111: "This camera does not support currently available bandwidth",
    112: "FFS file selector is invalid or NULL",
    113: "FFS file not found",
    114: "Parameter value cannot be set (might be out of range or invalid).",
    115: "Safe buffer policy is not supported. E.g. when transport target is set to GPU (GPUDirect).",
    116: "GPUDirect is not available. E.g. platform isn't supported or CUDA toolkit isn't installed.",
    201: "Processing error - other",
    202: "Error while image processing.",
    203: "Input format is not supported for processing.",
    204: "Output format is not supported for processing.",
    205: "Parameter value is out of range",
    }
# Enumerators

#Downsampling value enumerator.
XI_DOWNSAMPLING_VALUE = { 
    "XI_DWN_1x1": c_uint(1),    #Downsampling 1x1.
    "XI_DWN_2x2": c_uint(2),    #Downsampling 2x2.
    "XI_DWN_3x3": c_uint(3),    #Downsampling 3x3.
    "XI_DWN_4x4": c_uint(4),    #Downsampling 4x4.
    "XI_DWN_5x5": c_uint(5),    #Downsampling 5x5.
    "XI_DWN_6x6": c_uint(6),    #Downsampling 6x6.
    "XI_DWN_7x7": c_uint(7),    #Downsampling 7x7.
    "XI_DWN_8x8": c_uint(8),    #Downsampling 8x8.
    "XI_DWN_9x9": c_uint(9),    #Downsampling 9x9.
    "XI_DWN_10x10": c_uint(10),    #Downsampling 10x10.
    "XI_DWN_16x16": c_uint(16),    #Downsampling 16x16.
    }

#Test Pattern Generator
XI_TEST_PATTERN_GENERATOR = { 
    "XI_TESTPAT_GEN_SENSOR": c_uint(0),    # Sensor Test Pattern Generator
    "XI_TESTPAT_GEN_FPGA": c_uint(1),    # FPGA Test Pattern Generator
    }

#Test Pattern Type
XI_TEST_PATTERN = { 
    "XI_TESTPAT_OFF": c_uint(0),    # Testpattern turned off.
    "XI_TESTPAT_BLACK": c_uint(1),    # Image is filled with darkest possible image.
    "XI_TESTPAT_WHITE": c_uint(2),    # Image is filled with brightest possible image.
    "XI_TESTPAT_GREY_HORIZ_RAMP": c_uint(3),    # Image is filled horizontally with an image that goes from the darkest possible value to the brightest.
    "XI_TESTPAT_GREY_VERT_RAMP": c_uint(4),    # Image is filled vertically with an image that goes from the darkest possible value to the brightest.
    "XI_TESTPAT_GREY_HORIZ_RAMP_MOVING": c_uint(5),    # Image is filled horizontally with an image that goes from the darkest possible value to the brightest and moves from left to right.
    "XI_TESTPAT_GREY_VERT_RAMP_MOVING": c_uint(6),    # Image is filled vertically with an image that goes from the darkest possible value to the brightest and moves from left to right.
    "XI_TESTPAT_HORIZ_LINE_MOVING": c_uint(7),    # A moving horizontal line is superimposed on the live image.
    "XI_TESTPAT_VERT_LINE_MOVING": c_uint(8),    # A moving vertical line is superimposed on the live image.
    "XI_TESTPAT_COLOR_BAR": c_uint(9),    # Image is filled with stripes of color including White, Black, Red, Green, Blue, Cyan, Magenta and Yellow.
    "XI_TESTPAT_FRAME_COUNTER": c_uint(10),    # A frame counter is superimposed on the live image.
    "XI_TESTPAT_DEVICE_SPEC_COUNTER": c_uint(11),    # 128bit counter.
    }

#Decimation Pattern Format
XI_DEC_PATTERN = { 
    "XI_DEC_MONO": c_uint(1),    # Monochrome Decimation Format
    "XI_DEC_BAYER": c_uint(2),    # Bayer Decimation Format
    }

#Binning Pattern Format
XI_BIN_PATTERN = { 
    "XI_BIN_MONO": c_uint(1),    # Monochrome Binning Format
    "XI_BIN_BAYER": c_uint(2),    # Bayer Binning Format
    }

#Binning Engine Selector
XI_BIN_SELECTOR = { 
    "XI_BIN_SELECT_SENSOR": c_uint(0),    #Sensor Binning Engine
    "XI_BIN_SELECT_DEVICE_FPGA": c_uint(1),    #FPGA Binning Engine on device side
    "XI_BIN_SELECT_HOST_CPU": c_uint(2),    # CPU Binning Engine on Host side
    }

#Binning Mode
XI_BIN_MODE = { 
    "XI_BIN_MODE_SUM": c_uint(0),    #The response from the combined pixels will be added, resulting in increased sensitivity.
    "XI_BIN_MODE_AVERAGE": c_uint(1),    #The response from the combined pixels will be averaged, resulting in increased signal/noise ratio.
    }

#Decimation Engine Selector
XI_DEC_SELECTOR = { 
    "XI_DEC_SELECT_SENSOR": c_uint(0),    # Sensor Decimation Engine
    "XI_DEC_SELECT_DEVICE_FPGA": c_uint(1),    #FPGA Decimation Engine on device side
    "XI_DEC_SELECT_HOST_CPU": c_uint(2),    # CPU Decimation Engine on Host side
    }

#Sensor tap count enumerator.
XI_SENSOR_TAP_CNT = { 
    "XI_TAP_CNT_1": c_uint(1),    #1 sensor tap selected.
    "XI_TAP_CNT_2": c_uint(2),    #2 sensor taps selected.
    "XI_TAP_CNT_4": c_uint(4),    #4 sensor taps selected.
    }

#Bit depth enumerator.
XI_BIT_DEPTH = { 
    "XI_BPP_8": c_uint(8),    #8 bit per pixel
    "XI_BPP_9": c_uint(9),    #9 bit per pixel
    "XI_BPP_10": c_uint(10),    #10 bit per pixel
    "XI_BPP_11": c_uint(11),    #11 bit per pixel
    "XI_BPP_12": c_uint(12),    #12 bit per pixel
    "XI_BPP_14": c_uint(14),    #14 bit per pixel
    "XI_BPP_16": c_uint(16),    #16 bit per pixel
    }

#Debug level enumerator.
XI_DEBUG_LEVEL = { 
    "XI_DL_DETAIL": c_uint(0),    #Same as trace plus locking resources
    "XI_DL_TRACE": c_uint(1),    #Information level.
    "XI_DL_WARNING": c_uint(2),    #Warning level.
    "XI_DL_ERROR": c_uint(3),    #Error level.
    "XI_DL_FATAL": c_uint(4),    #Fatal error level.
    "XI_DL_DISABLED": c_uint(100),    #Print no errors at all.
    }

#Image output format enumerator.
XI_IMG_FORMAT = { 
    "XI_MONO8": c_uint(0),    #8 bits per pixel
    "XI_MONO16": c_uint(1),    #16 bits per pixel
    "XI_RGB24": c_uint(2),    #RGB data format
    "XI_RGB32": c_uint(3),    #RGBA data format
    "XI_RGB_PLANAR": c_uint(4),    #RGB planar data format
    "XI_RAW8": c_uint(5),    #8 bits per pixel raw data from sensor
    "XI_RAW16": c_uint(6),    #16 bits per pixel raw data from sensor
    "XI_FRM_TRANSPORT_DATA": c_uint(7),    #Data from transport layer (e.g. packed). Format see XI_PRM_TRANSPORT_PIXEL_FORMAT
    "XI_RGB48": c_uint(8),    #RGB data format
    "XI_RGB64": c_uint(9),    #RGBA data format
    "XI_RGB16_PLANAR": c_uint(10),    #RGB16 planar data format
    "XI_RAW8X2": c_uint(11),    #8 bits per pixel raw data from sensor(2 components in a row)
    "XI_RAW8X4": c_uint(12),    #8 bits per pixel raw data from sensor(4 components in a row)
    "XI_RAW16X2": c_uint(13),    #16 bits per pixel raw data from sensor(2 components in a row)
    "XI_RAW16X4": c_uint(14),    #16 bits per pixel raw data from sensor(4 components in a row)
    }

#Bayer color matrix enumerator.
XI_COLOR_FILTER_ARRAY = { 
    "XI_CFA_NONE": c_uint(0),    # B/W sensors
    "XI_CFA_BAYER_RGGB": c_uint(1),    #Regular RGGB
    "XI_CFA_CMYG": c_uint(2),    #AK Sony sens
    "XI_CFA_RGR": c_uint(3),    #2R+G readout
    "XI_CFA_BAYER_BGGR": c_uint(4),    #BGGR readout
    "XI_CFA_BAYER_GRBG": c_uint(5),    #GRBG readout
    "XI_CFA_BAYER_GBRG": c_uint(6),    #GBRG readout
    }

#structure containing information about buffer policy(can be safe, data will be copied to user/app buffer or unsafe, user will get internally allocated buffer without data copy).
XI_BP = { 
    "XI_BP_UNSAFE": c_uint(0),    #User gets pointer to internally allocated circle buffer and data may be overwritten by device.
    "XI_BP_SAFE": c_uint(1),    #Data from device will be copied to user allocated buffer or xiApi allocated memory.
    }

#structure containing information about trigger source
XI_TRG_SOURCE = { 
    "XI_TRG_OFF": c_uint(0),    #Camera works in free run mode.
    "XI_TRG_EDGE_RISING": c_uint(1),    #External trigger (rising edge).
    "XI_TRG_EDGE_FALLING": c_uint(2),    #External trigger (falling edge).
    "XI_TRG_SOFTWARE": c_uint(3),    #Software(manual) trigger.
    "XI_TRG_LEVEL_HIGH": c_uint(4),    #Specifies that the trigger is considered valid as long as the level of the source signal is high.
    "XI_TRG_LEVEL_LOW": c_uint(5),    #Specifies that the trigger is considered valid as long as the level of the source signal is low.
    }

#structure containing information about trigger functionality
XI_TRG_SELECTOR = { 
    "XI_TRG_SEL_FRAME_START": c_uint(0),    #Selects a trigger starting the capture of one frame
    "XI_TRG_SEL_EXPOSURE_ACTIVE": c_uint(1),    #Selects a trigger controlling the duration of one frame.
    "XI_TRG_SEL_FRAME_BURST_START": c_uint(2),    #Selects a trigger starting the capture of the bursts of frames in an acquisition.
    "XI_TRG_SEL_FRAME_BURST_ACTIVE": c_uint(3),    #Selects a trigger controlling the duration of the capture of the bursts of frames in an acquisition.
    "XI_TRG_SEL_MULTIPLE_EXPOSURES": c_uint(4),    #Selects a trigger which when first trigger starts exposure and consequent pulses are gating exposure(active HI)
    "XI_TRG_SEL_EXPOSURE_START": c_uint(5),    #Selects a trigger controlling the start of the exposure of one Frame.
    "XI_TRG_SEL_MULTI_SLOPE_PHASE_CHANGE": c_uint(6),    #Selects a trigger controlling the multi slope phase in one Frame (phase0 -> phase1) or (phase1 -> phase2).
    "XI_TRG_SEL_ACQUISITION_START": c_uint(7),    #Selects a trigger starting acquisition of first frame.
    }

#Trigger overlap modes
XI_TRG_OVERLAP = { 
    "XI_TRG_OVERLAP_OFF": c_uint(0),    #No trigger overlap is permitted. If camera is in read-out phase, all triggers are rejected.
    "XI_TRG_OVERLAP_READ_OUT": c_uint(1),    #Trigger is accepted only when sensor is ready to start next exposure with defined exposure time. Trigger is rejected when sensor is not ready for new exposure with defined exposure time.
    "XI_TRG_OVERLAP_PREV_FRAME": c_uint(2),    #Trigger is accepted by camera any time. If sensor is not ready for the next exposure - the trigger is latched and sensor starts exposure as soon as exposure can be started with defined exposure time.
    }

#structure containing information about acqisition timing modes
XI_ACQ_TIMING_MODE = { 
    "XI_ACQ_TIMING_MODE_FREE_RUN": c_uint(0),    #Selects a mode when sensor timing is given by fastest framerate possible (by exposure time and readout).
    "XI_ACQ_TIMING_MODE_FRAME_RATE": c_uint(1),    #Selects a mode when sensor frame acquisition frequency is set to XI_PRM_FRAMERATE
    "XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT": c_uint(2),    #Selects a mode when sensor frame acquisition frequency is limited by XI_PRM_FRAMERATE
    }

#Enumerator for data target modes
XI_TRANSPORT_DATA_TARGET_MODE = { 
    "XI_TRANSPORT_DATA_TARGET_CPU_RAM": c_uint(0),    #Selects a CPU RAM as target for delivered data from the camera.
    "XI_TRANSPORT_DATA_TARGET_GPU_RAM": c_uint(1),    #Selects a GPU RAM as target for delivered data from the camera.
    }

#Enumeration for XI_PRM_GPI_SELECTOR for CB cameras.
XI_GPI_SEL_CB = { 
    "XI_GPI_SEL_CB_IN1": c_uint(1),    #Input1 - Pin3 (Opto Isolated).
    "XI_GPI_SEL_CB_IN2": c_uint(2),    #Input2 - Pin4 (Opto Isolated).
    "XI_GPI_SEL_CB_INOUT1": c_uint(3),    #Input/Output1 - Pin6
    "XI_GPI_SEL_CB_INOUT2": c_uint(4),    #Input/Output2 - Pin7
    "XI_GPI_SEL_CB_INOUT3": c_uint(5),    #Input/Output3 - Pin11
    "XI_GPI_SEL_CB_INOUT4": c_uint(6),    #Input/Output4 - Pin12
    }

#Enumeration for XI_PRM_GPO_SELECTOR for CB cameras.
XI_GPO_SEL_CB = { 
    "XI_GPO_SEL_CB_OUT1": c_uint(1),    #Output1 - Pin8 (Opto Isolated).
    "XI_GPO_SEL_CB_OUT2": c_uint(2),    #Output2 - Pin9 (Opto Isolated).
    "XI_GPO_SEL_CB_INOUT1": c_uint(3),    #Input/Output1 - Pin6
    "XI_GPO_SEL_CB_INOUT2": c_uint(4),    #Input/Output2 - Pin7
    "XI_GPO_SEL_CB_INOUT3": c_uint(5),    #Input/Output3 - Pin11
    "XI_GPO_SEL_CB_INOUT4": c_uint(6),    #Input/Output4 - Pin12
    }

#structure containing information about GPI functionality
XI_GPI_MODE = { 
    "XI_GPI_OFF": c_uint(0),    #Input off. In this mode the input level can be get using parameter XI_PRM_GPI_LEVEL.
    "XI_GPI_TRIGGER": c_uint(1),    #Trigger input
    "XI_GPI_EXT_EVENT": c_uint(2),    #External signal input. It is not implemented yet.
    }

#Enumerator for GPI port selection.
XI_GPI_SELECTOR = { 
    "XI_GPI_PORT1": c_uint(1),    #GPI port 1
    "XI_GPI_PORT2": c_uint(2),    #GPI port 2
    "XI_GPI_PORT3": c_uint(3),    #GPI port 3
    "XI_GPI_PORT4": c_uint(4),    #GPI port 4
    "XI_GPI_PORT5": c_uint(5),    #GPI port 5
    "XI_GPI_PORT6": c_uint(6),    #GPI port 6
    }

#structure containing information about GPO functionality
XI_GPO_MODE = { 
    "XI_GPO_OFF": c_uint(0),    #Output off
    "XI_GPO_ON": c_uint(1),    #Logical level.
    "XI_GPO_FRAME_ACTIVE": c_uint(2),    #On from exposure started until read out finished.
    "XI_GPO_FRAME_ACTIVE_NEG": c_uint(3),    #Off from exposure started until read out finished.
    "XI_GPO_EXPOSURE_ACTIVE": c_uint(4),    #On during exposure(integration) time
    "XI_GPO_EXPOSURE_ACTIVE_NEG": c_uint(5),    #Off during exposure(integration) time
    "XI_GPO_FRAME_TRIGGER_WAIT": c_uint(6),    #On when sensor is ready for next trigger edge.
    "XI_GPO_FRAME_TRIGGER_WAIT_NEG": c_uint(7),    #Off when sensor is ready for next trigger edge.
    "XI_GPO_EXPOSURE_PULSE": c_uint(8),    #Short On/Off pulse on start of each exposure.
    "XI_GPO_EXPOSURE_PULSE_NEG": c_uint(9),    #Short Off/On pulse on start of each exposure.
    "XI_GPO_BUSY": c_uint(10),    #ON when camera is busy (trigger mode - starts with trigger reception and ends with end of frame transfer from sensor; freerun - active when acq active)
    "XI_GPO_BUSY_NEG": c_uint(11),    #OFF when camera is busy (trigger mode  - starts with trigger reception and ends with end of frame transfer from sensor; freerun - active when acq active)
    "XI_GPO_HIGH_IMPEDANCE": c_uint(12),    #Hi impedance of output (if three state logic is used).
    "XI_GPO_FRAME_BUFFER_OVERFLOW": c_uint(13),    #Frame buffer overflow status.
    }

#Enumerator for GPO port selection.
XI_GPO_SELECTOR = { 
    "XI_GPO_PORT1": c_uint(1),    #GPO port 1
    "XI_GPO_PORT2": c_uint(2),    #GPO port 2
    "XI_GPO_PORT3": c_uint(3),    #GPO port 3
    "XI_GPO_PORT4": c_uint(4),    #GPO port 4
    "XI_GPO_PORT5": c_uint(5),    #GPO port 5
    "XI_GPO_PORT6": c_uint(6),    #GPO port 6
    }

#structure containing information about LED functionality
XI_LED_MODE = { 
    "XI_LED_HEARTBEAT": c_uint(0),    #Blinking (1Hz) if all is OK (CURRERA-R only).
    "XI_LED_TRIGGER_ACTIVE": c_uint(1),    #On if trigger detected (CURRERA-R only).
    "XI_LED_EXT_EVENT_ACTIVE": c_uint(2),    #On if external signal detected (CURRERA-R only)
    "XI_LED_LINK": c_uint(3),    #On if link is OK (Currera-R only)
    "XI_LED_ACQUISITION": c_uint(4),    #On if data streaming is on
    "XI_LED_EXPOSURE_ACTIVE": c_uint(5),    #On if sensor is integrating
    "XI_LED_FRAME_ACTIVE": c_uint(6),    #On if frame is active (exposure or readout)
    "XI_LED_OFF": c_uint(7),    #Off
    "XI_LED_ON": c_uint(8),    #On
    "XI_LED_BLINK": c_uint(9),    #Blinking (1Hz)
    }

#Enumerator for LED selection.
XI_LED_SELECTOR = { 
    "XI_LED_SEL1": c_uint(1),    #LED 1
    "XI_LED_SEL2": c_uint(2),    #LED 2
    "XI_LED_SEL3": c_uint(3),    #LED 3
    "XI_LED_SEL4": c_uint(4),    #LED 4
    }

#structure contains frames counter
XI_COUNTER_SELECTOR = { 
    "XI_CNT_SEL_TRANSPORT_SKIPPED_FRAMES": c_uint(0),    #Number of skipped frames on transport layer (e.g. when image gets lost while transmission). Occur when capacity of transport channel does not allow to transfer all data.
    "XI_CNT_SEL_API_SKIPPED_FRAMES": c_uint(1),    #Number of skipped frames on API layer. Occur when application does not process the images as quick as they are received from the camera.
    "XI_CNT_SEL_TRANSPORT_TRANSFERRED_FRAMES": c_uint(2),    #Number of successfully transferred frames on transport layer.
    "XI_CNT_SEL_FRAME_MISSED_TRIGGER_DUETO_OVERLAP": c_uint(3),    #Number of missed triggers due to overlap.
    "XI_CNT_SEL_FRAME_MISSED_TRIGGER_DUETO_FRAME_BUFFER_OVR": c_uint(4),    #Number of missed triggers due to frame buffer full.
    "XI_CNT_SEL_FRAME_BUFFER_OVERFLOW": c_uint(5),    #Frame buffer full counter.
    }

#structure containing information about time stamp reset arming
XI_TS_RST_MODE = { 
    "XI_TS_RST_ARM_ONCE": c_uint(0),    #TimeStamp reset is armed once, after execution engine is disabled
    "XI_TS_RST_ARM_PERSIST": c_uint(1),    #TimeStamp reset is armed permanently if source is selected 
    }

#structure containing information about possible timestamp reset sources
XI_TS_RST_SOURCE = { 
    "XI_TS_RST_OFF": c_uint(0),    #No source selected, timestamp reset effectively disabled
    "XI_TS_RST_SRC_GPI_1": c_uint(1),    #TimeStamp reset source selected GPI1 (after de bounce)
    "XI_TS_RST_SRC_GPI_2": c_uint(2),    #TimeStamp reset source selected GPI2 (after de bounce)
    "XI_TS_RST_SRC_GPI_3": c_uint(3),    #TimeStamp reset source selected GPI3 (after de bounce)
    "XI_TS_RST_SRC_GPI_4": c_uint(4),    #TimeStamp reset source selected GPI4 (after de bounce)
    "XI_TS_RST_SRC_GPI_1_INV": c_uint(5),    #TimeStamp reset source selected GPI1 inverted (after de bounce)
    "XI_TS_RST_SRC_GPI_2_INV": c_uint(6),    #TimeStamp reset source selected GPI2 inverted (after de bounce)
    "XI_TS_RST_SRC_GPI_3_INV": c_uint(7),    #TimeStamp reset source selected GPI3 inverted (after de bounce)
    "XI_TS_RST_SRC_GPI_4_INV": c_uint(8),    #TimeStamp reset source selected GPI4 inverted (after de bounce)
    "XI_TS_RST_SRC_GPO_1": c_uint(9),    #TimeStamp reset source selected GPO1 (after de bounce)
    "XI_TS_RST_SRC_GPO_2": c_uint(10),    #TimeStamp reset source selected GPO2 (after de bounce)
    "XI_TS_RST_SRC_GPO_3": c_uint(11),    #TimeStamp reset source selected GPO3 (after de bounce)
    "XI_TS_RST_SRC_GPO_4": c_uint(12),    #TimeStamp reset source selected GPO4 (after de bounce)
    "XI_TS_RST_SRC_GPO_1_INV": c_uint(13),    #TimeStamp reset source selected GPO1 inverted (after de bounce)
    "XI_TS_RST_SRC_GPO_2_INV": c_uint(14),    #TimeStamp reset source selected GPO2 inverted (after de bounce)
    "XI_TS_RST_SRC_GPO_3_INV": c_uint(15),    #TimeStamp reset source selected GPO3 inverted (after de bounce)
    "XI_TS_RST_SRC_GPO_4_INV": c_uint(16),    #TimeStamp reset source selected GPO4 inverted (after de bounce)
    "XI_TS_RST_SRC_TRIGGER": c_uint(17),    #TimeStamp reset source selected TRIGGER (signal for sensor)
    "XI_TS_RST_SRC_TRIGGER_INV": c_uint(18),    #TimeStamp reset source selected TRIGGER (signal for sensor)
    "XI_TS_RST_SRC_SW": c_uint(19),    #TimeStamp reset source selected software (has immediate effect and is self cleared)
    "XI_TS_RST_SRC_EXPACTIVE": c_uint(20),    #TimeStamp reset source selected exposure active 
    "XI_TS_RST_SRC_EXPACTIVE_INV": c_uint(21),    #TimeStamp reset source selected exposure active 
    "XI_TS_RST_SRC_FVAL": c_uint(22),    #TimeStamp reset source selected frame valid signal from sensor
    "XI_TS_RST_SRC_FVAL_INV": c_uint(23),    #TimeStamp reset source selected frame valid inverted signal from sensor
    }

#structure containing information about parameters type
XI_PRM_TYPE = { 
    "xiTypeInteger": c_uint(0),    #integer parameter type
    "xiTypeFloat": c_uint(1),    #float parameter type
    "xiTypeString": c_uint(2),    #string parameter type
    "xiTypeEnum": c_uint(0),    #enumerator parameter type
    "xiTypeBoolean": c_uint(0),    #boolean parameter type
    "xiTypeCommand": c_uint(0),    #command parameter type
    }

#Turn parameter On/Off
XI_SWITCH = { 
    "XI_OFF": c_uint(0),    #Turn parameter off
    "XI_ON": c_uint(1),    #Turn parameter on
    }

#Temperature selector
XI_TEMP_SELECTOR = { 
    "XI_TEMP_IMAGE_SENSOR_DIE_RAW": c_uint(0),    #Not calibrated temperature of image sensor die (silicon) - e.g. sensor register value
    "XI_TEMP_IMAGE_SENSOR_DIE": c_uint(1),    #Calibrated temperature of image sensor die (silicon) - in degrees of Celsius
    "XI_TEMP_SENSOR_BOARD": c_uint(2),    #Sensor board temperature
    "XI_TEMP_INTERFACE_BOARD": c_uint(3),    #Interface board temperature
    "XI_TEMP_FRONT_HOUSING": c_uint(4),    #Front part of camera housing temperature
    "XI_TEMP_REAR_HOUSING": c_uint(5),    #Rear part of camera housing temperature
    "XI_TEMP_TEC1_COLD": c_uint(6),    #TEC1 cold side temperature
    "XI_TEMP_TEC1_HOT": c_uint(7),    #TEC1 hot side temperature
    }

#Temperature selector
XI_TEMP_CTRL_MODE_SELECTOR = { 
    "XI_TEMP_CTRL_MODE_OFF": c_uint(0),    #Temperature controlling is disabled (no fan or TEC (peltier) is enabled)
    "XI_TEMP_CTRL_MODE_AUTO": c_uint(1),    #Automated temperature controlling is enabled - based on selected thermomether and target temperature.
    "XI_TEMP_CTRL_MODE_MANUAL": c_uint(2),    #Manual controlling of temperature elements is enabled. Application can control the elements.
    }

#Temperature element selector
XI_TEMP_ELEMENT_SELECTOR = { 
    "XI_TEMP_ELEM_TEC1": c_uint(11),    #Temperature element TEC1 (peltier closest to sensor)
    "XI_TEMP_ELEM_TEC2": c_uint(12),    #Temperature element TEC2 (peltier)
    "XI_TEMP_ELEM_FAN1": c_uint(31),    #Temperature element fan current or rotation
    "XI_TEMP_ELEM_FAN1_THRS_TEMP": c_uint(32),    #Temperature element fan start rotation threshold temperature
    }

#Data packing(grouping) types.
XI_OUTPUT_DATA_PACKING_TYPE = { 
    "XI_DATA_PACK_XI_GROUPING": c_uint(0),    #Data grouping (10g160, 12g192, 14g224).
    "XI_DATA_PACK_PFNC_LSB_PACKING": c_uint(1),    #Data packing (10p, 12p)
    }

#Downsampling types
XI_DOWNSAMPLING_TYPE = { 
    "XI_BINNING": c_uint(0),    #Downsampling is using  binning
    "XI_SKIPPING": c_uint(1),    #Downsampling is using  skipping
    }

#Image correction function
XI_IMAGE_CORRECTION_SELECTOR = { 
    "XI_CORRECTION_TYPE_SELECTOR": c_uint(0),    #Correction Type selected see XI_TYPE_CORRECTION_SELECTOR
    "XI_DEFECT_ID": c_uint(1),    #Select defect id
    "XI_DEFECTS_COUNT_BY_TYPE": c_uint(2),    #Count of defects selected by current XI_DEFECT_TYPE
    "XI_DEFECT_TYPE": c_uint(3),    #Type of defect see XI_IMAGE_DEFECT_TYPE
    "XI_DEFECT_SUB_TYPE": c_uint(4),    #Defect sub type see XI_IMAGE_DEFECT_SUB_TYPE
    "XI_DEFECT_POS_X": c_uint(5),    #Defect position x
    "XI_DEFECT_POS_Y": c_uint(6),    #Defect position y
    "XI_DEFECT_CMD_ADD": c_uint(7),    #Write cached defect to the list
    "XI_DEFECT_CMD_DELETE": c_uint(8),    #Delete defect to the list
    "XI_DEFECT_CMD_APPLY_CHANGES": c_uint(9),    #Apply changes
    "XI_DEFECT_CMD_LIST_CLEAR": c_uint(10),    #Clear list
    "XI_DEFECT_CMD_LISTS_CLEAR": c_uint(11),    #Clear lists
    "XI_DEFECT_CMD_SAVE": c_uint(12),    #Save list to device
    "XI_CORRECTION_TYPE_ENABLED": c_uint(13),    #Enable or disable correction type
    "XI_DEFECT_ID_BY_TYPE": c_uint(14),    #Select defect id by type
    "XI_LIST_ID": c_uint(15),    #Select list id
    "XI_DEFECT_CMD_APPLY_CHANGES_ALL": c_uint(16),    #Apply changes to all lists
    "XI_LIST_STATUS": c_uint(17),    #Current list status (Read-only). Result is mask of bits XI_LIST_STATUS_GENERATED, XI_LIST_STATUS_...
    "XI_IMG_COR_TAP_SELECTOR": c_uint(64),    #Selected tap id (0-N) for image correction
    "XI_IMG_COR_GAIN_TUNE": c_uint(65),    #Adjustment of gain in dB. For multitap sensors, active tap is selected by XI_IMG_COR_TAP_SELECTOR.
    "XI_IMG_COR_OFFSET_TUNE": c_uint(66),    #Adjustment of pixel values offset. For multitap sensors, active tap is selected by XI_IMG_COR_TAP_SELECTOR.
    }

#Define image  correction type
XI_TYPE_CORRECTION_SELECTOR = { 
    "XI_CORR_TYPE_SENSOR_DEFECTS_FACTORY": c_uint(0),    #Factory defect list
    "XI_CORR_TYPE_SENSOR_COLUMN_FPN": c_uint(1),    #Select Fixed Pattern Noise Correction for Columns
    "XI_CORR_TYPE_SENSOR_ADC_BLO": c_uint(2),    #ADC gain and black level offset sensor register correction
    "XI_CORR_TYPE_SENSOR_ROW_FPN": c_uint(3),    #Select Fixed Pattern Noise Correction for Rows
    "XI_CORR_TYPE_SENSOR_DEFECTS_USER0": c_uint(4),    #User defect list
    "XI_CORR_TYPE_SENSOR_CHANNELS_TUNE": c_uint(5),    #Image channel/tap intensity correction
    }

#Define image defect types
XI_IMAGE_DEFECT_TYPE = { 
    "XI_IMAGE_DEFECT_TYPE_PIXEL": c_uint(0),    #Defect is pixel
    "XI_IMAGE_DEFECT_TYPE_COLUMN": c_uint(1),    #Defect is column
    "XI_IMAGE_DEFECT_TYPE_ROW": c_uint(2),    #Defect is row
    }

#Define image defect sub types
XI_IMAGE_DEFECT_SUB_TYPE = { 
    "XI_IMAGE_DEFECT_SUB_TYPE_DARK": c_uint(0),    #Defect pixel(s) is(are) too dark
    "XI_IMAGE_DEFECT_SUB_TYPE_BRIGHT": c_uint(1),    #Defect pixel(s) is(are) out of range
    "XI_IMAGE_DEFECT_SUB_TYPE_HOT": c_uint(2),    #Defect pixel(s) is(are) too bright
    }

#Gain selector
XI_GAIN_SELECTOR_TYPE = { 
    "XI_GAIN_SELECTOR_ALL": c_uint(0),    #Gain selector selects all channels. Implementation of gain type depends on camera.
    "XI_GAIN_SELECTOR_ANALOG_ALL": c_uint(1),    #Gain selector selects all analog channels. This is available only on some cameras.
    "XI_GAIN_SELECTOR_DIGITAL_ALL": c_uint(2),    #Gain selector selects all digital channels. This is available only on some cameras.
    "XI_GAIN_SELECTOR_ANALOG_TAP1": c_uint(3),    #Gain selector selects tap 1. This is available only on some cameras.
    "XI_GAIN_SELECTOR_ANALOG_TAP2": c_uint(4),    #Gain selector selects tap 2. This is available only on some cameras.
    "XI_GAIN_SELECTOR_ANALOG_TAP3": c_uint(5),    #Gain selector selects tap 3. This is available only on some cameras.
    "XI_GAIN_SELECTOR_ANALOG_TAP4": c_uint(6),    #Gain selector selects tap 4. This is available only on some cameras.
    }

#Shutter mode types
XI_SHUTTER_TYPE = { 
    "XI_SHUTTER_GLOBAL": c_uint(0),    #Sensor Global Shutter(CMOS sensor)
    "XI_SHUTTER_ROLLING": c_uint(1),    #Sensor Electronic Rolling Shutter(CMOS sensor)
    "XI_SHUTTER_GLOBAL_RESET_RELEASE": c_uint(2),    #Sensor Global Reset Release Shutter(CMOS sensor)
    }

#structure containing information about CMS functionality
XI_CMS_MODE = { 
    "XI_CMS_DIS": c_uint(0),    #CMS disable
    "XI_CMS_EN": c_uint(1),    #CMS enable
    "XI_CMS_EN_FAST": c_uint(2),    #CMS enable(fast)
    }

#structure containing information about ICC Intents
XI_CMS_INTENT = { 
    "XI_CMS_INTENT_PERCEPTUAL": c_uint(0),    #CMS intent perceptual
    "XI_CMS_INTENT_RELATIVE_COLORIMETRIC": c_uint(1),    #CMS intent relative colorimetric
    "XI_CMS_INTENT_SATURATION": c_uint(2),    #CMS intent saturation
    "XI_CMS_INTENT_ABSOLUTE_COLORIMETRIC": c_uint(3),    #CMS intent absolute colorimetric
    }

#structure containing information about options for selection of camera before onening
XI_OPEN_BY = { 
    "XI_OPEN_BY_INST_PATH": c_uint(0),    #Open camera by its hardware path
    "XI_OPEN_BY_SN": c_uint(1),    #Open camera by its serial number
    "XI_OPEN_BY_USER_ID": c_uint(2),    #open camera by its custom user ID
    "XI_OPEN_BY_LOC_PATH": c_uint(3),    #Open camera by its hardware location path
    }

#Lens feature selector selects which feature will be accessed.
XI_LENS_FEATURE = { 
    "XI_LENS_FEATURE_MOTORIZED_FOCUS_SWITCH": c_uint(1),    #Status of lens motorized focus switch
    "XI_LENS_FEATURE_MOTORIZED_FOCUS_BOUNDED": c_uint(2),    #On read = 1 if motorized focus is on one of limits.
    "XI_LENS_FEATURE_MOTORIZED_FOCUS_CALIBRATION": c_uint(3),    #On read = 1 if motorized focus is calibrated. Write 1 to start calibration.
    "XI_LENS_FEATURE_IMAGE_STABILIZATION_ENABLED": c_uint(4),    #On read = 1 if image stabilization is enabled. Write 1 to enable image stabilization.
    "XI_LENS_FEATURE_IMAGE_STABILIZATION_SWITCH_STATUS": c_uint(5),    #On read = 1 if image stabilization switch is in position On.
    "XI_LENS_FEATURE_IMAGE_ZOOM_SUPPORTED": c_uint(6),    #On read = 1 if lens supports zoom = are not prime.
    }

#Sensor feature selector selects which feature will be accessed.
XI_SENSOR_FEATURE_SELECTOR = { 
    "XI_SENSOR_FEATURE_ZEROROT_ENABLE": c_uint(0),    #Zero ROT enable for ONSEMI PYTHON family
    "XI_SENSOR_FEATURE_BLACK_LEVEL_CLAMP": c_uint(1),    #Black level offset clamping
    "XI_SENSOR_FEATURE_MD_FPGA_DIGITAL_GAIN_DISABLE": c_uint(2),    #Disable digital component of gain for MD family
    "XI_SENSOR_FEATURE_ACQUISITION_RUNNING": c_uint(3),    #Sensor acquisition is running status. Could be stopped by setting of 0.
    }

#Extended feature selector.
XI_EXT_FEATURE_SELECTOR = { 
    "XI_EXT_FEATURE_SEL_SIMULATOR_GENERATOR_FRAME_LOST_PERIOD_MIN": c_uint(1),    #Camera simulator lost frame generation minimum period (in frames).
    "XI_EXT_FEATURE_SEL_SIMULATOR_GENERATOR_FRAME_LOST_PERIOD_MAX": c_uint(2),    #Camera simulator lost frame generation random period (in frames).
    "XI_EXT_FEATURE_SEL_SIMULATOR_IMAGE_DATA_FORMAT": c_uint(3),    #Camera simulator image data format.
    "XI_EXT_FEATURE_SEL_BANDWIDTH_MEASUREMENT_TIME_SECONDS": c_uint(4),    #Number of seconds for bandwidth measurement. Default = 1.
    "XI_EXT_FEATURE_SEL_IMAGE_INTENSIFIER_VOLTAGE": c_uint(5),    #Input voltage for image intensifier. Default = 0.
    "XI_EXT_FEATURE_SEL_TRIG_FRAME": c_uint(6),    #Triggers frame(s) on internal event. Default = 0.
    "XI_EXT_FEATURE_SEL_IMAGE_OVERSAMPLING": c_uint(7),    #Enable/disable image pixels oversampling. Default = 0.
    "XI_EXT_FEATURE_SEL_APPLY_DATA_FINAL": c_uint(8),    #Enable/disable applying data final. Default = 1.
    "XI_EXT_FEATURE_SEL_FAN_RPM": c_uint(9),    #Sets camera cooling fan rpm (% from max). Default = 100.
    "XI_EXT_FEATURE_SEL_DITHERING_HOST": c_uint(10),    #Enables/Disables shifted(left/up) image data dithering on HOST side. Default = 0(off).
    "XI_EXT_FEATURE_SEL_DITHERING_DEVICE": c_uint(11),    #Enables/Disables shifted(left/up) image data dithering on DEVICE side. Default = 0(off).
    "XI_EXT_FEATURE_SEL_FAN_THR_TEMP": c_uint(12),    #Sets camera fan/back side threshold temperature. Default = 35.
    }

#Device unit selector
XI_DEVICE_UNIT_SELECTOR = { 
    "XI_DEVICE_UNIT_SENSOR1": c_uint(0),    #Selects first sensor on device
    "XI_DEVICE_UNIT_FPGA1": c_uint(1),    #Selects first FPGA on device
    "XI_DEVICE_UNIT_SAL": c_uint(2),    #Selects sensor abstraction layer
    "XI_DEVICE_UNIT_DAL": c_uint(3),    #Selects driver abstraction layer
    "XI_DEVICE_UNIT_SCM": c_uint(4),    #Selects sensor correction module
    "XI_DEVICE_UNIT_FGENTL": c_uint(5),    #Selects register in underlying GenTL layer
    "XI_DEVICE_UNIT_MCU1": c_uint(6),    #Selects first MCU on device
    "XI_DEVICE_UNIT_MCU2": c_uint(7),    #Selects second MCU on device
    "XI_DEVICE_UNIT_CHF": c_uint(8),    #Selects Camera High Features Model
    }

#Camera sensor mode enumerator.
XI_SENSOR_MODE = { 
    "XI_SENS_MD0": c_uint(0),    #Sensor mode number 0
    "XI_SENS_MD1": c_uint(1),    #Sensor mode number 1
    "XI_SENS_MD2": c_uint(2),    #Sensor mode number 2
    "XI_SENS_MD3": c_uint(3),    #Sensor mode number 3
    "XI_SENS_MD4": c_uint(4),    #Sensor mode number 4
    "XI_SENS_MD5": c_uint(5),    #Sensor mode number 5
    "XI_SENS_MD6": c_uint(6),    #Sensor mode number 6
    "XI_SENS_MD7": c_uint(7),    #Sensor mode number 7
    "XI_SENS_MD8": c_uint(8),    #Sensor mode number 8
    "XI_SENS_MD9": c_uint(9),    #Sensor mode number 9
    "XI_SENS_MD10": c_uint(10),    #Sensor mode number 10
    "XI_SENS_MD11": c_uint(11),    #Sensor mode number 11
    "XI_SENS_MD12": c_uint(12),    #Sensor mode number 12
    "XI_SENS_MD13": c_uint(13),    #Sensor mode number 13
    "XI_SENS_MD14": c_uint(14),    #Sensor mode number 14
    "XI_SENS_MD15": c_uint(15),    #Sensor mode number 15
    }

#Camera channel count enumerator.
XI_SENSOR_OUTPUT_CHANNEL_COUNT = { 
    "XI_CHANN_CNT2": c_uint(2),    #2 sensor readout channels.
    "XI_CHANN_CNT4": c_uint(4),    #4 sensor readout channels.
    "XI_CHANN_CNT8": c_uint(8),    #8 sensor readout channels.
    "XI_CHANN_CNT16": c_uint(16),    #16 sensor readout channels.
    "XI_CHANN_CNT32": c_uint(32),    #32 sensor readout channels.
    }

#Sensor defects correction list selector
XI_SENS_DEFFECTS_CORR_LIST_SELECTOR = { 
    "XI_SENS_DEFFECTS_CORR_LIST_SEL_FACTORY": c_uint(0),    #Factory defect correction list
    "XI_SENS_DEFFECTS_CORR_LIST_SEL_USER0": c_uint(1),    #User defect correction list
    }

#Acquisition status Selector
XI_ACQUISITION_STATUS_SELECTOR = { 
    "XI_ACQUISITION_STATUS_ACQ_ACTIVE": c_uint(0),    # Device is currently doing an acquisition of one or many frames.
    }


XI_GenTL_Image_Format_e = { 
    "XI_GenTL_Image_Format_Mono8": c_uint(0x01080001),    
    }
	
# Parameters

XI_PRM_EXPOSURE = "exposure"    #Exposure time in microseconds
XI_PRM_EXPOSURE_BURST_COUNT = "exposure_burst_count"    #Sets the number of times of exposure in one frame.
XI_PRM_GAIN_SELECTOR = "gain_selector"    #Gain selector for parameter Gain allows to select different type of gains.
XI_PRM_GAIN = "gain"    #Gain in dB
XI_PRM_DOWNSAMPLING = "downsampling"    #Change image resolution by binning or skipping.
XI_PRM_DOWNSAMPLING_TYPE = "downsampling_type"    #Change image downsampling type.
XI_PRM_TEST_PATTERN_GENERATOR_SELECTOR = "test_pattern_generator_selector"    #Selects which test pattern generator is controlled by the TestPattern feature.
XI_PRM_TEST_PATTERN = "test_pattern"    #Selects which test pattern type is generated by the selected generator.
XI_PRM_IMAGE_DATA_FORMAT = "imgdataformat"    #Output data format.
XI_PRM_SHUTTER_TYPE = "shutter_type"    #Change sensor shutter type(CMOS sensor).
XI_PRM_SENSOR_TAPS = "sensor_taps"    #Number of taps
XI_PRM_AEAG = "aeag"    #Automatic exposure/gain
XI_PRM_AEAG_ROI_OFFSET_X = "aeag_roi_offset_x"    #Automatic exposure/gain ROI offset X
XI_PRM_AEAG_ROI_OFFSET_Y = "aeag_roi_offset_y"    #Automatic exposure/gain ROI offset Y
XI_PRM_AEAG_ROI_WIDTH = "aeag_roi_width"    #Automatic exposure/gain ROI Width
XI_PRM_AEAG_ROI_HEIGHT = "aeag_roi_height"    #Automatic exposure/gain ROI Height
XI_PRM_SENS_DEFECTS_CORR_LIST_SELECTOR = "bpc_list_selector"    #Selector of list used by Sensor Defects Correction parameter
XI_PRM_SENS_DEFECTS_CORR_LIST_CONTENT = "sens_defects_corr_list_content"    #Sets/Gets sensor defects list in special text format
XI_PRM_SENS_DEFECTS_CORR = "bpc"    #Correction of sensor defects (pixels, columns, rows) enable/disable
XI_PRM_AUTO_WB = "auto_wb"    #Automatic white balance
XI_PRM_MANUAL_WB = "manual_wb"    #Calculates White Balance(xiGetImage function must be called)
XI_PRM_WB_KR = "wb_kr"    #White balance red coefficient
XI_PRM_WB_KG = "wb_kg"    #White balance green coefficient
XI_PRM_WB_KB = "wb_kb"    #White balance blue coefficient
XI_PRM_WIDTH = "width"    #Width of the Image provided by the device (in pixels).
XI_PRM_HEIGHT = "height"    #Height of the Image provided by the device (in pixels).
XI_PRM_OFFSET_X = "offsetX"    #Horizontal offset from the origin to the area of interest (in pixels).
XI_PRM_OFFSET_Y = "offsetY"    #Vertical offset from the origin to the area of interest (in pixels).
XI_PRM_REGION_SELECTOR = "region_selector"    #Selects Region in Multiple ROI which parameters are set by width, height, ... ,region mode
XI_PRM_REGION_MODE = "region_mode"    #Activates/deactivates Region selected by Region Selector
XI_PRM_HORIZONTAL_FLIP = "horizontal_flip"    #Horizontal flip enable
XI_PRM_VERTICAL_FLIP = "vertical_flip"    #Vertical flip enable
XI_PRM_FFC = "ffc"    #Image flat field correction
XI_PRM_FFC_FLAT_FIELD_FILE_NAME = "ffc_flat_field_file_name"    #Set name of file to be applied for FFC processor.
XI_PRM_FFC_DARK_FIELD_FILE_NAME = "ffc_dark_field_file_name"    #Set name of file to be applied for FFC processor.
XI_PRM_BINNING_SELECTOR = "binning_selector"    #Binning engine selector.
XI_PRM_BINNING_VERTICAL_MODE = "binning_vertical_mode"    #Sets the mode to use to combine vertical pixel together.
XI_PRM_BINNING_VERTICAL = "binning_vertical"    #Vertical Binning - number of vertical photo-sensitive cells to combine together.
XI_PRM_BINNING_HORIZONTAL_MODE = "binning_horizontal_mode"    #Sets the mode to use to combine horizontal pixel together.
XI_PRM_BINNING_HORIZONTAL = "binning_horizontal"    #Horizontal Binning - number of horizontal photo-sensitive cells to combine together.
XI_PRM_BINNING_HORIZONTAL_PATTERN = "binning_horizontal_pattern"    #Binning horizontal pattern type.
XI_PRM_BINNING_VERTICAL_PATTERN = "binning_vertical_pattern"    #Binning vertical pattern type.
XI_PRM_DECIMATION_SELECTOR = "decimation_selector"    #Decimation engine selector.
XI_PRM_DECIMATION_VERTICAL = "decimation_vertical"    #Vertical Decimation - vertical sub-sampling of the image - reduces the vertical resolution of the image by the specified vertical decimation factor.
XI_PRM_DECIMATION_HORIZONTAL = "decimation_horizontal"    #Horizontal Decimation - horizontal sub-sampling of the image - reduces the horizontal resolution of the image by the specified vertical decimation factor.
XI_PRM_DECIMATION_HORIZONTAL_PATTERN = "decimation_horizontal_pattern"    #Decimation horizontal pattern type.
XI_PRM_DECIMATION_VERTICAL_PATTERN = "decimation_vertical_pattern"    #Decimation vertical pattern type.
XI_PRM_EXP_PRIORITY = "exp_priority"    #Exposure priority (0.8 - exposure 80%, gain 20%).
XI_PRM_AG_MAX_LIMIT = "ag_max_limit"    #Maximum limit of gain in AEAG procedure
XI_PRM_AE_MAX_LIMIT = "ae_max_limit"    #Maximum time (us) used for exposure in AEAG procedure
XI_PRM_AEAG_LEVEL = "aeag_level"    #Average intensity of output signal AEAG should achieve(in %)
XI_PRM_LIMIT_BANDWIDTH = "limit_bandwidth"    #Set/get bandwidth(datarate)(in Megabits)
XI_PRM_LIMIT_BANDWIDTH_MODE = "limit_bandwidth_mode"    #Bandwidth limit enabled
XI_PRM_SENSOR_LINE_PERIOD = "sensor_line_period"    #Image sensor line period in us
XI_PRM_SENSOR_DATA_BIT_DEPTH = "sensor_bit_depth"    #Sensor output data bit depth.
XI_PRM_OUTPUT_DATA_BIT_DEPTH = "output_bit_depth"    #Device output data bit depth.
XI_PRM_IMAGE_DATA_BIT_DEPTH = "image_data_bit_depth"    #bitdepth of data returned by function xiGetImage
XI_PRM_OUTPUT_DATA_PACKING = "output_bit_packing"    #Device output data packing (or grouping) enabled. Packing could be enabled if output_data_bit_depth > 8 and packing capability is available.
XI_PRM_OUTPUT_DATA_PACKING_TYPE = "output_bit_packing_type"    #Data packing type. Some cameras supports only specific packing type.
XI_PRM_IS_COOLED = "iscooled"    #Returns 1 for cameras that support cooling.
XI_PRM_COOLING = "cooling"    #Temperature control mode.
XI_PRM_TARGET_TEMP = "target_temp"    #Set sensor target temperature for cooling.
XI_PRM_TEMP_SELECTOR = "temp_selector"    #Selector of mechanical point where thermometer is located.
XI_PRM_TEMP = "temp"    #Camera temperature (selected by XI_PRM_TEMP_SELECTOR)
XI_PRM_TEMP_CONTROL_MODE = "device_temperature_ctrl_mode"    #Temperature control mode.
XI_PRM_CHIP_TEMP = "chip_temp"    #Camera sensor temperature
XI_PRM_HOUS_TEMP = "hous_temp"    #Camera housing tepmerature
XI_PRM_HOUS_BACK_SIDE_TEMP = "hous_back_side_temp"    #Camera housing back side tepmerature
XI_PRM_SENSOR_BOARD_TEMP = "sensor_board_temp"    #Camera sensor board temperature
XI_PRM_TEMP_ELEMENT_SEL = "device_temperature_element_sel"    #Temperature element selector (TEC(Peltier), Fan).
XI_PRM_TEMP_ELEMENT_VALUE = "device_temperature_element_val"    #Temperature element value in percents of full control range
XI_PRM_CMS = "cms"    #Mode of color management system.
XI_PRM_CMS_INTENT = "cms_intent"    #Intent of color management system.
XI_PRM_APPLY_CMS = "apply_cms"    #Enable applying of CMS profiles to xiGetImage (see XI_PRM_INPUT_CMS_PROFILE, XI_PRM_OUTPUT_CMS_PROFILE).
XI_PRM_INPUT_CMS_PROFILE = "input_cms_profile"    #Filename for input cms profile (e.g. input.icc)
XI_PRM_OUTPUT_CMS_PROFILE = "output_cms_profile"    #Filename for output cms profile (e.g. input.icc)
XI_PRM_IMAGE_IS_COLOR = "iscolor"    #Returns 1 for color cameras.
XI_PRM_COLOR_FILTER_ARRAY = "cfa"    #Returns color filter array type of RAW data.
XI_PRM_GAMMAY = "gammaY"    #Luminosity gamma
XI_PRM_GAMMAC = "gammaC"    #Chromaticity gamma
XI_PRM_SHARPNESS = "sharpness"    #Sharpness Strenght
XI_PRM_CC_MATRIX_00 = "ccMTX00"    #Color Correction Matrix element [0][0]
XI_PRM_CC_MATRIX_01 = "ccMTX01"    #Color Correction Matrix element [0][1]
XI_PRM_CC_MATRIX_02 = "ccMTX02"    #Color Correction Matrix element [0][2]
XI_PRM_CC_MATRIX_03 = "ccMTX03"    #Color Correction Matrix element [0][3]
XI_PRM_CC_MATRIX_10 = "ccMTX10"    #Color Correction Matrix element [1][0]
XI_PRM_CC_MATRIX_11 = "ccMTX11"    #Color Correction Matrix element [1][1]
XI_PRM_CC_MATRIX_12 = "ccMTX12"    #Color Correction Matrix element [1][2]
XI_PRM_CC_MATRIX_13 = "ccMTX13"    #Color Correction Matrix element [1][3]
XI_PRM_CC_MATRIX_20 = "ccMTX20"    #Color Correction Matrix element [2][0]
XI_PRM_CC_MATRIX_21 = "ccMTX21"    #Color Correction Matrix element [2][1]
XI_PRM_CC_MATRIX_22 = "ccMTX22"    #Color Correction Matrix element [2][2]
XI_PRM_CC_MATRIX_23 = "ccMTX23"    #Color Correction Matrix element [2][3]
XI_PRM_CC_MATRIX_30 = "ccMTX30"    #Color Correction Matrix element [3][0]
XI_PRM_CC_MATRIX_31 = "ccMTX31"    #Color Correction Matrix element [3][1]
XI_PRM_CC_MATRIX_32 = "ccMTX32"    #Color Correction Matrix element [3][2]
XI_PRM_CC_MATRIX_33 = "ccMTX33"    #Color Correction Matrix element [3][3]
XI_PRM_DEFAULT_CC_MATRIX = "defccMTX"    #Set default Color Correction Matrix
XI_PRM_TRG_SOURCE = "trigger_source"    #Defines source of trigger.
XI_PRM_TRG_SOFTWARE = "trigger_software"    #Generates an internal trigger. XI_PRM_TRG_SOURCE must be set to TRG_SOFTWARE.
XI_PRM_TRG_SELECTOR = "trigger_selector"    #Selects the type of trigger.
XI_PRM_TRG_OVERLAP = "trigger_overlap"    #The mode of Trigger Overlap. This influences of trigger acception/rejection policy
XI_PRM_ACQ_FRAME_BURST_COUNT = "acq_frame_burst_count"    #Sets number of frames acquired by burst. This burst is used only if trigger is set to FrameBurstStart
XI_PRM_GPI_SELECTOR = "gpi_selector"    #Selects GPI
XI_PRM_GPI_MODE = "gpi_mode"    #Defines GPI functionality
XI_PRM_GPI_LEVEL = "gpi_level"    #GPI level
XI_PRM_GPO_SELECTOR = "gpo_selector"    #Selects GPO
XI_PRM_GPO_MODE = "gpo_mode"    #Defines GPO functionality
XI_PRM_LED_SELECTOR = "led_selector"    #Selects LED
XI_PRM_LED_MODE = "led_mode"    #Defines LED functionality
XI_PRM_DEBOUNCE_EN = "dbnc_en"    #Enable/Disable debounce to selected GPI
XI_PRM_DEBOUNCE_T0 = "dbnc_t0"    #Debounce time (x * 10us)
XI_PRM_DEBOUNCE_T1 = "dbnc_t1"    #Debounce time (x * 10us)
XI_PRM_DEBOUNCE_POL = "dbnc_pol"    #Debounce polarity (pol = 1 t0 - falling edge, t1 - rising edge)
XI_PRM_LENS_MODE = "lens_mode"    #Status of lens control interface. This shall be set to XI_ON before any Lens operations.
XI_PRM_LENS_APERTURE_VALUE = "lens_aperture_value"    #Current lens aperture value in stops. Examples: 2.8, 4, 5.6, 8, 11
XI_PRM_LENS_FOCUS_MOVEMENT_VALUE = "lens_focus_movement_value"    #Lens current focus movement value to be used by XI_PRM_LENS_FOCUS_MOVE in motor steps.
XI_PRM_LENS_FOCUS_MOVE = "lens_focus_move"    #Moves lens focus motor by steps set in XI_PRM_LENS_FOCUS_MOVEMENT_VALUE.
XI_PRM_LENS_FOCUS_DISTANCE = "lens_focus_distance"    #Lens focus distance in cm.
XI_PRM_LENS_FOCAL_LENGTH = "lens_focal_length"    #Lens focal distance in mm.
XI_PRM_LENS_FEATURE_SELECTOR = "lens_feature_selector"    #Selects the current feature which is accessible by XI_PRM_LENS_FEATURE.
XI_PRM_LENS_FEATURE = "lens_feature"    #Allows access to lens feature value currently selected by XI_PRM_LENS_FEATURE_SELECTOR.
XI_PRM_LENS_COMM_DATA = "lens_comm_data"    #Write/Read data sequences to/from lens
XI_PRM_DEVICE_NAME = "device_name"    #Return device name
XI_PRM_DEVICE_TYPE = "device_type"    #Return device type
XI_PRM_DEVICE_MODEL_ID = "device_model_id"    #Return device model id
XI_PRM_SENSOR_MODEL_ID = "sensor_model_id"    #Return device sensor model id
XI_PRM_DEVICE_SN = "device_sn"    #Return device serial number
XI_PRM_DEVICE_SENS_SN = "device_sens_sn"    #Return sensor serial number
XI_PRM_DEVICE_ID = "device_id"    #Return unique device ID
XI_PRM_DEVICE_INSTANCE_PATH = "device_inst_path"    #Return device system instance path.
XI_PRM_DEVICE_LOCATION_PATH = "device_loc_path"    #Represents the location of the device in the device tree.
XI_PRM_DEVICE_USER_ID = "device_user_id"    #Return custom ID of camera.
XI_PRM_DEVICE_MANIFEST = "device_manifest"    #Return device capability description XML.
XI_PRM_IMAGE_USER_DATA = "image_user_data"    #User image data at image header to track parameters synchronization.
XI_PRM_IMAGE_DATA_FORMAT_RGB32_ALPHA = "imgdataformatrgb32alpha"    #The alpha channel of RGB32 output image format.
XI_PRM_IMAGE_PAYLOAD_SIZE = "imgpayloadsize"    #Buffer size in bytes sufficient for output image returned by xiGetImage
XI_PRM_TRANSPORT_PIXEL_FORMAT = "transport_pixel_format"    #Current format of pixels on transport layer.
XI_PRM_TRANSPORT_DATA_TARGET = "transport_data_target"    #Target selector for data - CPU RAM or GPU RAM
XI_PRM_SENSOR_CLOCK_FREQ_HZ = "sensor_clock_freq_hz"    #Sensor clock frequency in Hz.
XI_PRM_SENSOR_CLOCK_FREQ_INDEX = "sensor_clock_freq_index"    #Sensor clock frequency index. Sensor with selected frequencies have possibility to set the frequency only by this index.
XI_PRM_SENSOR_OUTPUT_CHANNEL_COUNT = "sensor_output_channel_count"    #Number of output channels from sensor used for data transfer.
XI_PRM_FRAMERATE = "framerate"    #Define framerate in Hz
XI_PRM_COUNTER_SELECTOR = "counter_selector"    #Select counter
XI_PRM_COUNTER_VALUE = "counter_value"    #Counter status
XI_PRM_ACQ_TIMING_MODE = "acq_timing_mode"    #Type of sensor frames timing.
XI_PRM_AVAILABLE_BANDWIDTH = "available_bandwidth"    #Measure and return available interface bandwidth(int Megabits)
XI_PRM_BUFFER_POLICY = "buffer_policy"    #Data move policy
XI_PRM_LUT_EN = "LUTEnable"    #Activates LUT.
XI_PRM_LUT_INDEX = "LUTIndex"    #Control the index (offset) of the coefficient to access in the LUT.
XI_PRM_LUT_VALUE = "LUTValue"    #Value at entry LUTIndex of the LUT
XI_PRM_TRG_DELAY = "trigger_delay"    #Specifies the delay in microseconds (us) to apply after the trigger reception before activating it.
XI_PRM_TS_RST_MODE = "ts_rst_mode"    #Defines how time stamp reset engine will be armed
XI_PRM_TS_RST_SOURCE = "ts_rst_source"    #Defines which source will be used for timestamp reset. Writing this parameter will trigger settings of engine (arming)
XI_PRM_IS_DEVICE_EXIST = "isexist"    #Returns 1 if camera connected and works properly.
XI_PRM_ACQ_BUFFER_SIZE = "acq_buffer_size"    #Acquisition buffer size in buffer_size_unit. Default bytes.
XI_PRM_ACQ_BUFFER_SIZE_UNIT = "acq_buffer_size_unit"    #Acquisition buffer size unit in bytes. Default 1. E.g. Value 1024 means that buffer_size is in KiBytes
XI_PRM_ACQ_TRANSPORT_BUFFER_SIZE = "acq_transport_buffer_size"    #Acquisition transport buffer size in bytes
XI_PRM_ACQ_TRANSPORT_PACKET_SIZE = "acq_transport_packet_size"    #Acquisition transport packet size in bytes
XI_PRM_BUFFERS_QUEUE_SIZE = "buffers_queue_size"    #Queue of field/frame buffers
XI_PRM_ACQ_TRANSPORT_BUFFER_COMMIT = "acq_transport_buffer_commit"    #Number of buffers to commit to low level
XI_PRM_RECENT_FRAME = "recent_frame"    #GetImage returns most recent frame
XI_PRM_DEVICE_RESET = "device_reset"    #Resets the camera to default state.
XI_PRM_COLUMN_FPN_CORRECTION = "column_fpn_correction"    #Correction of column FPN
XI_PRM_ROW_FPN_CORRECTION = "row_fpn_correction"    #Correction of row FPN
XI_PRM_IMAGE_CORRECTION_SELECTOR = "image_correction_selector"    #Select image correction function
XI_PRM_IMAGE_CORRECTION_VALUE = "image_correction_value"    #Select image correction selected function value
XI_PRM_SENSOR_MODE = "sensor_mode"    #Current sensor mode. Allows to select sensor mode by one integer. Setting of this parameter affects: image dimensions and downsampling.
XI_PRM_HDR = "hdr"    #Enable High Dynamic Range feature.
XI_PRM_HDR_KNEEPOINT_COUNT = "hdr_kneepoint_count"    #The number of kneepoints in the PWLR.
XI_PRM_HDR_T1 = "hdr_t1"    #position of first kneepoint(in % of XI_PRM_EXPOSURE)
XI_PRM_HDR_T2 = "hdr_t2"    #position of second kneepoint (in % of XI_PRM_EXPOSURE)
XI_PRM_KNEEPOINT1 = "hdr_kneepoint1"    #value of first kneepoint (% of sensor saturation)
XI_PRM_KNEEPOINT2 = "hdr_kneepoint2"    #value of second kneepoint (% of sensor saturation)
XI_PRM_IMAGE_BLACK_LEVEL = "image_black_level"    #Last image black level counts. Can be used for Offline processing to recall it.
XI_PRM_API_VERSION = "api_version"    #Returns version of API.
XI_PRM_DRV_VERSION = "drv_version"    #Returns version of current device driver.
XI_PRM_MCU1_VERSION = "version_mcu1"    #Returns version of MCU1 firmware.
XI_PRM_MCU2_VERSION = "version_mcu2"    #Returns version of MCU2 firmware.
XI_PRM_MCU3_VERSION = "version_mcu3"    #Returns version of MCU3 firmware.
XI_PRM_FPGA1_VERSION = "version_fpga1"    #Returns version of FPGA1 firmware.
XI_PRM_XMLMAN_VERSION = "version_xmlman"    #Returns version of XML manifest.
XI_PRM_HW_REVISION = "hw_revision"    #Returns hardware revision number.
XI_PRM_DEBUG_LEVEL = "debug_level"    #Set debug level
XI_PRM_AUTO_BANDWIDTH_CALCULATION = "auto_bandwidth_calculation"    #Automatic bandwidth calculation,
XI_PRM_NEW_PROCESS_CHAIN_ENABLE = "new_process_chain_enable"    #Enables (2015/FAPI) processing chain for MQ MU cameras
XI_PRM_CAM_ENUM_GOLDEN_ENABLED = "cam_enum_golden_enabled"    #Enable enumeration of golden devices
XI_PRM_RESET_USB_IF_BOOTLOADER = "reset_usb_if_bootloader"    #Resets USB device if started as bootloader
XI_PRM_CAM_SIMULATORS_COUNT = "cam_simulators_count"    #Number of camera simulators to be available.
XI_PRM_CAM_SENSOR_INIT_DISABLED = "cam_sensor_init_disabled"    #Camera sensor will not be initialized when 1=XI_ON is set.
XI_PRM_READ_FILE_FFS = "read_file_ffs"    #Read file from camera flash filesystem.
XI_PRM_WRITE_FILE_FFS = "write_file_ffs"    #Write file to camera flash filesystem.
XI_PRM_FFS_FILE_NAME = "ffs_file_name"    #Set name of file to be written/read from camera FFS.
XI_PRM_FFS_FILE_ID = "ffs_file_id"    #File number.
XI_PRM_FFS_FILE_SIZE = "ffs_file_size"    #Size of file.
XI_PRM_FREE_FFS_SIZE = "free_ffs_size"    #Size of free camera FFS.
XI_PRM_USED_FFS_SIZE = "used_ffs_size"    #Size of used camera FFS.
XI_PRM_FFS_ACCESS_KEY = "ffs_access_key"    #Setting of key enables file operations on some cameras.
XI_PRM_API_CONTEXT_LIST = "xiapi_context_list"    #List of current parameters settings context - parameters with values. Used for offline processing.
XI_PRM_SENSOR_FEATURE_SELECTOR = "sensor_feature_selector"    #Selects the current feature which is accessible by XI_PRM_SENSOR_FEATURE_VALUE.
XI_PRM_SENSOR_FEATURE_VALUE = "sensor_feature_value"    #Allows access to sensor feature value currently selected by XI_PRM_SENSOR_FEATURE_SELECTOR.
XI_PRM_EXTENDED_FEATURE_SELECTOR = "ext_feature_selector"    #Selection of extended feature.
XI_PRM_EXTENDED_FEATURE = "ext_feature"    #Extended feature value.
XI_PRM_DEVICE_UNIT_SELECTOR = "device_unit_selector"    #Selects device unit.
XI_PRM_DEVICE_UNIT_REGISTER_SELECTOR = "device_unit_register_selector"    #Selects register of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).
XI_PRM_DEVICE_UNIT_REGISTER_VALUE = "device_unit_register_value"    #Sets/gets register value of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).
XI_PRM_API_PROGRESS_CALLBACK = "api_progress_callback"    #Callback address of pointer that is called upon long tasks (e.g. XI_PRM_WRITE_FILE_FFS).
XI_PRM_ACQUISITION_STATUS_SELECTOR = "acquisition_status_selector"    #Selects the internal acquisition signal to read using XI_PRM_ACQUISITION_STATUS.
XI_PRM_ACQUISITION_STATUS = "acquisition_status"    #Acquisition status(True/False)

VAL_TYPE = {
    "exposure": "xiTypeInteger",    #Exposure time in microseconds
    "exposure_burst_count": "xiTypeInteger",    #Sets the number of times of exposure in one frame.
    "gain_selector": "xiTypeEnum",    #Gain selector for parameter Gain allows to select different type of gains.
    "gain": "xiTypeFloat",    #Gain in dB
    "downsampling": "xiTypeEnum",    #Change image resolution by binning or skipping.
    "downsampling_type": "xiTypeEnum",    #Change image downsampling type.
    "test_pattern_generator_selector": "xiTypeEnum",    #Selects which test pattern generator is controlled by the TestPattern feature.
    "test_pattern": "xiTypeEnum",    #Selects which test pattern type is generated by the selected generator.
    "imgdataformat": "xiTypeEnum",    #Output data format.
    "shutter_type": "xiTypeEnum",    #Change sensor shutter type(CMOS sensor).
    "sensor_taps": "xiTypeEnum",    #Number of taps
    "aeag": "xiTypeBoolean",    #Automatic exposure/gain
    "aeag_roi_offset_x": "xiTypeInteger",    #Automatic exposure/gain ROI offset X
    "aeag_roi_offset_y": "xiTypeInteger",    #Automatic exposure/gain ROI offset Y
    "aeag_roi_width": "xiTypeInteger",    #Automatic exposure/gain ROI Width
    "aeag_roi_height": "xiTypeInteger",    #Automatic exposure/gain ROI Height
    "bpc_list_selector": "xiTypeEnum",    #Selector of list used by Sensor Defects Correction parameter
    "sens_defects_corr_list_content": "xiTypeString",    #Sets/Gets sensor defects list in special text format
    "bpc": "xiTypeBoolean",    #Correction of sensor defects (pixels, columns, rows) enable/disable
    "auto_wb": "xiTypeBoolean",    #Automatic white balance
    "manual_wb": "xiTypeCommand",    #Calculates White Balance(xiGetImage function must be called)
    "wb_kr": "xiTypeFloat",    #White balance red coefficient
    "wb_kg": "xiTypeFloat",    #White balance green coefficient
    "wb_kb": "xiTypeFloat",    #White balance blue coefficient
    "width": "xiTypeInteger",    #Width of the Image provided by the device (in pixels).
    "height": "xiTypeInteger",    #Height of the Image provided by the device (in pixels).
    "offsetX": "xiTypeInteger",    #Horizontal offset from the origin to the area of interest (in pixels).
    "offsetY": "xiTypeInteger",    #Vertical offset from the origin to the area of interest (in pixels).
    "region_selector": "xiTypeInteger",    #Selects Region in Multiple ROI which parameters are set by width, height, ... ,region mode
    "region_mode": "xiTypeInteger",    #Activates/deactivates Region selected by Region Selector
    "horizontal_flip": "xiTypeBoolean",    #Horizontal flip enable
    "vertical_flip": "xiTypeBoolean",    #Vertical flip enable
    "ffc": "xiTypeBoolean",    #Image flat field correction
    "ffc_flat_field_file_name": "xiTypeString",    #Set name of file to be applied for FFC processor.
    "ffc_dark_field_file_name": "xiTypeString",    #Set name of file to be applied for FFC processor.
    "binning_selector": "xiTypeEnum",    #Binning engine selector.
    "binning_vertical_mode": "xiTypeEnum",    #Sets the mode to use to combine vertical pixel together.
    "binning_vertical": "xiTypeInteger",    #Vertical Binning - number of vertical photo-sensitive cells to combine together.
    "binning_horizontal_mode": "xiTypeEnum",    #Sets the mode to use to combine horizontal pixel together.
    "binning_horizontal": "xiTypeInteger",    #Horizontal Binning - number of horizontal photo-sensitive cells to combine together.
    "binning_horizontal_pattern": "xiTypeEnum",    #Binning horizontal pattern type.
    "binning_vertical_pattern": "xiTypeEnum",    #Binning vertical pattern type.
    "decimation_selector": "xiTypeEnum",    #Decimation engine selector.
    "decimation_vertical": "xiTypeInteger",    #Vertical Decimation - vertical sub-sampling of the image - reduces the vertical resolution of the image by the specified vertical decimation factor.
    "decimation_horizontal": "xiTypeInteger",    #Horizontal Decimation - horizontal sub-sampling of the image - reduces the horizontal resolution of the image by the specified vertical decimation factor.
    "decimation_horizontal_pattern": "xiTypeEnum",    #Decimation horizontal pattern type.
    "decimation_vertical_pattern": "xiTypeEnum",    #Decimation vertical pattern type.
    "exp_priority": "xiTypeFloat",    #Exposure priority (0.8 - exposure 80%, gain 20%).
    "ag_max_limit": "xiTypeFloat",    #Maximum limit of gain in AEAG procedure
    "ae_max_limit": "xiTypeInteger",    #Maximum time (us) used for exposure in AEAG procedure
    "aeag_level": "xiTypeInteger",    #Average intensity of output signal AEAG should achieve(in %)
    "limit_bandwidth": "xiTypeInteger",    #Set/get bandwidth(datarate)(in Megabits)
    "limit_bandwidth_mode": "xiTypeEnum",    #Bandwidth limit enabled
    "sensor_line_period": "xiTypeFloat",    #Image sensor line period in us
    "sensor_bit_depth": "xiTypeEnum",    #Sensor output data bit depth.
    "output_bit_depth": "xiTypeEnum",    #Device output data bit depth.
    "image_data_bit_depth": "xiTypeEnum",    #bitdepth of data returned by function xiGetImage
    "output_bit_packing": "xiTypeBoolean",    #Device output data packing (or grouping) enabled. Packing could be enabled if output_data_bit_depth > 8 and packing capability is available.
    "output_bit_packing_type": "xiTypeEnum",    #Data packing type. Some cameras supports only specific packing type.
    "iscooled": "xiTypeBoolean",    #Returns 1 for cameras that support cooling.
    "cooling": "xiTypeEnum",    #Temperature control mode.
    "target_temp": "xiTypeFloat",    #Set sensor target temperature for cooling.
    "temp_selector": "xiTypeEnum",    #Selector of mechanical point where thermometer is located.
    "temp": "xiTypeFloat",    #Camera temperature (selected by XI_PRM_TEMP_SELECTOR)
    "device_temperature_ctrl_mode": "xiTypeEnum",    #Temperature control mode.
    "chip_temp": "xiTypeFloat",    #Camera sensor temperature
    "hous_temp": "xiTypeFloat",    #Camera housing tepmerature
    "hous_back_side_temp": "xiTypeFloat",    #Camera housing back side tepmerature
    "sensor_board_temp": "xiTypeFloat",    #Camera sensor board temperature
    "device_temperature_element_sel": "xiTypeEnum",    #Temperature element selector (TEC(Peltier), Fan).
    "device_temperature_element_val": "xiTypeFloat",    #Temperature element value in percents of full control range
    "cms": "xiTypeEnum",    #Mode of color management system.
    "cms_intent": "xiTypeEnum",    #Intent of color management system.
    "apply_cms": "xiTypeBoolean",    #Enable applying of CMS profiles to xiGetImage (see XI_PRM_INPUT_CMS_PROFILE, XI_PRM_OUTPUT_CMS_PROFILE).
    "input_cms_profile": "xiTypeString",    #Filename for input cms profile (e.g. input.icc)
    "output_cms_profile": "xiTypeString",    #Filename for output cms profile (e.g. input.icc)
    "iscolor": "xiTypeBoolean",    #Returns 1 for color cameras.
    "cfa": "xiTypeEnum",    #Returns color filter array type of RAW data.
    "gammaY": "xiTypeFloat",    #Luminosity gamma
    "gammaC": "xiTypeFloat",    #Chromaticity gamma
    "sharpness": "xiTypeFloat",    #Sharpness Strenght
    "ccMTX00": "xiTypeFloat",    #Color Correction Matrix element [0][0]
    "ccMTX01": "xiTypeFloat",    #Color Correction Matrix element [0][1]
    "ccMTX02": "xiTypeFloat",    #Color Correction Matrix element [0][2]
    "ccMTX03": "xiTypeFloat",    #Color Correction Matrix element [0][3]
    "ccMTX10": "xiTypeFloat",    #Color Correction Matrix element [1][0]
    "ccMTX11": "xiTypeFloat",    #Color Correction Matrix element [1][1]
    "ccMTX12": "xiTypeFloat",    #Color Correction Matrix element [1][2]
    "ccMTX13": "xiTypeFloat",    #Color Correction Matrix element [1][3]
    "ccMTX20": "xiTypeFloat",    #Color Correction Matrix element [2][0]
    "ccMTX21": "xiTypeFloat",    #Color Correction Matrix element [2][1]
    "ccMTX22": "xiTypeFloat",    #Color Correction Matrix element [2][2]
    "ccMTX23": "xiTypeFloat",    #Color Correction Matrix element [2][3]
    "ccMTX30": "xiTypeFloat",    #Color Correction Matrix element [3][0]
    "ccMTX31": "xiTypeFloat",    #Color Correction Matrix element [3][1]
    "ccMTX32": "xiTypeFloat",    #Color Correction Matrix element [3][2]
    "ccMTX33": "xiTypeFloat",    #Color Correction Matrix element [3][3]
    "defccMTX": "xiTypeCommand",    #Set default Color Correction Matrix
    "trigger_source": "xiTypeEnum",    #Defines source of trigger.
    "trigger_software": "xiTypeCommand",    #Generates an internal trigger. XI_PRM_TRG_SOURCE must be set to TRG_SOFTWARE.
    "trigger_selector": "xiTypeEnum",    #Selects the type of trigger.
    "trigger_overlap": "xiTypeEnum",    #The mode of Trigger Overlap. This influences of trigger acception/rejection policy
    "acq_frame_burst_count": "xiTypeInteger",    #Sets number of frames acquired by burst. This burst is used only if trigger is set to FrameBurstStart
    "gpi_selector": "xiTypeEnum",    #Selects GPI
    "gpi_mode": "xiTypeEnum",    #Defines GPI functionality
    "gpi_level": "xiTypeInteger",    #GPI level
    "gpo_selector": "xiTypeEnum",    #Selects GPO
    "gpo_mode": "xiTypeEnum",    #Defines GPO functionality
    "led_selector": "xiTypeEnum",    #Selects LED
    "led_mode": "xiTypeEnum",    #Defines LED functionality
    "dbnc_en": "xiTypeBoolean",    #Enable/Disable debounce to selected GPI
    "dbnc_t0": "xiTypeInteger",    #Debounce time (x * 10us)
    "dbnc_t1": "xiTypeInteger",    #Debounce time (x * 10us)
    "dbnc_pol": "xiTypeInteger",    #Debounce polarity (pol = 1 t0 - falling edge, t1 - rising edge)
    "lens_mode": "xiTypeBoolean",    #Status of lens control interface. This shall be set to XI_ON before any Lens operations.
    "lens_aperture_value": "xiTypeFloat",    #Current lens aperture value in stops. Examples: 2.8, 4, 5.6, 8, 11
    "lens_focus_movement_value": "xiTypeInteger",    #Lens current focus movement value to be used by XI_PRM_LENS_FOCUS_MOVE in motor steps.
    "lens_focus_move": "xiTypeCommand",    #Moves lens focus motor by steps set in XI_PRM_LENS_FOCUS_MOVEMENT_VALUE.
    "lens_focus_distance": "xiTypeFloat",    #Lens focus distance in cm.
    "lens_focal_length": "xiTypeFloat",    #Lens focal distance in mm.
    "lens_feature_selector": "xiTypeEnum",    #Selects the current feature which is accessible by XI_PRM_LENS_FEATURE.
    "lens_feature": "xiTypeFloat",    #Allows access to lens feature value currently selected by XI_PRM_LENS_FEATURE_SELECTOR.
    "lens_comm_data": "xiTypeString",    #Write/Read data sequences to/from lens
    "device_name": "xiTypeString",    #Return device name
    "device_type": "xiTypeString",    #Return device type
    "device_model_id": "xiTypeInteger",    #Return device model id
    "sensor_model_id": "xiTypeInteger",    #Return device sensor model id
    "device_sn": "xiTypeString",    #Return device serial number
    "device_sens_sn": "xiTypeString",    #Return sensor serial number
    "device_id": "xiTypeString",    #Return unique device ID
    "device_inst_path": "xiTypeString",    #Return device system instance path.
    "device_loc_path": "xiTypeString",    #Represents the location of the device in the device tree.
    "device_user_id": "xiTypeString",    #Return custom ID of camera.
    "device_manifest": "xiTypeString",    #Return device capability description XML.
    "image_user_data": "xiTypeInteger",    #User image data at image header to track parameters synchronization.
    "imgdataformatrgb32alpha": "xiTypeInteger",    #The alpha channel of RGB32 output image format.
    "imgpayloadsize": "xiTypeInteger",    #Buffer size in bytes sufficient for output image returned by xiGetImage
    "transport_pixel_format": "xiTypeEnum",    #Current format of pixels on transport layer.
    "transport_data_target": "xiTypeEnum",    #Target selector for data - CPU RAM or GPU RAM
    "sensor_clock_freq_hz": "xiTypeFloat",    #Sensor clock frequency in Hz.
    "sensor_clock_freq_index": "xiTypeInteger",    #Sensor clock frequency index. Sensor with selected frequencies have possibility to set the frequency only by this index.
    "sensor_output_channel_count": "xiTypeEnum",    #Number of output channels from sensor used for data transfer.
    "framerate": "xiTypeFloat",    #Define framerate in Hz
    "counter_selector": "xiTypeEnum",    #Select counter
    "counter_value": "xiTypeInteger",    #Counter status
    "acq_timing_mode": "xiTypeEnum",    #Type of sensor frames timing.
    "available_bandwidth": "xiTypeInteger",    #Measure and return available interface bandwidth(int Megabits)
    "buffer_policy": "xiTypeEnum",    #Data move policy
    "LUTEnable": "xiTypeBoolean",    #Activates LUT.
    "LUTIndex": "xiTypeInteger",    #Control the index (offset) of the coefficient to access in the LUT.
    "LUTValue": "xiTypeInteger",    #Value at entry LUTIndex of the LUT
    "trigger_delay": "xiTypeInteger",    #Specifies the delay in microseconds (us) to apply after the trigger reception before activating it.
    "ts_rst_mode": "xiTypeEnum",    #Defines how time stamp reset engine will be armed
    "ts_rst_source": "xiTypeEnum",    #Defines which source will be used for timestamp reset. Writing this parameter will trigger settings of engine (arming)
    "isexist": "xiTypeBoolean",    #Returns 1 if camera connected and works properly.
    "acq_buffer_size": "xiTypeInteger",    #Acquisition buffer size in buffer_size_unit. Default bytes.
    "acq_buffer_size_unit": "xiTypeInteger",    #Acquisition buffer size unit in bytes. Default 1. E.g. Value 1024 means that buffer_size is in KiBytes
    "acq_transport_buffer_size": "xiTypeInteger",    #Acquisition transport buffer size in bytes
    "acq_transport_packet_size": "xiTypeInteger",    #Acquisition transport packet size in bytes
    "buffers_queue_size": "xiTypeInteger",    #Queue of field/frame buffers
    "acq_transport_buffer_commit": "xiTypeInteger",    #Number of buffers to commit to low level
    "recent_frame": "xiTypeBoolean",    #GetImage returns most recent frame
    "device_reset": "xiTypeCommand",    #Resets the camera to default state.
    "column_fpn_correction": "xiTypeEnum",    #Correction of column FPN
    "row_fpn_correction": "xiTypeEnum",    #Correction of row FPN
    "image_correction_selector": "xiTypeEnum",    #Select image correction function
    "image_correction_value": "xiTypeFloat",    #Select image correction selected function value
    "sensor_mode": "xiTypeEnum",    #Current sensor mode. Allows to select sensor mode by one integer. Setting of this parameter affects: image dimensions and downsampling.
    "hdr": "xiTypeBoolean",    #Enable High Dynamic Range feature.
    "hdr_kneepoint_count": "xiTypeInteger",    #The number of kneepoints in the PWLR.
    "hdr_t1": "xiTypeInteger",    #position of first kneepoint(in % of XI_PRM_EXPOSURE)
    "hdr_t2": "xiTypeInteger",    #position of second kneepoint (in % of XI_PRM_EXPOSURE)
    "hdr_kneepoint1": "xiTypeInteger",    #value of first kneepoint (% of sensor saturation)
    "hdr_kneepoint2": "xiTypeInteger",    #value of second kneepoint (% of sensor saturation)
    "image_black_level": "xiTypeInteger",    #Last image black level counts. Can be used for Offline processing to recall it.
    "api_version": "xiTypeString",    #Returns version of API.
    "drv_version": "xiTypeString",    #Returns version of current device driver.
    "version_mcu1": "xiTypeString",    #Returns version of MCU1 firmware.
    "version_mcu2": "xiTypeString",    #Returns version of MCU2 firmware.
    "version_mcu3": "xiTypeString",    #Returns version of MCU3 firmware.
    "version_fpga1": "xiTypeString",    #Returns version of FPGA1 firmware.
    "version_xmlman": "xiTypeString",    #Returns version of XML manifest.
    "hw_revision": "xiTypeString",    #Returns hardware revision number.
    "debug_level": "xiTypeEnum",    #Set debug level
    "auto_bandwidth_calculation": "xiTypeBoolean",    #Automatic bandwidth calculation,
    "new_process_chain_enable": "xiTypeBoolean",    #Enables (2015/FAPI) processing chain for MQ MU cameras
    "cam_enum_golden_enabled": "xiTypeBoolean",    #Enable enumeration of golden devices
    "reset_usb_if_bootloader": "xiTypeBoolean",    #Resets USB device if started as bootloader
    "cam_simulators_count": "xiTypeInteger",    #Number of camera simulators to be available.
    "cam_sensor_init_disabled": "xiTypeBoolean",    #Camera sensor will not be initialized when 1=XI_ON is set.
    "read_file_ffs": "xiTypeString",    #Read file from camera flash filesystem.
    "write_file_ffs": "xiTypeString",    #Write file to camera flash filesystem.
    "ffs_file_name": "xiTypeString",    #Set name of file to be written/read from camera FFS.
    "ffs_file_id": "xiTypeInteger",    #File number.
    "ffs_file_size": "xiTypeInteger",    #Size of file.
    "free_ffs_size": "xiTypeInteger",    #Size of free camera FFS.
    "used_ffs_size": "xiTypeInteger",    #Size of used camera FFS.
    "ffs_access_key": "xiTypeInteger",    #Setting of key enables file operations on some cameras.
    "xiapi_context_list": "xiTypeString",    #List of current parameters settings context - parameters with values. Used for offline processing.
    "sensor_feature_selector": "xiTypeEnum",    #Selects the current feature which is accessible by XI_PRM_SENSOR_FEATURE_VALUE.
    "sensor_feature_value": "xiTypeInteger",    #Allows access to sensor feature value currently selected by XI_PRM_SENSOR_FEATURE_SELECTOR.
    "ext_feature_selector": "xiTypeEnum",    #Selection of extended feature.
    "ext_feature": "xiTypeInteger",    #Extended feature value.
    "device_unit_selector": "xiTypeEnum",    #Selects device unit.
    "device_unit_register_selector": "xiTypeInteger",    #Selects register of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).
    "device_unit_register_value": "xiTypeInteger",    #Sets/gets register value of selected device unit(XI_PRM_DEVICE_UNIT_SELECTOR).
    "api_progress_callback": "xiTypeString",    #Callback address of pointer that is called upon long tasks (e.g. XI_PRM_WRITE_FILE_FFS).
    "acquisition_status_selector": "xiTypeEnum",    #Selects the internal acquisition signal to read using XI_PRM_ACQUISITION_STATUS.
    "acquisition_status": "xiTypeEnum",    #Acquisition status(True/False)
    }

ASSOC_ENUM = {
    "gain_selector": XI_GAIN_SELECTOR_TYPE,    #Gain selector for parameter Gain allows to select different type of gains.
    "downsampling": XI_DOWNSAMPLING_VALUE,    #Change image resolution by binning or skipping.
    "downsampling_type": XI_DOWNSAMPLING_TYPE,    #Change image downsampling type.
    "test_pattern_generator_selector": XI_TEST_PATTERN_GENERATOR,    #Selects which test pattern generator is controlled by the TestPattern feature.
    "test_pattern": XI_TEST_PATTERN,    #Selects which test pattern type is generated by the selected generator.
    "imgdataformat": XI_IMG_FORMAT,    #Output data format.
    "shutter_type": XI_SHUTTER_TYPE,    #Change sensor shutter type(CMOS sensor).
    "sensor_taps": XI_SENSOR_TAP_CNT,    #Number of taps
    "bpc_list_selector": XI_SENS_DEFFECTS_CORR_LIST_SELECTOR,    #Selector of list used by Sensor Defects Correction parameter
    "binning_selector": XI_BIN_SELECTOR,    #Binning engine selector.
    "binning_vertical_mode": XI_BIN_MODE,    #Sets the mode to use to combine vertical pixel together.
    "binning_horizontal_mode": XI_BIN_MODE,    #Sets the mode to use to combine horizontal pixel together.
    "binning_horizontal_pattern": XI_BIN_PATTERN,    #Binning horizontal pattern type.
    "binning_vertical_pattern": XI_BIN_PATTERN,    #Binning vertical pattern type.
    "decimation_selector": XI_DEC_SELECTOR,    #Decimation engine selector.
    "decimation_horizontal_pattern": XI_DEC_PATTERN,    #Decimation horizontal pattern type.
    "decimation_vertical_pattern": XI_DEC_PATTERN,    #Decimation vertical pattern type.
    "limit_bandwidth_mode": XI_SWITCH,    #Bandwidth limit enabled
    "sensor_bit_depth": XI_BIT_DEPTH,    #Sensor output data bit depth.
    "output_bit_depth": XI_BIT_DEPTH,    #Device output data bit depth.
    "image_data_bit_depth": XI_BIT_DEPTH,    #bitdepth of data returned by function xiGetImage
    "output_bit_packing_type": XI_OUTPUT_DATA_PACKING_TYPE,    #Data packing type. Some cameras supports only specific packing type.
    "cooling": XI_TEMP_CTRL_MODE_SELECTOR,    #Temperature control mode.
    "temp_selector": XI_TEMP_SELECTOR,    #Selector of mechanical point where thermometer is located.
    "device_temperature_ctrl_mode": XI_TEMP_CTRL_MODE_SELECTOR,    #Temperature control mode.
    "device_temperature_element_sel": XI_TEMP_ELEMENT_SELECTOR,    #Temperature element selector (TEC(Peltier), Fan).
    "cms": XI_CMS_MODE,    #Mode of color management system.
    "cms_intent": XI_CMS_INTENT,    #Intent of color management system.
    "cfa": XI_COLOR_FILTER_ARRAY,    #Returns color filter array type of RAW data.
    "trigger_source": XI_TRG_SOURCE,    #Defines source of trigger.
    "trigger_selector": XI_TRG_SELECTOR,    #Selects the type of trigger.
    "trigger_overlap": XI_TRG_OVERLAP,    #The mode of Trigger Overlap. This influences of trigger acception/rejection policy
    "gpi_selector": XI_GPI_SELECTOR,    #Selects GPI
    "gpi_mode": XI_GPI_MODE,    #Defines GPI functionality
    "gpo_selector": XI_GPO_SELECTOR,    #Selects GPO
    "gpo_mode": XI_GPO_MODE,    #Defines GPO functionality
    "led_selector": XI_LED_SELECTOR,    #Selects LED
    "led_mode": XI_LED_MODE,    #Defines LED functionality
    "lens_feature_selector": XI_LENS_FEATURE,    #Selects the current feature which is accessible by XI_PRM_LENS_FEATURE.
    "transport_pixel_format": XI_GenTL_Image_Format_e,    #Current format of pixels on transport layer.
    "transport_data_target": XI_TRANSPORT_DATA_TARGET_MODE,    #Target selector for data - CPU RAM or GPU RAM
    "sensor_output_channel_count": XI_SENSOR_OUTPUT_CHANNEL_COUNT,    #Number of output channels from sensor used for data transfer.
    "counter_selector": XI_COUNTER_SELECTOR,    #Select counter
    "acq_timing_mode": XI_ACQ_TIMING_MODE,    #Type of sensor frames timing.
    "buffer_policy": XI_BP,    #Data move policy
    "ts_rst_mode": XI_TS_RST_MODE,    #Defines how time stamp reset engine will be armed
    "ts_rst_source": XI_TS_RST_SOURCE,    #Defines which source will be used for timestamp reset. Writing this parameter will trigger settings of engine (arming)
    "column_fpn_correction": XI_SWITCH,    #Correction of column FPN
    "row_fpn_correction": XI_SWITCH,    #Correction of row FPN
    "image_correction_selector": XI_IMAGE_CORRECTION_SELECTOR,    #Select image correction function
    "sensor_mode": XI_SENSOR_MODE,    #Current sensor mode. Allows to select sensor mode by one integer. Setting of this parameter affects: image dimensions and downsampling.
    "debug_level": XI_DEBUG_LEVEL,    #Set debug level
    "sensor_feature_selector": XI_SENSOR_FEATURE_SELECTOR,    #Selects the current feature which is accessible by XI_PRM_SENSOR_FEATURE_VALUE.
    "ext_feature_selector": XI_EXT_FEATURE_SELECTOR,    #Selection of extended feature.
    "device_unit_selector": XI_DEVICE_UNIT_SELECTOR,    #Selects device unit.
    "acquisition_status_selector": XI_ACQUISITION_STATUS_SELECTOR,    #Selects the internal acquisition signal to read using XI_PRM_ACQUISITION_STATUS.
    "acquisition_status": XI_SWITCH,    #Acquisition status(True/False)
    }

# Structures

class XI_IMG_DESC(Structure):
    '''
    structure containing description of image areas and format.
    '''
    _fields_ = [
        ("Area0Left",    DWORD),        #Pixels of Area0 of image left.
        ("Area1Left",    DWORD),        #Pixels of Area1 of image left.
        ("Area2Left",    DWORD),        #Pixels of Area2 of image left.
        ("Area3Left",    DWORD),        #Pixels of Area3 of image left.
        ("Area4Left",    DWORD),        #Pixels of Area4 of image left.
        ("Area5Left",    DWORD),        #Pixels of Area5 of image left.
        ("ActiveAreaWidth",    DWORD),        #Width of active area.
        ("Area5Right",    DWORD),        #Pixels of Area5 of image right.
        ("Area4Right",    DWORD),        #Pixels of Area4 of image right.
        ("Area3Right",    DWORD),        #Pixels of Area3 of image right.
        ("Area2Right",    DWORD),        #Pixels of Area2 of image right.
        ("Area1Right",    DWORD),        #Pixels of Area1 of image right.
        ("Area0Right",    DWORD),        #Pixels of Area0 of image right.
        ("Area0Top",    DWORD),        #Pixels of Area0 of image top.
        ("Area1Top",    DWORD),        #Pixels of Area1 of image top.
        ("Area2Top",    DWORD),        #Pixels of Area2 of image top.
        ("Area3Top",    DWORD),        #Pixels of Area3 of image top.
        ("Area4Top",    DWORD),        #Pixels of Area4 of image top.
        ("Area5Top",    DWORD),        #Pixels of Area5 of image top.
        ("ActiveAreaHeight",    DWORD),        #Height of active area.
        ("Area5Bottom",    DWORD),        #Pixels of Area5 of image bottom.
        ("Area4Bottom",    DWORD),        #Pixels of Area4 of image bottom.
        ("Area3Bottom",    DWORD),        #Pixels of Area3 of image bottom.
        ("Area2Bottom",    DWORD),        #Pixels of Area2 of image bottom.
        ("Area1Bottom",    DWORD),        #Pixels of Area1 of image bottom.
        ("Area0Bottom",    DWORD),        #Pixels of Area0 of image bottom.
        ("format",    DWORD),        #Current format of pixels. XI_GenTL_Image_Format_e.
        ("flags",    DWORD),        #description of areas and image.
        ]

class XI_IMG(Structure):
    '''
    structure containing information about incoming image.
    '''
    _fields_ = [
        ("size",    DWORD),        #Size of current structure on application side. When xiGetImage is called and size>=SIZE_XI_IMG_V2 then GPI_level, tsSec and tsUSec are filled.
        ("bp",    LPVOID),        #Pointer to data. In XI_BP_UNSAFE mode the bp will be set to buffer allocated by API. If XI_BP_SAFE mode the data will be copied to bp, which should be allocated by application.
        ("bp_size",    DWORD),        #Filled buffer size. When buffer policy is set to XI_BP_SAFE, xiGetImage will fill this field with current size of image data received.
        ("frm",    c_uint),        #Format of image data get from xiGetImage.
        ("width",    DWORD),        #width of incoming image.
        ("height",    DWORD),        #height of incoming image.
        ("nframe",    DWORD),        #Frame number. On some cameras it is reset by exposure, gain, downsampling change, auto exposure (AEAG).
        ("tsSec",    DWORD),        #Seconds part of timestamp delivered by camera (at start of read-out phase). Typical range: 0-4294 sec.
        ("tsUSec",    DWORD),        #Micro-seconds part of timestamp delivered by camera (at start of read-out phase). Range 0-999999 us.
        ("GPI_level",    DWORD),        #Levels of digital inputs/outputs of the camera at time of exposure start/end (sample time and bits are specific for each camera model)
        ("black_level",    DWORD),        #Black level of image (ONLY for MONO and RAW formats)
        ("padding_x",    DWORD),        #Number of extra bytes provided at the end of each line to facilitate image alignment in buffers.
        ("AbsoluteOffsetX",    DWORD),        #Horizontal offset of origin of sensor and buffer image first pixel.
        ("AbsoluteOffsetY",    DWORD),        #Vertical offset of origin of sensor and buffer image first pixel.
        ("transport_frm",    DWORD),        #Current format of pixels on transport layer. XI_GenTL_Image_Format_e.
        ("img_desc",    XI_IMG_DESC),        #description of image areas and format.
        ("DownsamplingX",    DWORD),        #Horizontal downsampling
        ("DownsamplingY",    DWORD),        #Vertical downsampling
        ("flags",    DWORD),        #description of XI_IMG.
        ("exposure_time_us",    DWORD),        #Exposure time of this image in microseconds
        ("gain_db",    FLOAT),        #Gain used for this image in deci-bells
        ("acq_nframe",    DWORD),        #Frame number. Reset only by acquisition start. NOT reset by change of exposure, gain, downsampling, auto exposure (AEAG).
        ("image_user_data",    DWORD),        #ImageUserData controlled by user application using ImageUserData or XI_PRM_IMAGE_USER_DATA parameter
        ("exposure_sub_times_us[5]",    DWORD),        #Array with five sub exposures times in microseconds used by XI_TRG_SEL_MULTIPLE_EXPOSURES or hardware controlled HDR
        ]


