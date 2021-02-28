import nifpga
import os.path
import numpy as np
import sys
import time
from doptical.log import get_logger

logger = get_logger(__name__)

class FPGA():
    """
    FPGA factory class

    returns FPGA instances of appropriate type.
    """
    @staticmethod
    def get(bitfile=None, fpga_type='NI', **kwargs):
        if fpga_type.lower() == 'ni':
            try:
                nifpga.nifpga._NiFpga()
                fpga = FPGA_NI(bitfile, **kwargs)
            except nifpga.InvalidResourceNameError as e:
                raise RuntimeError('NI FPGA device not found') from e
            except nifpga.LibraryNotFoundError as e:
                raise RuntimeError('NI driver library for FPGA not found') from e

            return fpga
        elif fpga_type.lower() == 'debug':
            return FPGA_NI_dummy(bitfile)

# Converstion between NI and numpy data-types
NI_DATATYPE = {
    'I8': 'int8',
    'I16': 'int16',
    'I32': 'int32',
    'I64': 'int64',
    'U8': 'uint8',
    'U16': 'uint16',
    'U32': 'uint32',
    'U64': 'uint64',
    'FXP': 'double'     # Gets converted to Decimal by NI lib
}

class FPGA_NI():
    """
    FPGA_NI FPGA interface class

    TODO:
        - Handle register and FIFO timeouts

    """
    def __init__(self, bitfile, resource="RIO0", data_format = 'FXP'):
        self.resource = resource
        self.session = None
        self.tick_rate = 40e6
        self.load(bitfile)

        # Set data format for analog inputs/outputs
        self.data_format = data_format.upper()
        assert data_format in ['FXP', 'I16']

        if self.data_format == 'FXP':
            self.data_min = -10
            self.data_max = 10
        elif self.data_format == 'I16':
            self.data_min = 0
            self.data_max = 2**16
            

    def __del__(self):
        if self is not None:
            try:
                self.session.close()
            except:
                pass

    def load(self, bitfile):
        self.bitfile = bitfile

        if self.session is None:
            self.session = nifpga.Session(bitfile=self.bitfile,
                                   resource=self.resource)

    def run(self):
        if self.session is not None:
            self.session.run()

    def reset(self):
        self.session.reset()

    def close(self):
        if self.session is not None:
            self.session.close()
            self.session = None

    def read(self, control):
        return self.session.registers[control].read()

    def read_all(self):
        r = []
        for register in self.session.registers:
            r.append((register,self.session.registers[register].read()))

        return r

    def write(self, control, value):
        self.session.registers[control].write(value)

    def write_fifo(self, fifo, data, timeout=0):
        elements_remaining = self.session.fifos[fifo].write(data, timeout_ms=timeout)
        return elements_remaining

    def read_fifo_raw(self, fifo, length, timeout=0):
        d = self.session.fifos[fifo].read(length, timeout_ms=timeout)
        return d.data, d.elements_remaining

    def read_fifo(self, fifo, length=None, timeout=0, chunk=None, dtype=None):
        """
        Read FIFO data

        inputs:
            - length:  length of data to read. Reads all available data if None
            - timeout: timeout before throwing timeout error
            - chunk: size of data chunks to request from FIFO
            - dtype: datatype to return. If None, set from FIFO

        returns:
            - Return array of data, formatted according to FIFO datatype, or optional dtype provided

        TODO: 
            - handle timeout
            - handle buffer underflow/overflow
        """

        # Initialise empty data list
        data = []

        # Handle length = None
        # Will return all FIFO elements, but set length to 1 for now
        if length is None:
            try:
                data_fifo = self.session.fifos[fifo].read(toread, timeout_ms=timeout)
                length = data_fifo.elements_remaining
            except Exception as e:
                length = 0

        # Return None if no data to read
        if not length:
            return None

        # Set required data length to read, chunked, if requested
        if chunk is None:
            toread = length
        else:
            toread = min(length, chunk)

        # Read data from FIFO in a loop until required data acquired
        read = 0
        keepreading = True

        ts_begin = time.perf_counter()

        while keepreading:
            try:
                (_, data_available) =  self.session.fifos[fifo].read(0, timeout_ms=0)
                # print('data_available:', data_available)
                # print('toread:', toread)
                if data_available >= toread:
                    # Get data from FIFO and append to data list
                    data_fifo = self.session.fifos[fifo].read(toread, timeout_ms=timeout)
                    
                    # Append returned data to array
                    data = data + data_fifo.data

                    # Increment read counter
                    read += toread

                    # Break read loop if required data acquired
                    if read == length or (data.elements_remaining == 0):
                        keepreading = False
                        continue

                    # Chunk data if requested
                    if chunk is None:
                        toread = length-read
                    else:
                        toread = min((length-read), chunk)

            except Exception as e:
                # Handle timeout
                print(e)
                raise
        
            # Handle timeout
            ts_elapsed = time.perf_counter() - ts_begin
            if ts_elapsed > timeout:
                return None

        # Get FIFO datatype from bitfile, if not explicitly set
        if not dtype:
            dtype = self.session.fifos[fifo].datatype
            dtype = NI_DATATYPE.get(dtype.name.upper(),None)

        # Format numpy array with correct dtype and return
        data = np.array(data,dtype=dtype)
        return data

    def clear_fifo(self, fifo=None):
        if fifo:
            fifos = [fifo]
        else:
            fifos = self.session.fifos

        for fifo in fifos:
            fifo = self.session.fifos[fifo]
            fifo.stop()
            fifo.start()

    def list_registers(self):
        registers = []
        for register_name in self.session.registers:
            register = self.session.registers[register_name]
            registers.append((register.name, str(register.datatype)))
        logger.info("Registers: " + str(registers))
        return registers

    def list_fifos(self):
        print('FIFOs:')
        for fifo_name in self.session.fifos:
            fifo = self.session.fifos[fifo_name]
            print("\t{}, {}".format(fifo.name, fifo.datatype))

    def list_inputs(self):
        pass

    def list_outputs(self):
        pass

    def wait_for_value(self, register, target_value, delay=0, timeout=2):
        # Wait for FPGA value
        time_start = time.perf_counter()
        while self.read(register) is not target_value:
            # Throw error if elapsed time greater than timeout
            time_elapsed = time.perf_counter() - time_start
            time.sleep(delay)
            if time_elapsed > timeout:
                raise RuntimeError("A timeout occured while waiting for {} to equal {}.".format(register, target_value))

    def to_fpga_voltage(self, x):
        # TODO take both array and number input
        x = np.clip(x, self.data_min, self.data_max)

        if self.data_format == 'I16':
            v = x/20*2**16
            v = v.astype(NI_DATATYPE['I16'])
        else:
            v = x.astype(NI_DATATYPE['FXP'])

        return v

    def from_fpga_voltage(self, x):
        if self.data_format == 'I16':
            y = x * 20/2**16
        else:
            y = x

        return y

    def to_ticks(self, x, clock_rate=40e6):
        # x = np.clip(x,-10,10)
        y = int(clock_rate/x)
        return int(y)

    def from_ticks(self, x, clock_rate=40e6):
        try:
            y = clock_rate/x
            return y
        except ZeroDivisionError:
            return 0

class Session_dummy():
    def __init__(self, **kwargs):
        logger.info('Dummy FPGA session started')
        self.registers = {}

    def __getattr__(self,name):
        def handler(*args,**kwargs):
            logger.info("*** Session_dummy *** name: {}; args: {}; kwargs: {}".format(name,args,kwargs))
        return handler

    def close(self):
        logger.info('Dummy FPGA session closed')
        pass

class FPGA_NI_dummy():
    def __init__(self, bitfile=None, data_format='FXP'):
        logger.info('Dummy FPGA loaded')
        self.resource = "RIO0"
        self.session = None
        self.tick_rate = 40e6
        self.log_toggle = False

        # Set data format for analog inputs/outputs
        self.data_format = data_format.upper()
        assert data_format in ['FXP', 'I16']

        if self.data_format == 'FXP':
            self.data_min = -10
            self.data_max = 10
        elif self.data_format == 'I16':
            self.data_min = 0
            self.data_max = 2**16

        if bitfile is not None:
            self.load(bitfile)

    def __del__(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        logger.info('Closed FPGA')

    def __getattr__(self,name):
        try:
            def handler(*args,**kwargs):
                self.log("name: {}; args: {}; kwargs: {}".format(name,args,kwargs))
            return handler
        except Exception as e:
            logger.warn('Error in __getattr__', e)
            pass

    def log_enable(self, value):
        self.log('Set logging: {}'.format(value))
        self.log_toggle = value

    def log(self, msg):
        if self.log_toggle:
            logger.info('*** FPGA_dummy *** ' + msg)

    def load(self, bitfile):
        self.log("name: {}; args: {}; kwargs: {}".format('load',bitfile,None))
        self.bitfile = bitfile

        if self.session is None:
            self.session = Session_dummy(bitfile=self.bitfile,
                                   resource=self.resource)

    def read(self, *args, **kwargs):
        self.log("name: {}; args: {}; kwargs: {}".format('read', args, kwargs))
        register = args[0]
        if register == 'fpga state':
            return 0
        else:
            return 0

    def read_fifo(self, name, length, **kwargs):
        time.sleep(0.001)
        if self.data_format == 'I16':
            data = np.random.randint(2**16, size=length, dtype='uint16')
        else:
            data = np.random.rand(length) * (self.data_max - self.data_min) + self.data_min
        
        return data

    def read_fifo_raw(self, fifo, length, timeout=10):
        if self.data_format == 'I16':
            data = np.random.randint(2**16, size=length, dtype='uint16')
        else:
            data = np.random.rand(length) * (self.data_max - self.data_min) + self.data_min

        self.remaining = length
        self.remaining = self.remaining - length
        return data, self.remaining

    def to_fpga_voltage(self, x):
        # TODO take both array and number input
        x = np.clip(x, self.data_min, self.data_max)

        if self.data_format == 'I16':
            v = x/20*2**16
            v = v.astype(NI_DATATYPE['I16'])
        else:
            v = x.astype(NI_DATATYPE['FXP'])

        return v

    def from_fpga_voltage(self, x):
        if self.data_format== 'I16':
            y = x * 20/2**16
        else:
            y = x

        return y

    def to_ticks(self, x, clock_rate=40e6):
        # x = np.clip(x,-10,10)
        y = int(clock_rate/x)
        return int(y)

    def from_ticks(self, x, clock_rate=40e6):
        try:
            y = clock_rate/x
            return y
        except ZeroDivisionError:
            return 0
