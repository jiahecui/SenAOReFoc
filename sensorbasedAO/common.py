import collections
import numpy as np
import math
from config import config

class Stack:
    def __init__(self, items=None, max_length=None):
        self._stack = collections.deque(maxlen=max_length)
        
        if items is not None:
            for item in items:
                self._stack.append(item)


    def __str__(self):
        return str(self._stack)

    def __getitem__(self, indices):
        return self._stack.__getitem__(indices)

    def __setitem__(self, key, value):
        self._stack.__setitem__(key, value)

    def __iter__(self):
        return self._stack.__iter__()    
    
    def __len__(self):
        return self._stack.__len__()

    def pop(self):
        return self._stack.pop()

    def push(self, item):
        self._stack.append(item)

if __name__ == '__main__':
    pass

def get_slope_from_phase(settings, phase):
    """
    Retrieves S-H spot slope values from given phase image
    """
    # Initialise array to store S-H spot centroid position within each search block
    slope_x, slope_y = (np.zeros(settings['act_ref_cent_num']) for i in range(2))

    # Get S-H spot centroid position within each search block
    for i in range(settings['act_ref_cent_num']):

        # Get 2D coords of pixels in each search block that need to be summed
        if settings['odd_pix']:
            SB_pix_coord_x = np.arange(math.ceil(math.ceil(settings['act_ref_cent_coord_x'][i]) - settings['SB_rad']) + 1, \
                (math.ceil(math.ceil(settings['act_ref_cent_coord_x'][i]) - settings['SB_rad']) + 1 + settings['SB_diam'] - 2))
            SB_pix_coord_y = np.arange(math.ceil(math.ceil(settings['act_ref_cent_coord_y'][i]) - settings['SB_rad']) + 1, \
                (math.ceil(math.ceil(settings['act_ref_cent_coord_y'][i]) - settings['SB_rad']) + 1 + settings['SB_diam'] - 2))
        else:
            SB_pix_coord_x = np.arange(math.ceil(math.ceil(settings['act_ref_cent_coord_x'][i]) - settings['SB_rad']) + 2, \
                (math.ceil(math.ceil(settings['act_ref_cent_coord_x'][i]) - settings['SB_rad']) + 2 + settings['SB_diam'] - 3))
            SB_pix_coord_y = np.arange(math.ceil(math.ceil(settings['act_ref_cent_coord_y'][i]) - settings['SB_rad']) + 2, \
                (math.ceil(math.ceil(settings['act_ref_cent_coord_y'][i]) - settings['SB_rad']) + 2 + settings['SB_diam'] - 3))

        # Initialise instance variables for calculating wavefront tilt within each search block
        a_x, a_y = (np.zeros(len(SB_pix_coord_x)) for i in range(2))

        # Get wavefront tilt of each row and column within each search block
        for j in range(len(SB_pix_coord_x)):
            a_x[j] = np.polyfit(SB_pix_coord_x, phase[math.ceil(SB_pix_coord_y[j]), math.ceil(SB_pix_coord_x[0]) : \
                math.ceil(SB_pix_coord_x[0]) + len(SB_pix_coord_x)], 1)[0] 
            a_y[j] = np.polyfit(SB_pix_coord_y, phase[math.ceil(SB_pix_coord_y[0]) : math.ceil(SB_pix_coord_y[0]) + \
                len(SB_pix_coord_y), math.ceil(SB_pix_coord_x[j])], 1)[0] 

        # Calculate average wavefront tilt within each search block
        a_x_ave = -np.mean(a_x) #/ settings['pixel_size']
        a_y_ave = -np.mean(a_y) #/ settings['pixel_size']

        # Calculate S-H spot centroid position along x and y axis
        slope_x[i] = a_x_ave / settings['pixel_size'] * config['lenslet']['lenslet_focal_length'] 
        slope_y[i] = a_y_ave / settings['pixel_size'] * config['lenslet']['lenslet_focal_length'] 

    return slope_x, slope_y

def fft_spot_from_phase(settings, phase):
    """
    Performs Fourier transform of given phase image to retrieve S-H spots within each search block
    """
    def array_integrate(A, B, A_start, B_start, B_end):
        """
        Integrates array A into array B and returns array B

        Args:
            A as numpy array
            B as numpy array
            A_start: index with respect to A of the upper left corner of the overlap
            B_start: index with respect to B of the upper left corner of the overlap
            B_end：index with respect to B of the lower right corner of the overlap
        """

        # Convert coordinate list to 1D array to facilitate calculation
        A_start, B_start, B_end = map(np.asarray, [A_start, B_start, B_end])
 
        # Get shape of overlapping part
        shape = B_end - B_start

        # Get slice coords of overlapping part relative to both A and B, i.e. B_slices returns
        # (slice(6, 16, None), slice(0, 10, None)) and A_slices returns (slice(11, 21, None), slice(5, 15, None)) \
        # means overlapping part stretches from (6, 0) to (16, 10) in B and from (11, 5) and (21, 15) in A
        B_slices = tuple(map(slice, B_start, B_end))   
        A_slices = tuple(map(slice, A_start, A_start + shape))

        # Replace corresponding elements in B with that in A
        B[B_slices] = A[A_slices]

        return B

    # Initialise spot image array and get the size of one search block
    SH_spot_img = np.zeros([settings['sensor_width'], settings['sensor_height']])
    array_size = settings['SB_diam']

    # Use actual search block reference centroid coords centre position for tiling of individual S-H spots
    offset_coord = settings['act_ref_cent_coord'] - int(settings['SB_rad']) * \
        settings['sensor_width'] - int(settings['SB_rad'])
    offset_coord_x = settings['act_ref_cent_coord_x'].astype(int) + 1 - settings['SB_rad']
    offset_coord_y = settings['act_ref_cent_coord_y'].astype(int) + 1 - settings['SB_rad']

    # Retrieve S-H spot profiles within each search block
    for i in range(settings['act_ref_cent_num']):

        # Get 2D coords of pixels in each search block
        if settings['odd_pix']:
            SB_pix_coord_x = np.arange(math.ceil(math.ceil(settings['act_ref_cent_coord_x'][i]) - settings['SB_rad']), \
                (math.ceil(math.ceil(settings['act_ref_cent_coord_x'][i]) - settings['SB_rad']) + 1 + settings['SB_diam'] - 1))
            SB_pix_coord_y = np.arange(math.ceil(math.ceil(settings['act_ref_cent_coord_y'][i]) - settings['SB_rad']), \
                (math.ceil(math.ceil(settings['act_ref_cent_coord_y'][i]) - settings['SB_rad']) + 1 + settings['SB_diam'] - 1))
        else:
            SB_pix_coord_x = np.arange(math.ceil(math.ceil(settings['act_ref_cent_coord_x'][i]) - settings['SB_rad']) + 1, \
                (math.ceil(math.ceil(settings['act_ref_cent_coord_x'][i]) - settings['SB_rad']) + 2 + settings['SB_diam'] - 2))
            SB_pix_coord_y = np.arange(math.ceil(math.ceil(settings['act_ref_cent_coord_y'][i]) - settings['SB_rad']) + 1, \
                (math.ceil(math.ceil(settings['act_ref_cent_coord_y'][i]) - settings['SB_rad']) + 2 + settings['SB_diam'] - 2))

        # Crop image within each search area
        image_crop = phase[SB_pix_coord_y[0] : SB_pix_coord_y[0] + len(SB_pix_coord_y), \
            SB_pix_coord_x[0] : SB_pix_coord_x[0] + len(SB_pix_coord_x)]

        # Perform Fourier transform within search area
        spot_crop = (np.abs(np.fft.fftshift(np.fft.fft2(image_crop)))) ** 2

        # Integrate it to pre-initialised spot image array
        A_start = [0, 0]
        B_start = [math.ceil(offset_coord_y[i]), math.ceil(offset_coord_x[i])]
        B_end = [math.ceil(offset_coord_y[i] + array_size), math.ceil(offset_coord_x[i] + array_size)]
    
        SH_spot_img = array_integrate(spot_crop, SH_spot_img, A_start, B_start, B_end)

    return SH_spot_img        