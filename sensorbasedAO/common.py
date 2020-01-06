import collections
import numpy as np
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
            SB_pix_coord_x = np.arange(int(settings['act_ref_cent_coord_x'][i]) - settings['SB_rad'] + 1, \
                int(settings['act_ref_cent_coord_x'][i]) + settings['SB_rad'] - 1)
            SB_pix_coord_y = np.arange(int(settings['act_ref_cent_coord_y'][i]) - settings['SB_rad'] + 1, \
                int(settings['act_ref_cent_coord_y'][i]) + settings['SB_rad'] - 1)
        else:
            SB_pix_coord_x = np.arange(int(settings['act_ref_cent_coord_x'][i]) - settings['SB_rad'] + 2, \
                int(settings['act_ref_cent_coord_x'][i]) + settings['SB_rad'] - 1)
            SB_pix_coord_y = np.arange(int(settings['act_ref_cent_coord_y'][i]) - settings['SB_rad'] + 2, \
                int(settings['act_ref_cent_coord_y'][i]) + settings['SB_rad'] - 1)

        # Initialise instance variables for calculating wavefront tilt within each search block
        a_x, a_y = (np.zeros(len(SB_pix_coord_x)) for i in range(2))

        # Get wavefront tilt of each row and column within each search block
        for j in range(len(SB_pix_coord_x)):
            a_x[j] = np.polyfit(SB_pix_coord_x, phase[int(round(SB_pix_coord_y[j])), int(round(SB_pix_coord_x[0])) : \
                int(round(SB_pix_coord_x[0])) + len(SB_pix_coord_x)], 1)[0] 
            a_y[j] = np.polyfit(SB_pix_coord_y, phase[int(round(SB_pix_coord_y[0])) : int(round(SB_pix_coord_y[0])) + \
                len(SB_pix_coord_y), int(round(SB_pix_coord_x[j]))], 1)[0] 

        # Calculate average wavefront tilt within each search block
        a_x_ave = -np.mean(a_x) / settings['pixel_size']
        a_y_ave = -np.mean(a_y) / settings['pixel_size']

        # Calculate S-H spot centroid position along x and y axis
        slope_x[i] = a_x_ave * config['lenslet']['lenslet_focal_length'] / settings['pixel_size']
        slope_y[i] = a_y_ave * config['lenslet']['lenslet_focal_length'] / settings['pixel_size']

    return slope_x, slope_y
