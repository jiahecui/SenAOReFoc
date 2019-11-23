import numpy as np
import PIL.Image
import random

class SpotSim():
    """
    Generates a simulated S-H spot image
    """
    def __init__(self, settings):
        
        # Get sensor and search block parameters
        self.settings = settings

    def SH_spot_sim(self):
        """
        Integrates the same number and geometry of Gaussian profile S-H spots as search blocks into one image
        """
        # Initialise spot image array
        self.SH_spot_img = np.zeros([self.settings['sensor_width'], self.settings['sensor_height']])

        # Use actual search block reference centroid coords as upper left corner position for tiling of individual S-H spots
        offset_coord = self.settings['act_ref_cent_coord']

        # Get other non-variable parameters for generation of a Gaussian profile S-H spot
        sigma = 0.1
        array_size = self.settings['SB_diam']
        x = np.linspace(-1, 1, array_size)
        y = np.linspace(-1, 1, array_size)

        # Generate individual Gaussian profile S-H spots and integrate them to pre-initialised spot image array
        for i in range(len(offset_coord)):
  
            # Generate random positions for centre of each S-H spot
            xc = round(random.uniform(-0.5, 0.5), 2)
            yc = round(random.uniform(-0.5, 0.5), 2)

            # Generate a Gaussian profile S-H spot
            Gaus_spot = self.dirac_function(x, y, xc, yc, sigma, array_size)

            # Integrate it to pre-initialised spot image array
            A_start = [0, 0]
            B_start = [offset_coord[i] // self.settings['sensor_width'], offset_coord[i] % self.settings['sensor_width']]
            B_end = [offset_coord[i] // self.settings['sensor_width'] + array_size, offset_coord[i] % self.settings['sensor_width'] + array_size]

            self.SH_spot_img = self.array_integrate(Gaus_spot, self.SH_spot_img, A_start, B_start, B_end)

        return self.SH_spot_img

    def dirac_function(self, x, y, xc, yc, sigma, size):
        """
        Generates a 2D Gaussian profile array

        Args:
            x: 1D vector of positions along x axis
            y: 1D vector of positions along y axis
            xc: centre of profile along x axis
            yc: centre of profile along y axis
            sigma: standard deviation of Gaussian profile
            size: size of Gaussian profile array
        """

        # Initialise Gaussian profile array
        GaussProf = np.zeros([size, size])

        # Generate Gaussian profile
        for i in range(size):
            for j in range(size):
                GaussValue = (255 * np.exp( - ((x[j] - xc) ** 2 + (y[i] - yc) ** 2) / (2 *sigma ** 2))).astype(np.uint8)
                GaussProf[j, i] = GaussValue

        return GaussProf

    def array_integrate(self, A, B, A_start, B_start, B_end):
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