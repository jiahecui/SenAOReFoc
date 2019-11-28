import numpy as np
import PIL.Image
import random

from config import config

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
        # Use actual search block reference centroid coords as upper left corner position for tiling of individual S-H spots
        offset_coord = self.settings['act_ref_cent_coord']
        offset_coord_x = self.settings['act_ref_cent_coord_x']
        offset_coord_y = self.settings['act_ref_cent_coord_y']

        # Initialise spot image array and S-H spot centre coords array
        self.SH_spot_img = np.zeros([self.settings['sensor_width'], self.settings['sensor_height']])
        self.spot_cent = np.zeros(len(offset_coord))
        self.spot_cent_x = np.zeros(len(offset_coord))
        self.spot_cent_y = np.zeros(len(offset_coord))

        # Get other non-variable parameters for generation of a Gaussian profile S-H spot
        sigma = 4
        array_size = self.settings['SB_diam']
        x = np.linspace(-int(self.settings['SB_rad']), int(self.settings['SB_rad']), array_size)
        y = np.linspace(-int(self.settings['SB_rad']), int(self.settings['SB_rad']), array_size)

        # print('x: {}, y: {}, length: {}'.format(x, y, len(x)))

        # Generate individual Gaussian profile S-H spots and integrate them to pre-initialised spot image array
        for i in range(len(offset_coord)):
  
            # Generate random positions for centre of each S-H spot
            # xc = random.randrange(-5, 6, 1)
            # yc = random.randrange(-5, 6, 1)
            xc = round(random.uniform(-5, 6), 1)
            yc = round(random.uniform(-5, 6), 1)
            # xc = 0
            # yc = 0

            # print('xc: {}, yc: {}'.format(xc, yc))

            # Generate a Gaussian profile S-H spot
            Gaus_spot = self.dirac_function(x, y, xc, yc, sigma, array_size)

            # Calculate centre of S-H spot
            self.spot_cent_x[i] = int(offset_coord_x[i]) + int(self.settings['SB_rad']) + xc
            self.spot_cent_y[i] = int(offset_coord_y[i]) + int(self.settings['SB_rad']) + yc
            self.spot_cent[i] = int(self.spot_cent_y[i]) * self.settings['sensor_width'] + int(self.spot_cent_x[i])

            # Integrate it to pre-initialised spot image array
            A_start = [0, 0]
            B_start = [int(offset_coord_y[i]), int(offset_coord_x[i])]
            B_end = [int(offset_coord_y[i] + array_size), int(offset_coord_x[i] + array_size)]

            self.SH_spot_img = self.array_integrate(Gaus_spot, self.SH_spot_img, A_start, B_start, B_end)

        # Add noise to S-H spot image
        self.SH_spot_img = self.add_noise(self.SH_spot_img, noise_type = 'uniform')
            
        # print('Theoretical S-H spot cent:', self.spot_cent)
        # print('Theoretical S-H spot cent x:', self.spot_cent_x)
        # print('Theoretical S-H spot cent y:', self.spot_cent_y)

        return self.SH_spot_img, self.spot_cent_x, self.spot_cent_y

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
                GaussValue = (255 * np.exp( - ((x[j] - xc) ** 2 + (y[i] - yc) ** 2) / (2 * sigma ** 2))).astype(np.uint8)
                GaussProf[i, j] = GaussValue

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

    def add_noise(self, image, noise_type = 'uniform'):
        """
        Adds a particular type of noise to image

        Types of noise:
            'uniform'   Uniform noise across the whole image
            'gauss'     Gaussian-distributed additive noise.
            'poisson'   Poisson-distributed noise generated from the data.
            'speckle'   Multiplicative noise using out = image + n * image, where
                        n is uniform noise with specified mean & variance.
        """
        # 1) Add uniform noise across the whole image
        if noise_type == 'uniform':

            noisy = image + config['image']['noise']

            return noisy

        # 2) Add Gaussian-distributed additive noise
        elif noise_type == 'gauss':

            shape = image.shape
            mean = 0
            var = 0.1
            sigma = var ** 0.5
            gauss = np.random.normal(mean, sigma, shape)
            gauss = gauss.reshape(shape[0], shape[1])
            noisy = image + gauss

            return noisy
        
        # 3) Add Poisson-distributed noise generated from the data
        elif noise_type == 'poisson':

            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)

            return noisy

        # 4) Add multiplicative speckle noise
        elif noise_type == "speckle":

            shape = image.shape
            gauss = np.random.randn(shape[0], shape[1])
            gauss = gauss.reshape(shape[0], shape[1])        
            noisy = image + image * gauss
 
            return noisy