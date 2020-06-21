import numpy as np
import scipy as sp
from scipy import signal
from scipy import ndimage
from scipy import io
from scipy.ndimage import gaussian_filter

from config import config
from reflectance_profile import get_samp_sim

def reflect_process(settings, phase, pupil_diam, scan_num_x = None, scan_num_y = None):
    """
    Simulates the process of a beam reflecting off a specimen then arriving at the SHWS
    """
    def rebin(arr, new_shape):
        """
        Function to bin a 2D array to the shape specified by new_shape
        """
        shape = (new_shape[0], arr.shape[0] // new_shape[0], new_shape[1], arr.shape[1] // new_shape[1])

        return arr.reshape(shape).mean(-1).mean(1)

    def add_noise(image, noise_type = 'uniform'):
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
        elif noise_type == 'speckle':

            shape = image.shape
            gauss = np.random.randn(shape[0], shape[1])
            gauss = gauss.reshape(shape[0], shape[1])        
            noisy = image + image * gauss
 
            return noisy

    try:
        # Generate boolean phase mask
        phase_rad = int(pupil_diam * 1e3 / settings['pixel_size']) // 2
        coord_x, coord_y = (np.arange(int(-settings['sensor_width'] / 2), int(-settings['sensor_width'] / 2) + \
            settings['sensor_width']) for i in range(2))
        coord_xx, coord_yy = np.meshgrid(coord_x, coord_y)
        phase_mask = np.sqrt(coord_xx ** 2 + coord_yy ** 2) < phase_rad

        # Get detection path phase by flipping original phase left/right up/down
        phase_out = np.flipud(np.fliplr(phase))

        # Get pupil function and detection path pupil function from phase aberrations and multiply with phase mask
        pupil_func = np.exp(-2 * np.pi * 1j / config['AO']['lambda'] * phase) * phase_mask
        pupil_func_out = np.exp(-2 * np.pi * 1j / config['AO']['lambda'] * phase_out) * phase_mask

        # Bin both pupil functions to the size of object space grid
        pupil_func_binned = rebin(pupil_func, (config['reflect_prof']['obj_grid_size'], \
            config['reflect_prof']['obj_grid_size']))
        pupil_func_out_binned = rebin(pupil_func_out, (config['reflect_prof']['obj_grid_size'], \
            config['reflect_prof']['obj_grid_size']))

        # Pad pupil function with zeros before Fourier transform
        pupil_func_pad = np.pad(pupil_func_binned, (np.shape(pupil_func_binned)[0] // 2, \
            np.shape(pupil_func_binned)[0] // 2), 'constant', constant_values = (0, 0))

        # Perform Fourier transform and shift zero frequency components to centre to get amplitude PSF
        amp_PSF = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_func_pad)))

        # Normalise amplitude PSF
        # amp_PSF = amp_PSF / 100
        # phase_det = np.abs(amp_PSF) ** 2
        amp_PSF = amp_PSF / np.amax(amp_PSF)

        # Crop amplitude PSF to convolve with sample reflectance profile
        start_1 = (np.shape(amp_PSF)[0] - config['reflect_prof']['obj_grid_size']) // 2
        amp_PSF_crop = amp_PSF[start_1 : start_1 + config['reflect_prof']['obj_grid_size'], \
            start_1 : start_1 + config['reflect_prof']['obj_grid_size']]

        # sp.io.savemat('data/amp_PSF_crop/amp_PSF_crop_' + str(scan_num_y * config['zern_test']['scan_num_x'] + scan_num_x)\
        #      + '.mat', dict(amp_PSF_crop = amp_PSF_crop))

        # Generate sample reflectance profile
        samp_prof_crop = get_samp_sim(config['reflect_prof']['samp_num'])

        # Check whether noise is to be added to the synthetic images
        if config['noise_on']:
            samp_prof_crop = add_noise(samp_prof_crop, noise_type = 'uniform')

        # Generate reflection amplitude PSF
        reflect_amp_PSF = amp_PSF_crop * samp_prof_crop

        # sp.io.savemat('data/reflect_amp_PSF/reflect_amp_PSF_' + str(scan_num_y * config['zern_test']['scan_num_x'] + scan_num_x)\
        #      + '.mat', dict(reflect_amp_PSF = reflect_amp_PSF))
        
        # Pad reflection amplitude PSF with zeros before inverse Fourier transform
        reflect_amp_PSF_pad = np.pad(reflect_amp_PSF, (np.shape(reflect_amp_PSF)[0] // 2, \
            np.shape(reflect_amp_PSF)[0] // 2), 'constant', constant_values = (0, 0))

        # Perform inverse Fourier transform
        pupil_func_2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(reflect_amp_PSF_pad)))
        
        # Crop reflection pupil function
        start_2 = (np.shape(pupil_func_2)[0] - config['reflect_prof']['obj_grid_size']) // 2
        pupil_func_2 = pupil_func_2[start_1 : start_1 + config['reflect_prof']['obj_grid_size'], \
            start_1 : start_1 + config['reflect_prof']['obj_grid_size']]

        # Check phase of reflection pupil function
        phase_reflect = np.arctan2(np.imag(pupil_func_2), np.real(pupil_func_2)) / (2 * np.pi / config['AO']['lambda'])

        # sp.io.savemat('data/phase_reflect/phase_reflect_' + str(scan_num_y * config['zern_test']['scan_num_x'] + scan_num_x)\
        #      + '.mat', dict(phase_reflect = phase_reflect))

        # Multiply reflection pupil function with detection path pupil function to get final pupil function
        pupil_func_3 = pupil_func_2 * pupil_func_out_binned

        # Get detection phase profile
        phase_det = np.arctan2(np.imag(pupil_func_3), np.real(pupil_func_3)) / (2 * np.pi / config['AO']['lambda'])

        # sp.io.savemat('data/phase_det/phase_det_' + str(scan_num_y * config['zern_test']['scan_num_x'] + scan_num_x)\
        #      + '.mat', dict(phase_det = phase_det))

        # Pad detection phase profile to size of sensor
        # pad_num_2 = (settings['sensor_width'] - np.shape(phase_det)[0]) // 2
        # phase_det = np.pad(phase_det, (pad_num_2, pad_num_2), 'constant', constant_values = (0, 0))
        
        # Interpolate detection phase profile to size of sensor
        mag_fac = settings['sensor_width'] / np.shape(phase_det)[0]
        phase_det = sp.ndimage.zoom(phase_det, mag_fac, prefilter = True)

        # Apply phase mask to detection phase profile
        phase_det = phase_det * phase_mask

    except Exception as e:
        print(e)

    return phase_det