import numpy as np
import scipy as sp
from scipy import signal
from scipy import ndimage
from scipy.ndimage import gaussian_filter

from config import config
from reflectance_profile import get_samp_sim

def reflect_process(settings, phase, pupil_diam):
    """
    Simulates the process of a beam reflecting off a specimen then arriving at the SHWS
    """
    def rebin(arr, new_shape):
        """
        Function to bin a 2D array to the shape specified by new_shape
        """
        shape = (new_shape[0], arr.shape[0] // new_shape[0], new_shape[1], arr.shape[1] // new_shape[1])

        return arr.reshape(shape).mean(-1).mean(1)

    try:
        # Bin phase to the size of object space grid
        phase_binned = rebin(phase, (config['reflect_prof']['obj_grid_size'], config['reflect_prof']['obj_grid_size']))

        # Get detection path phase by flipping original phase left/right up/down
        phase_binned_det = np.flipud(np.fliplr(phase_binned))

        # Get pupil function and detection path pupil function from phase aberrations
        pupil_func = np.exp(-2 * np.pi * 1j / config['AO']['lambda'] * phase_binned)
        pupil_func_det = np.exp(-2 * np.pi * 1j / config['AO']['lambda'] * phase_binned_det)

        # Pad pupil function with zeros before Fourier transform
        pupil_func_pad = np.pad(pupil_func, (np.shape(pupil_func)[0] // 2, \
            np.shape(pupil_func)[0] // 2), 'constant', constant_values = (0, 0))

        # Perform Fourier transform and shift zero frequency components to centre to get amplitude PSF
        amp_PSF = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_func_pad)))

        # Normalise amplitude PSF
        amp_PSF = amp_PSF / np.amax(amp_PSF)

        # Crop amplitude PSF to convolve with sample reflectance profile
        start_1 = (np.shape(amp_PSF)[0] - config['reflect_prof']['obj_grid_size']) // 2
        amp_PSF_crop = amp_PSF[start_1 : start_1 + config['reflect_prof']['obj_grid_size'], \
            start_1 : start_1 + config['reflect_prof']['obj_grid_size']]

        # Generate sample reflectance profile
        samp_prof_crop = get_samp_sim(config['reflect_prof']['samp_num'])

        # Generate reflection amplitude PSF
        reflect_amp_PSF = amp_PSF_crop * sp.signal.convolve2d(amp_PSF_crop, samp_prof_crop, mode = 'same')

        # Pad reflection amplitude PSF with zeros before inverse Fourier transform
        reflect_amp_PSF_pad = np.pad(reflect_amp_PSF, (np.shape(reflect_amp_PSF)[0] // 2, \
            np.shape(reflect_amp_PSF)[0] // 2), 'constant', constant_values = (0, 0))

        # Perform inverse Fourier transform
        pupil_func_2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(reflect_amp_PSF_pad)))
        
        # Crop reflection pupil function
        start_2 = (np.shape(pupil_func_2)[0] - config['reflect_prof']['obj_grid_size']) // 2
        pupil_func_2 = pupil_func_2[start_1 : start_1 + config['reflect_prof']['obj_grid_size'], \
            start_1 : start_1 + config['reflect_prof']['obj_grid_size']]

        # Multiply reflection pupil function with detection path pupil function to get final pupil function
        pupil_func_3 = pupil_func_2 * pupil_func_det

        # Get detection phase profile
        phase_det = np.arctan2(np.imag(pupil_func_3), np.real(pupil_func_3)) / (2 * np.pi / config['AO']['lambda'])

        # Pad detection phase profile to size of sensor
        # pad_num_2 = (settings['sensor_width'] - np.shape(phase_det)[0]) // 2
        # phase_det = np.pad(phase_det, (pad_num_2, pad_num_2), 'constant', constant_values = (0, 0))
        
        # Interpolate detection phase profile to size of sensor
        mag_fac = settings['sensor_width'] / np.shape(phase_det)[0]
        phase_det = sp.ndimage.zoom(phase_det, mag_fac, prefilter = True)

        # Generate boolean phase mask
        phase_rad = int(pupil_diam * 1e3 / settings['pixel_size']) // 2
        coord_x, coord_y = (np.arange(int(-settings['sensor_width'] / 2), int(-settings['sensor_width'] / 2) + \
            settings['sensor_width']) for i in range(2))
        coord_xx, coord_yy = np.meshgrid(coord_x, coord_y)
        phase_mask = np.sqrt(coord_xx ** 2 + coord_yy ** 2) < phase_rad

        # Apply phase mask to detection phase profile
        phase_det = phase_det * phase_mask

    except Exception as e:
        print(e)

    return phase_det