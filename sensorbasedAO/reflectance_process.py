import numpy as np
import scipy as sp
from scipy import signal
from scipy import ndimage
from scipy.ndimage import gaussian_filter

from config import config
from reflectance_profile import mirror_samp_sim, bead_samp_sim

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
        # Bin phase to the shape of reflectance profile
        phase_binned = rebin(phase, (config['reflect_prof']['PSF_kern_size'], config['reflect_prof']['PSF_kern_size']))
        phase_binned_det = np.flipud(np.fliplr(phase_binned))

        # Get pupil function and detection pupil function from phase aberrations
        pupil_func = np.exp(-2 * np.pi * 1j / config['AO']['lambda'] * phase_binned)
        pupil_func_det = np.exp(-2 * np.pi * 1j / config['AO']['lambda'] * phase_binned_det)

        # Calculate double-pass pupil function 
        pupil_func = pupil_func * pupil_func_det

        # Pad double-pass pupil function with zeros before Fourier transform
        pupil_func_pad = np.pad(pupil_func, (np.shape(pupil_func)[0] // 2, \
            np.shape(pupil_func)[0] // 2), 'constant', constant_values = (0, 0))

        # Perform Fourier transform and shift zero frequency components to centre
        amp_PSF = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pupil_func_pad)))

        # Crop double-pass amplitude PSF to convolve with sample reflectance profile
        start_1 = (np.shape(amp_PSF)[0] - config['reflect_prof']['PSF_kern_size']) // 2
        amp_PSF_crop = amp_PSF[start_1 : start_1 + config['reflect_prof']['PSF_kern_size'], \
            start_1 : start_1 + config['reflect_prof']['PSF_kern_size']]

        # Generate sample reflectance profile
        if config['reflect_prof']['samp_num'] == 0:
            samp_prof = mirror_samp_sim()
        elif config['reflect_prof']['samp_num'] == 1:
            samp_prof = bead_samp_sim()

        # Crop sample reflectance profile to convolve with double-pass amplitude PSF
        start_2 = int((np.shape(samp_prof)[0] - config['reflect_prof']['samp_kern_size']) // 2) + 1
        samp_prof_crop = samp_prof[start_2 : start_2 + config['reflect_prof']['samp_kern_size'], \
            start_2 : start_2 + config['reflect_prof']['samp_kern_size']]

        # Generate detection amplitude PSF
        reflect_amp_PSF = sp.signal.convolve2d(amp_PSF_crop, samp_prof_crop, mode = 'same')

        # Multiply double-pass amplitude PSF with detection amplitude PSF to yield detection intensity PSF
        detect_amp_PSF = amp_PSF_crop * reflect_amp_PSF

        # Pad detection intensity PSF with zeros before inverse Fourier transform
        detect_amp_PSF_pad = np.pad(detect_amp_PSF, (np.shape(detect_amp_PSF)[0] // 2, \
            np.shape(detect_amp_PSF)[0] // 2), 'constant', constant_values = (0, 0))

        # Perform inverse Fourier transform and calculate detection pupil function
        pupil_func_2 = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(detect_amp_PSF_pad)))

        # Get detection phase profile
        phase_2 = np.arctan2(np.imag(pupil_func_2), np.real(pupil_func_2)) / (2 * np.pi / config['AO']['lambda'])

        # Crop detection phase profile
        start_3 = config['reflect_prof']['PSF_kern_size'] // 2
        phase_2 = phase_2[start_3: start_3 + start_3 * 2, start_3: start_3 + start_3 * 2]

        # Pad detection phase profile to size of sensor
        # pad_num_2 = (settings['sensor_width'] - np.shape(phase_2)[0]) // 2
        # phase_2 = np.pad(phase_2, (pad_num_2, pad_num_2), 'constant', constant_values = (0, 0))
        
        # Interpolate detection phase profile to size of sensor
        mag_fac = settings['sensor_width'] / np.shape(phase_2)[0]
        phase_2 = sp.ndimage.zoom(phase_2, mag_fac, prefilter = True)

        # Generate boolean phase mask
        phase_rad = int(pupil_diam * 1e3 / settings['pixel_size']) // 2
        coord_x, coord_y = (np.arange(int(-settings['sensor_width'] / 2), int(-settings['sensor_width'] / 2) + \
            settings['sensor_width']) for i in range(2))
        coord_xx, coord_yy = np.meshgrid(coord_x, coord_y)
        phase_mask = np.sqrt(coord_xx ** 2 + coord_yy ** 2) < phase_rad

        # Apply phase mask to detection phase profile
        phase_2 = phase_2 * phase_mask

    except Exception as e:
        print(e)

    return phase_2