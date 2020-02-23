import h5py
import PIL.Image
import numpy as np
import scipy as sp
from scipy import interpolate
from HDF5_dset import get_mat_dset

data = np.abs(get_mat_dset()) * 64
# x = np.linspace(-3, 3, 164)
# x = x.reshape(-1, 1)
# data = np.exp(-0.5 * x ** 2)
# print(np.shape(data))
# data = np.dot(data, data.T)
im1 = PIL.Image.fromarray(data, 'L')
im1.show()
# data_pad = np.pad(data, (430, 430), 'constant', constant_values = (0, 0))
x, y = (np.arange(164) for i in range(2))
interp_func_1 = sp.interpolate.RectBivariateSpline(x, y, data, kx = 3, ky = 3)
interp_func_2 = sp.interpolate.interp2d(x, y, data, kind = 'cubic')
xx, yy = (np.arange(656) for i in range(2))
data_interp_1 = interp_func_1(xx, yy)
data_interp_2 = interp_func_2(xx, yy)
data_diff = data_interp_2 - data_interp_1
# f1 = np.fft.fft2(data)
# f1 = np.fft.fftshift(f1)
# im2 = PIL.Image.fromarray(f1, 'L')
# im2.show()
# f1 = np.pad(f1, (246, 246), 'constant',  constant_values = (0 + 0j, 0 + 0j))
# im3 = PIL.Image.fromarray(f1, 'L')
# im3.show()
# f1 = np.fft.ifftshift(f1)
# data_interp = np.fft.ifft2(f1) * 16
# img_real = np.real(data_interp)
# img_imag = np.imag(data_interp)
# print(np.shape(data_interp))
im4 = PIL.Image.fromarray(data_interp_1, 'L')
im4.show()
im5 = PIL.Image.fromarray(data_interp_2, 'L')
im5.show()
# im5 = PIL.Image.fromarray(img_real, 'L')
# im5.show()
# im6 = PIL.Image.fromarray(img_imag, 'L')
# im6.show()

@Slot(object)
    def run2(self):
        try:
            # Set process flags
            self.loop = True 
            self.log = True

            # Start thread
            self.start.emit()

            """
            Closed-loop AO process to handle obscured S-H spots using a FIXED GAIN, iterated until residual phase error is below value 
            given by Marechel criterion or iteration has reached maximum
            """ 
            # Initialise AO information parameter
            self.AO_info = {'zern_AO_2': {}}
        
            # Create new datasets in HDF5 file to store closed-loop AO data and open file
            get_dset(self.SB_settings, 'zern_AO_2', flag = 1)
            data_file = h5py.File('data_info.h5', 'a')
            data_set_1 = data_file['AO_img']['zern_AO_2']
            data_set_2 = data_file['AO_info']['zern_AO_2']

            self.message.emit('Process started for closed-loop AO via Zernikes with obscured subapertures...')

            # Initialise deformable mirror voltage array
            voltages = np.zeros(config['DM']['actuator_num'])

            prev1 = time.perf_counter()

            # Run closed-loop control until residual phase error is below a certain value or iteration has reached specified maximum
            for i in range(config['AO']['loop_max']):
                
                if self.loop:

                    try:
                        # Update mirror control voltages
                        if i == 0:
                            voltages = config['DM']['vol_bias']
                        else:
                            voltages -= config['AO']['loop_gain'] * np.ravel(np.dot(self.mirror_settings['control_matrix_zern'], \
                                zern_err[:config['AO']['control_coeff_num']] / 2))

                            # print('Voltages {}: {}'.format(i, voltages))
                            print('Max and min values of voltages {} are: {}, {}'.format(i, np.max(voltages), np.min(voltages)))

                        if config['dummy']:
                            if config['real_phase']:

                                # Update phase profile, slope data, S-H spot image, and calculate Strehl ratio 
                                if i == 0:

                                    # Retrieve phase profile
                                    phase_init = get_mat_dset(self.SB_settings, flag = 1)

                                    # Display correction phase introduced by DM and corrected phase
                                    self.image.emit(abs(phase_init))
                                    time.sleep(2)

                                    print('Max and min values of initial phase are: {} um, {} um'.format(np.amax(phase_init), np.amin(phase_init)))

                                    # Calculate strehl ratio of initial phase profile
                                    strehl_init = self.strehl_calc(phase_init / (config['AO']['lambda'] / (2 * np.pi)))
                                    self.strehl[i] = strehl_init

                                    print('Strehl ratio of initial phase is:', strehl_init)  

                                    # Retrieve slope data from updated phase profile to use as theoretical centroids for simulation of S-H spots
                                    spot_cent_slope_x, spot_cent_slope_y = get_mat_dset(self.SB_settings, flag = 2)

                                    # Get simulated S-H spots and append to list
                                    spot_img = SpotSim(self.SB_settings)
                                    AO_image, spot_cent_x, spot_cent_y = spot_img.SH_spot_sim(centred = 1, xc = spot_cent_slope_x, yc = spot_cent_slope_y)
                                    dset_append(data_set_1, 'dummy_AO_img', AO_image)
                                    dset_append(data_set_1, 'dummy_spot_cent_x', spot_cent_x)
                                    dset_append(data_set_1, 'dummy_spot_cent_y', spot_cent_y)

                                    # print('spot_cent_slope_x:', spot_cent_slope_x)
                                    # print('spot_cent_slope_y:', spot_cent_slope_y)

                                    phase = phase_init.copy()
                                    strehl = strehl_init.copy()
    
                                else:

                                    # Calculate phase profile introduced by DM
                                    delta_phase = self.phase_calc(voltages)

                                    # Update phase data
                                    phase = phase_init - delta_phase

                                    # Display correction phase introduced by DM and corrected phase
                                    self.image.emit(abs(delta_phase))
                                    time.sleep(2)
                                    self.image.emit(abs(phase))
                                    time.sleep(2)

                                    print('Max and min values of phase are: {} um, {} um'.format(np.amax(phase), np.amin(phase)))

                                    # Calculate strehl ratio of updated phase profile
                                    strehl = self.strehl_calc(phase / (config['AO']['lambda'] / (2 * np.pi)))
                                    self.strehl[i] = strehl

                                    print('Strehl ratio of phase {} is: {}'.format(i + 1, strehl))

                                    # Retrieve slope data from initial phase profile to use as theoretical centroids for simulation of S-H spots
                                    spot_cent_slope_x, spot_cent_slope_y = get_slope_from_phase(self.SB_settings, phase)
                                    
                                    # Get simulated S-H spots and append to list
                                    spot_img = SpotSim(self.SB_settings)
                                    AO_image, spot_cent_x, spot_cent_y = spot_img.SH_spot_sim(centred = 1, xc = spot_cent_slope_x, yc = spot_cent_slope_y)
                                    dset_append(data_set_1, 'dummy_AO_img', AO_image)
                                    dset_append(data_set_1, 'dummy_spot_cent_x', spot_cent_x)
                                    dset_append(data_set_1, 'dummy_spot_cent_y', spot_cent_y)

                                    # print('spot_cent_slope_x:', spot_cent_slope_x)
                                    # print('spot_cent_slope_y:', spot_cent_slope_y)
                            else:

                                self.done.emit(2)

                        else:                     

                            # Send values vector to mirror
                            self.mirror.Send(voltages)
                            
                            # Wait for DM to settle
                            time.sleep(config['DM']['settling_time'])
                        
                            # Acquire S-H spots using camera and append to list
                            AO_image = acq_image(self.sensor, self.SB_settings['sensor_width'], self.SB_settings['sensor_height'], acq_mode = 0)
                            dset_append(data_set_1, 'real_AO_img', AO_image)

                        # Image thresholding to remove background
                        AO_image = AO_image - config['image']['threshold'] * np.amax(AO_image)
                        AO_image[AO_image < 0] = 0
                        self.image.emit(AO_image)

                        # Calculate centroids of S-H spots
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y, slope_x, slope_y = acq_centroid(self.SB_settings, flag = 5)
                        act_cent_coord, act_cent_coord_x, act_cent_coord_y = map(np.asarray, [act_cent_coord, act_cent_coord_x, act_cent_coord_y])

                        # print('slope_x:', slope_x)
                        # print('slope_y:', slope_y)

                        # Remove corresponding elements from slopes and rows from influence function matrix, zernike matrix and zernike derivative matrix
                        index_remove = np.where(slope_x + self.SB_settings['act_ref_cent_coord_x'].astype(int) + 1 == 0)[1]
                        # print('Shape index_remove:', np.shape(index_remove))
                        # print('index_remove:', index_remove)
                        index_remove_inf = np.concatenate((index_remove, index_remove + self.SB_settings['act_ref_cent_num']), axis = None)
                        # print('Shape index_remove_inf:', np.shape(index_remove_inf))
                        # print('index_remove_inf:', index_remove_inf)
                        slope_x = np.delete(slope_x, index_remove, axis = 1)
                        slope_y = np.delete(slope_y, index_remove, axis = 1)
                        # print('Shape slope_x:', np.shape(slope_x))
                        # print('Shape slope_y:', np.shape(slope_y))
                        act_cent_coord = np.delete(act_cent_coord, index_remove, axis = None)
                        # print('Shape act_cent_coord:', np.shape(act_cent_coord))
                        zern_matrix = np.delete(self.mirror_settings['zern_matrix'].copy(), index_remove, axis = 0)
                        # print('Shape zern_matrix:', np.shape(zern_matrix))
                        inf_matrix_slopes = np.delete(self.mirror_settings['inf_matrix_slopes'].copy(), index_remove_inf, axis = 0)
                        # print('Shape inf_matrix_slopes:', np.shape(inf_matrix_slopes))
                        diff_matrix = np.delete(self.mirror_settings['diff_matrix'].copy(), index_remove_inf, axis = 0)
                        # print('Shape diff_matrix:', np.shape(diff_matrix))

                        # Draw actual S-H spot centroids on image layer
                        AO_image.ravel()[act_cent_coord.astype(int)] = 0
                        self.image.emit(AO_image)

                        # Recalculate Cholesky decomposition of np.dot(zern_matrix.T, zern_matrix)
                        p_matrix = np.linalg.cholesky(np.dot(zern_matrix.T, zern_matrix))
                        # print('Shape p_matrix:', np.shape(p_matrix))

                        # Recalculate conversion matrix
                        conv_matrix = np.dot(p_matrix, np.linalg.pinv(diff_matrix))
                        # print('Shape conv_matrix:', np.shape(conv_matrix))

                        # Recalculate control function via zernikes
                        control_matrix_zern = np.linalg.pinv(np.dot(conv_matrix, inf_matrix_slopes))
                        # print('Shape control_matrix_zern:', np.shape(control_matrix_zern))

                        # Concatenate slopes into one slope matrix
                        slope = (np.concatenate((slope_x, slope_y), axis = 1)).T

                        # Get detected zernike coefficients from slope matrix
                        self.zern_coeff_detect = np.dot(conv_matrix, slope)

                        # Get phase residual (zernike coefficient residual error) and calculate root mean square (rms) error
                        zern_err = self.zern_coeff_detect.copy()
                        rms = np.sqrt((zern_err ** 2).mean())
                        self.loop_rms[i] = rms
                        
                        print('Root mean square error {} is {}'.format(i + 1, rms))                        

                        # Append data to list
                        if config['dummy']:
                        #     dset_append(data_set_2, 'dummy_spot_slope_x', slope_x)
                        #     dset_append(data_set_2, 'dummy_spot_slope_y', slope_y)
                        #     dset_append(data_set_2, 'dummy_spot_slope', slope)
                            dset_append(data_set_2, 'dummy_spot_zern_err', zern_err)
                        else:
                        #     dset_append(data_set_2, 'real_spot_slope_x', slope_x)
                        #     dset_append(data_set_2, 'real_spot_slope_y', slope_y)
                        #     dset_append(data_set_2, 'real_spot_slope', slope)
                            dset_append(data_set_2, 'real_spot_zern_err', zern_err)

                        # Compare rms error with tolerance factor (Marechel criterion) and decide whether to break from loop
                        if strehl >= config['AO']['tolerance_fact_strehl']:
                            break                 

                    except Exception as e:
                        print(e)
                else:

                    self.done.emit(2)

            # Close HDF5 file
            data_file.close()

            self.message.emit('Process complete.')
            print('Final root mean square error of detected wavefront is: {} microns'.format(rms))

            prev2 = time.perf_counter()
            print('Time for closed-loop AO process is:', (prev2 - prev1))

            """
            Returns closed-loop AO information into self.AO_info
            """             
            if self.log:

                self.AO_info['zern_AO_2']['loop_num'] = i
                self.AO_info['zern_AO_2']['residual_phase_err_1'] = self.loop_rms
                self.AO_info['zern_AO_2']['strehl_ratio'] = self.strehl

                self.info.emit(self.AO_info)
                self.write.emit()
            else:

                self.done.emit(2)

            # Finished closed-loop AO process
            self.done.emit(2)

        except Exception as e:
            raise
            self.error.emit(e)