from scipy import io
import numpy as np
import scipy as sp
import h5py

# Parameters for reshaping of ML dataset
full_dataset = 0
select_dataset = 0
folder_flag = 4
folder_num = 2
samp_num = 1
aberr_num = 68
aberr_num_per_mode = 2
zern_num = 20
scan_num_x = 20
scan_num_y = 20

# Initialise list to store final stacked dataset
dataset = []
dataset_coord = []
dataset_target = []

# Create coordinate layer
coord_x = np.arange(scan_num_x)
coord_y = np.arange(scan_num_y)
coord_xx, coord_yy = np.meshgrid(coord_x, coord_y)

"""
Reshape dataset from one folder
"""
if not full_dataset and not select_dataset:

    aberr_matrix = sp.io.loadmat('data/ML_dataset/test/test' + str(folder_flag) + '/aberr_matrix')['aberr_matrix']

    for i in range(samp_num):

        for j in range(aberr_num):

            # Retrieve 2D-scan dataset
            temp_data_full = sp.io.loadmat('data/ML_dataset/test/test' + str(folder_flag) + '/samp_' + str(i) + '_aberr_' + str(j))['scan_point_aberrations']

            # Retrieve applied aberration coefficients
            temp_aberr = aberr_matrix[j,:]
            
            # Remove tip/tilt/defocus from 2D-scan dataset
            # temp_data[:, [0,1,3]] = 0

            # Retrieve column in 2D-scan dataset corresponding to applied zernike mode
            if j < 34:
                if j // aberr_num_per_mode == 0:
                    temp_data = temp_data_full[:, 2]
                else:
                    temp_data = temp_data_full[:, j // aberr_num_per_mode + 3]
            else:
                if (j - 34) // aberr_num_per_mode == 0:
                    temp_data = temp_data_full[:, 2]
                else:
                    temp_data = temp_data_full[:, (j - 34) // aberr_num_per_mode + 3]
            # if j == 0:
            #     temp_data = temp_data_full[:, 4]
            # elif j == 1:
            #     temp_data = temp_data_full[:, 6]
            # elif j == 2:
            #     temp_data = temp_data_full[:, 10]
            
            # Reshape the dataset to (x-axis, y-axis, channel)
            # temp_data = np.reshape(temp_data.T, (zern_num, scan_num_x, scan_num_y))
            temp_data = np.reshape(temp_data.T, (1, scan_num_x, scan_num_y))
            temp_data = np.moveaxis(temp_data, 0, -1)

            # Reshape aberration coefficients to (x-axis, y-axis, channel)
            # temp_aberr = np.reshape(temp_aberr.T, (zern_num, 1, 1))
            # temp_aberr = np.moveaxis(temp_aberr, 0, -1)

            # Add coordinate layer to back of dataset
            temp_data_coord = np.dstack((temp_data, coord_xx))
            temp_data_coord = np.dstack((temp_data_coord, coord_yy))

            # Append dataset to list
            dataset.append(temp_data)
            dataset_coord.append(temp_data_coord)

            # Append output target to list
            dataset_target.append(temp_aberr)

    # Stack the datasets to form (sample, x-axis, y-axis, channel)
    dataset = np.stack(dataset)
    dataset_coord = np.stack(dataset_coord)
    dataset_target = np.stack(dataset_target)

    print(np.shape(dataset))
    print(np.shape(dataset_coord))
    print(np.shape(dataset_target))

    # Save dataset
    sp.io.savemat('data/ML_dataset/test/test' + str(folder_flag) + '/dataset_' + str(folder_flag) + '.mat', dict(dataset = dataset))
    sp.io.savemat('data/ML_dataset/test/test' + str(folder_flag) + '/dataset_coord_' + str(folder_flag) + '.mat', dict(dataset_coord = dataset_coord))
    sp.io.savemat('data/ML_dataset/test/test' + str(folder_flag) + '/dataset_target_' + str(folder_flag) + '.mat', dict(dataset_target = dataset_target))
    

"""
Reshape dataset from all folders to create one full dataset
"""
if full_dataset:
    
    for f in range(folder_num):

        aberr_matrix = sp.io.loadmat('data/ML_dataset/test/test' + str(f + 1) + '/aberr_matrix')['aberr_matrix']
        
        for i in range(samp_num):

            for j in range(aberr_num):

                # Retrieve 2D-scan dataset 
                temp_data = sp.io.loadmat('data/ML_dataset/test/test' + str(f + 1) + '/samp_' + str(i) + '_aberr_' + str(j))['scan_point_aberrations']

                # Retrieve applied aberration coefficients
                temp_aberr = aberr_matrix[j,:]

                # Remove tip/tilt/defocus
                temp_data[:, [0,1,3]] = 0
                
                # Reshape the dataset to (x-axis, y-axis, channel)
                temp_data = np.reshape(temp_data.T, (zern_num, scan_num_x, scan_num_y))
                temp_data = np.moveaxis(temp_data, 0, -1)

                # Reshape aberration coefficients to (x-axis, y-axis, channel)
                temp_aberr = np.reshape(temp_aberr.T, (zern_num, 1, 1))
                temp_aberr = np.moveaxis(temp_aberr, 0, -1)

                # Add coordinate layer to back of dataset
                temp_data_coord = np.dstack((temp_data, coord_xx))
                temp_data_coord = np.dstack((temp_data_coord, coord_yy))

                # Append dataset to list
                dataset.append(temp_data)
                dataset_coord.append(temp_data_coord)

                # Append output target to list
                dataset_target.append(temp_aberr)

    # Stack the datasets to form (sample, x-axis, y-axis, channel)
    dataset = np.stack(dataset)
    dataset_coord = np.stack(dataset_coord)
    dataset_target = np.stack(dataset_target)

    print(np.shape(dataset))
    print(np.shape(dataset_coord))
    print(np.shape(dataset_target))

    # Save full dataset
    sp.io.savemat('data/ML_dataset/test/dataset.mat', dict(dataset = dataset))
    sp.io.savemat('data/ML_dataset/test/dataset_coord.mat', dict(dataset_coord = dataset_coord))
    sp.io.savemat('data/ML_dataset/test/dataset_target.mat', dict(dataset_target = dataset_target))