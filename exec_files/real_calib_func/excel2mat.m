%% This script converts the real calibration functions from excel to mat files

close all
clear
clc

calib_slope_x = xlsread('calib_slope_x.xlsx');
calib_slope_y = xlsread('calib_slope_y.xlsx');
control_matrix_slopes = xlsread('control_matrix_slopes.xlsx');
inf_matrix_slopes = xlsread('inf_matrix_slopes.xlsx');
inf_matrix_slopes_SV = xlsread('inf_matrix_slopes_SV.xlsx');
svd_check_slopes = xlsread('svd_check_slopes.xlsx');

save('calib_slope_x.mat', '-v7.3', 'calib_slope_x');
save('calib_slope_y.mat', '-v7.3', 'calib_slope_y');
save('control_matrix_slopes.mat', '-v7.3', 'control_matrix_slopes');
save('inf_matrix_slopes.mat', '-v7.3', 'inf_matrix_slopes');
save('inf_matrix_slopes_SV.mat', '-v7.3', 'inf_matrix_slopes_SV');
save('svd_check_slopes.mat', '-v7.3', 'svd_check_slopes');