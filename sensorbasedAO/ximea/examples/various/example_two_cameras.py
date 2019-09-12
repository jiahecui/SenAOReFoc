from ximea import xiapi

CAMERAS_ON_SAME_CONTROLLER = 2

#create instance for cameras
cam1 = xiapi.Camera(dev_id=0)
cam2 = xiapi.Camera(dev_id=1)

#start communication
print('Opening cameras...')
cam1.open_device()
cam2.open_device()

#set interface data rate
interface_data_rate=cam1.get_limit_bandwidth();
camera_data_rate = int(interface_data_rate / CAMERAS_ON_SAME_CONTROLLER) ;

#set data rate
cam1.set_limit_bandwidth(camera_data_rate);
cam2.set_limit_bandwidth(camera_data_rate);

#print device serial numbers
print('Camera 1 serial number: ' + str(cam1.get_device_sn()))
print('Camera 2 serial number: ' + str(cam2.get_device_sn()))

#settings
cam1.set_exposure(10000)
print('Cam1: Exposure was set to %i us' %cam1.get_exposure())
cam2.set_exposure(10000)
print('Cam2: Exposure was set to %i us' %cam2.get_exposure())


#create instance of Image to store image data and metadata
img1 = xiapi.Image()
img2 = xiapi.Image()

#start data acquisition
print('Starting data acquisition...\n')
cam1.start_acquisition()
cam2.start_acquisition()

#get data and pass them from cameras to img
cam1.get_image(img1)
cam2.get_image(img2)

#get raw data from cameras
#for Python2.x function returns string
#for Python3.x function returns bytes
data_raw1 = img1.get_image_data_raw()
data_raw2 = img2.get_image_data_raw()

#transform data to list
data1 = list(data_raw1)
data2 = list(data_raw2)

#print image data and metadata
print('Cam1: image (' + str(img1.width) + 'x' + str(img1.height) + ') received from camera.')
print('First 10 pixels: ' + str(data1[:10]) + '\n')

print('Cam2: image (' + str(img2.width) + 'x' + str(img2.height) + ') received from camera.')
print('First 10 pixels: ' + str(data2[:10]))
print('\n')

#stop data acquisition
print('Cam1: Stopping acquisition...')
cam1.stop_acquisition()
print('Cam2: Stopping acquisition...')
cam2.stop_acquisition()

#stop communication
cam1.close_device()
cam2.close_device()

print('Done.')



