from ximea import xiapi
import PIL.Image

#create instance for first connected camera 
cam = xiapi.Camera()

#start communication
print('Opening first camera...')
cam.open_device()

#settings
cam.set_imgdataformat('XI_RGB24')
cam.set_exposure(10000)

#create instance of Image to store image data and metadata
img = xiapi.Image()

#start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

#get data and pass them from camera to img
cam.get_image(img)

#create numpy array with data from camera. Dimensions of array are determined
#by imgdataformat
#NOTE: PIL takes RGB bytes in opposite order, so invert_rgb_order is True
data = img.get_image_data_numpy(invert_rgb_order=True)

#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

#stop communication
cam.close_device()

#show acquired image
print('Saving image...')
img = PIL.Image.fromarray(data, 'RGB') 
img.save('xi_example.bmp')

print('Done.')
