from ximea import xiapi
import PIL.Image    

#Note: PIL does not support 16bit images (XI_MONO16 format setting)

#create instance for first connected camera 
cam = xiapi.Camera()

#start communication
print('Opening first camera...')
cam.open_device()

#settings
cam.set_imgdataformat('XI_MONO8')
cam.set_exposure(20000)

#create instance of Image to store image data and metadata
img = xiapi.Image()

#start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

#get data and pass them from camera to img
cam.get_image(img)

#create numpy array with data from camera. Dimensions of array are determined
#by imgdataformat
data = img.get_image_data_numpy()

#stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

#stop communication
cam.close_device()

#show acquired image
print('Drawing image...')
img = PIL.Image.fromarray(data, 'L')
img.show()

print('Done.')
