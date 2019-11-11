from ximea import xiapi
import numpy as np
import PIL.Image

# create instance for first connected camera
cam = xiapi.Camera()

# start camera communication
print('Opening camera...')
cam.open_device_by_SN('26883050')
print('Device type is %s' % cam.get_device_type())
print('Device model ID is %s' % cam.get_device_model_id())
print('Device name is %s' % cam.get_device_name())

# camera settings
cam.set_imgdataformat("XI_MONO8")
cam.set_exposure(2000)
print('Exposure set to %i us' % cam.get_exposure())

# create instance of Image to store image data and metadata
img = xiapi.Image()
dataimages = np.zeros((2048,2048))

# start data acquisition
print('Starting data acquisition...')
cam.start_acquisition()

# get data and pass them from camera to img
# if timeout error is raised, print it and continue
try:
    cam.get_image(img, timeout=10)

    #create numpy array with data from camera. Dimensions of array are determined
    #by imgdataformats
    dataimages = img.get_image_data_numpy()

    # print image data and metadata
    print('Image width (pixels):  ' + str(img.width))
    print('Image height (pixels): ' + str(img.height))
    print('First line of pixels: ' + str(dataimages[0]))
    
except xiapi.Xi_error as err:
    if err.status == 10:
        print('Timeout error occured.')
    else:
        raise

# stop data acquisition
print('Stopping acquisition...')
cam.stop_acquisition()

# stop communication
cam.close_device()

# show and save acquired image
print('Drawing image...')
img = PIL.Image.fromarray(dataimages, 'L')
img.show()
# img.save('xi_example.bmp')

print('Done.')