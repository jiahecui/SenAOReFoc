from ximea import xiapi
import numpy as np
from PIL import Image
from config import config

# Create instance for first connected camera
cam = xiapi.Camera()

print('Opening camera...')

# Start camera communication
cam.open_device_by_SN(config['camera']['SN'])

print('Device type is %s' % cam.get_device_type())
print('Device model ID is %s' % cam.get_device_model_id())
print('Device name is %s' % cam.get_device_name())

# Set camera settings
cam.set_imgdataformat(config['camera']['dataformat'])
cam.set_exposure(config['camera']['exposure'])

print('Exposure set to %i us' % cam.get_exposure())

# Create instance of Image to store image data and metadata
img = xiapi.Image()
dataimages = np.zeros((config['camera']['sensor_height'], config['camera']['sensor_width']))

print('Starting data acquisition...')

# Start data acquisition
cam.start_acquisition()

# Get data and pass them from camera to img, if timeout error is raised, print and continue
try:
    cam.get_image(img, timeout = config['camera']['timeout'])

    # Create numpy array with data from camera
    dataimages = img.get_image_data_numpy()

    # Print image data and metadata
    print('Image width (pixels): ' + str(img.width))
    print('Image height (pixels): ' + str(img.height))
    print('First line of pixels: ' + str(dataimages[0]))
    
except xiapi.Xi_error as err:
    if err.status == 10:
        print('Timeout error occured.')
    else:
        raise

print('Stopping acquisition...')

# Stop data acquisition
cam.stop_acquisition()

# Stop communication
cam.close_device()

print('Displaying image...')

# Show and save acquired image
img = Image.fromarray(dataimages, 'L')
img.show()
img.save('data/xi_example.bmp')

print('Done.')