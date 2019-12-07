from ximea import xiapi
import numpy as np
from error_camera_2 import img_acq

sensor = xiapi.Camera()

# start camera communication
sensor.open_device_by_SN('26883050')

# camera settings
sensor.set_imgdataformat('XI_MONO16')
sensor.set_exposure(4000)

# Acquire images in a loop
for i in range(200):
    print('image {}'.format(i))
    img = img_acq(sensor)

# stop communication
sensor.close_device()