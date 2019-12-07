from ximea import xiapi
import numpy as np

def img_acq(sensor):

    # Create instance of Ximea Image to store image data and metadata
    img = xiapi.Image()
    dataimage = np.zeros((2048,2048))

    # Start data acquisition for one frame
    sensor.start_acquisition()

    try:

        # Get data and pass them from camera to img
        sensor.get_image(img, timeout = 25)

        # Create numpy array with data from camera, dimensions are determined by imgdataformats
        dataimage = img.get_image_data_numpy()

        # Bin numpy arrays by averaging pixels to fit on viewer
        shape = (1024, 2, 1024, 2)
        dataimage = dataimage.reshape(shape).mean(-1).mean(1)

    except Exception as e:
        print(e)

    # Stop data acquisition
    sensor.stop_acquisition()

    return dataimage
    
