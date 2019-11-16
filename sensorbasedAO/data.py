import time
from datetime import datetime
from threading import Thread
from queue import Queue
import numpy as np
from collections import defaultdict
from pathlib import Path
import imageio
from sensorbasedAO.common import Stack
from sensorbasedAO.log import get_logger

logger = get_logger(__name__)

class Image(np.ndarray):
    """
    Custom image class using numpy array
    """
    def __new__(cls, input_array, metadata = None):
        obj = np.asarray(input_array).view(cls)
        obj.metadata = metadata
        return obj

    def __array_finalize__(self, obj):
        if obj is None: 
            return
        self.metadata = getattr(obj, 'metadata', None)
