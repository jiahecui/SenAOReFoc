from setuptools import setup, find_packages

setup(name='sensorbasedAO',
      version='0.1',
      description='Sensor-based AO control software',
      author='Jiahe Cui',
      author_email='jiahe.cui@eng.ox.ac.uk',
      packages=find_packages(include=['sensorbasedAO.*']),
      python_requires='>=3.6',
      install_requires=[
            'numpy', 
            'Pillow', 
            'nifpga', 
            'PySide2', 
            'qtawesome',
            'qimage2ndarray', 
            'tifffile', 
            'imageio', 
            'pyyaml',
            'pyserial'
            ],
      extras_require={
            'test': ['pytest', 'pytest-qt']
      },
      entry_points={
            'console_scripts': [
                  'sensorbasedAO=sensorbasedAO.app:main',
                  'sensorbasedAO-debug=sensorbasedAO.app:debug',
            ]
      }
      )
