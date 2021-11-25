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
            'PySide2==5.15.0', 
            'qimage2ndarray', 
            'Click',
            'scipy==1.5.4',
            'h5py==2.9.0',
            'Cython',
            'rpyc',
            'tensorflow-gpu==2.5.0',
            'joblib',
            'sklearn',
            'pandas'],
      entry_points={
            'console_scripts': [
                  'sensorbasedAO=sensorbasedAO.app:main',
                  'sensorbasedAO-debug=sensorbasedAO.app:debug',
            ]
      }
      )
