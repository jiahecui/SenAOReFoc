from setuptools import setup, find_packages

setup(name='SenAOReFoc',
      version='v1.0.0',
      description='Sensorbased AO and remote focusing control software',
      author='Jiahe Cui',
      author_email='jiahe.cui@eng.ox.ac.uk',
      packages=find_packages(include=['sensorbasedAO.*']),
      python_requires='>=3.6',
      install_requires=[
            'numpy', 
            'PySide2==5.13.2', 
            'qimage2ndarray', 
            'Click',
            'scipy==1.5.4',
            'h5py==2.9.0',
            'pyyaml'],
      entry_points={
            'console_scripts': [
                  'sensorbasedAO=sensorbasedAO.app:main',
                  'sensorbasedAO-debug=sensorbasedAO.app:debug',
            ]
      }
      )
