#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Install the PIPython library."""
# Unnecessary parens after u'print' keyword pylint: disable=C0325

import os
import sys
from setuptools import setup

__version__ = '1.5.1.7'

try:
    # Redefining built-in 'input' pylint: disable=W0622
    # Invalid constant name "input" pylint: disable=C0103
    input = raw_input
except NameError:
    pass


def setwinreg():
    """Write __version__ to windows registry for the PI UpdateFinder."""
    if sys.platform not in ('win32', 'cygwin'):
        return
    try:
        import winreg  # Python 3
    except ImportError:
        import _winreg as winreg  # Python 2
    print('Updating Windows registry...')
    if 'PROCESSOR_ARCHITEW6432' in os.environ:
        key = r'SOFTWARE\Wow6432Node\PI\PIPython'
    else:
        key = r'SOFTWARE\PI\PIPython'
    reghandle = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
    keyhandle = winreg.CreateKey(reghandle, key)
    winreg.SetValueEx(keyhandle, 'KeyValue', None, winreg.REG_SZ, 'PIPython')
    winreg.SetValueEx(keyhandle, 'Version', None, winreg.REG_SZ, __version__)
    winreg.SetValueEx(keyhandle, 'Path', None, winreg.REG_SZ, sys.prefix)
    winreg.CloseKey(keyhandle)
    winreg.CloseKey(reghandle)


if __name__ == '__main__':
    try:
        setwinreg()
    except:  # exception can be different, No exception type(s) specified pylint: disable=W0702
        print("\nWARNING: It's recommended to run this setup with administrator permissions.")
        print('You can install PIPython with user permissions but then PIUpdateFinder will')
        print('not notify you about updates.\n')
        input('Press ENTER to install PIPython with user permissions or CTRL+C to cancel...')
    setup(name='PIPython',
          version=__version__,
          description='Collection of libraries to use PI devices and process GCS data.',
          long_description='PIPython is a collection of Python modules to access a PI '
                           'device and process GCS data. It can be used with Python '
                           '2.7+ and 3.3+ on Windows, Linux and OS X and without the '
                           'GCS DLL also on any other platform.',
          url='http://www.physikinstrumente.com',
          author='Physik Instrumente (PI) GmbH & Co. KG',
          author_email='support-software@pi.ws',
          license='See the included eula.md file.',
          classifiers=[
              # https://pypi.python.org/pypi?%3Aaction=list_classifiers
              'Development Status :: 5 - Production/Stable',
              'License :: Other/Proprietary License',
              'Programming Language :: Python :: 2.7',
              'Programming Language :: Python :: 3.3',
              'Programming Language :: Python :: 3.4',
              'Programming Language :: Python :: 3.5',
          ],
          packages=[
              'pipython',
              'pipython.interfaces',
          ],
          zip_safe=False)
