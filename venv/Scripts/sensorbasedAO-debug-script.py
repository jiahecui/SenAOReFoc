#!f:\documents\oxford\sensorbasedao\venv\scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'sensorbasedAO','console_scripts','sensorbasedAO-debug'
__requires__ = 'sensorbasedAO'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('sensorbasedAO', 'console_scripts', 'sensorbasedAO-debug')()
    )
