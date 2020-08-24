#!/usr/bin/python
# -*- coding: utf8 -*-

import sys
import os
import time
import struct

''' Add '/Lib' or '/Lib64' to path '''
if (8 * struct.calcsize("P")) == 32:
    print("Use x86 libraries.")
    sys.path.append(os.path.join(os.path.dirname(__file__), 'Lib'))
else:
    print("Use x86_64 libraries.")
    sys.path.append(os.path.join(os.path.dirname(__file__), 'Lib64'))

''' Import Alpao SDK class '''
from asdk import DM

''' Start example '''
def main(args):
    print("Please enter the S/N within the following format BXXYYY (see DM backside): ")
    serialName = sys.stdin.readline().rstrip()
    
    print("Connect the mirror")
    dm = DM( serialName )
        
    print("Retrieve number of actuators")
    nbAct = int( dm.Get('NBOfActuator') )
    print( "Number of actuator for " + serialName + ": " + str(nbAct) )
    
    print("Send 0 on each actuators")
    values = [0.] * nbAct
    dm.Send( values )
    
    print("We will send on all actuator 50% for 1/2 second")
    for i in range( nbAct ):
        values[i] = 0.5
        # Send values vector
        dm.Send( values )
        print("Send 0.50 on actuator " + str(i))
        time.sleep(0.5) # Wait for 0.5 second
        values[i] = 0
    
    print("Reset")
    dm.Reset()
    
    print("Exit")

if __name__=="__main__":
    main(sys.argv)