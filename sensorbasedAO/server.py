import rpyc
import rpyc.lib
from rpyc.utils.server import ThreadedServer
from threading import Thread
import os
import sys
import time

import log
from config import config

logger = log.get_logger(__name__)
rpyc.lib.setup_logger()

class RemoteService(rpyc.Service):
    """
    Creates an instance of the rpyc server to access its internal methods and attributes, 
    as well as allow external client to access exposed methods and attributes.
    """
    def __init__(self, app, main):
        self.server = rpyc.Service()
        self.app = app
        self.main = main

        super().__init__() 

    def work(self): # Internal method
        """
        Detect change in RF status
        """
        try:
            while self.active:
                self.stat = self.main.ui.serverSpin.value()
                if self.last_stat != 1 and self.stat == 1:
                    self.callback()   # Notify the client of the change
                self.last_stat = self.stat
                _ = time.perf_counter() + self.interval
                while time.perf_counter() < _:
                    pass
        except Exception as e:
            print(e)

    def exposed_connect_server(self): # Exposed method
        return True

    def exposed_get_RF_settings(self): # Exposed method
        """
        Get remote focussing settings for xz-scan
        """
        try:
            # Get remote focussing parameter settings from GUI
            self.RF_settings = self.main.get_focus_settings()
            self.RF_settings['focus_mode_flag'] = 1
            self.RF_settings['is_xz_scan'] = 1

            # Set remote focusing flag to 1 and update AO_info
            self.AO_settings = {}
            self.AO_settings['loop_max'] = self.main.get_AO_loop_max()
            self.AO_settings['focus_enable'] = 1
            self.app.handle_AO_info(self.AO_settings)
            self.app.write_AO_info()

            # Update RF_info
            self.app.handle_focusing_info(self.RF_settings)
            self.app.write_focusing_info()

            return True

        except Exception as e:
            print(e)
            return False

    def exposed_start_RF(self): # Exposed method
        """
        Start remote focusing thread
        """
        try:
            self.app.handle_focus_start(AO_type = self.RF_settings['AO_type'])
            return True
        except Exception as e:
            print(e)
            return False

    def exposed_focusing_monitor(self, callback, interval = 0.000001): # Exposed method
        """
        xz scan line acquisition trigger monitor thread
        """
        try:
            self.interval = interval
            self.last_stat = None
            self.callback = rpyc.async_(callback) # Create an async callback
            self.active = True
            self.thread = Thread(target = self.work)
            self.thread.start()
        except Exception as e:
            print(e)

    def exposed_line_term(self):
        """
        Handles line termination trigger from scanning software
        """
        try:
            self.main.ui.serverSpin.setValue(0)
            return True
        except Exception as e:
            print(e)
            return False

    def exposed_stop(self): # Exposed method
        """
        Stop remote focusing and trigger monitor thread
        """
        try:
            self.active = False
            self.thread.join()
            self.app.stop_focus()
        except Exception as e:
            print(e)
  

class SERVER():
    """
    Server controller factory class, returns rpyc server instance.
    """
    def __init__(self):
        logger.info('Server factory class loaded.')

    @staticmethod
    def get(app, main, portNum = 18812):
        if portNum == 18812:
            try:
                server = ThreadedServer(RemoteService(app, main), port = portNum)
            except:
                logger.warning('Unable to initialise server instance.')
        else:
            server = None

        return server

