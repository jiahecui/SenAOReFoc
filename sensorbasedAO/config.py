import os
import sys
import yaml

# Set config file path depending on if bundled app (pyinstaller)
if getattr(sys, 'frozen', False):
    config_path = os.path.dirname(sys.executable)
else:
    config_path = os.path.dirname(__file__)

with open(os.path.join(config_path, "config.yaml"), "r") as stream:
    config = yaml.load(stream)

# Generated options
config['general'] = {}
config['general']['module_path'] = os.path.dirname(__file__)