# Main GUI
- Move GUI components into separate ui files
- Status line in main gui
- Easier load
- error notifications
- offset and size for GUI
- Set up main GUI and children on ui namespace eg. app.ui['scan_settings']
- dynamic GUI:
    - disable tabs based on hardware present
- Consider moving direct hardware calls out of GUI code
- Could define component class as abc with ui/widget/get_settings etc as attributes

# Scan GUI
- Put advanced settings in own box
- Hide/show buttons based on scan type

# Data handling
- toggle for pinned images - these could be moved to own list
- memory usage indicator
- image binning (inc from sampling rate)
- HDF read/write

# Image entity
- stores gui settings?
- support colour images
- Set up properties for image array

# Image viewer
- Support colour images
- rescale and invert for voltage scale

# FPGA
- set up timeout in read method to account for no data available

# Galvo
- work out microns per volt
- better interface

# Scan
- check for scan outside bounds
- Return actual master/pixel rates
- Multiple scan types:
    - slow z
    - fast z

## Live scan
- Consider: store stack on live worker and query from GUI, rather than repeated signals

# Error handling
- Add error classes
- Add user feedback for errors

# Testing
- Mock out devices
- Add test framework

# Misc ideas
- could implement GUI as framework using abcs

# Bugs