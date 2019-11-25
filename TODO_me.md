# SB_geometry.py: 
	- check why some geometries aren't symmetrical

# centroiding.py: 
	- check centroiding with single bright pixel
	- change dynamic range algorithm to shift + shrink and how it improves accuracy for single bright pixel
	- check thresholding by adding noise to simulated S-H spot image (random value within a range)

# SHViewer.py:
	- change how SB layer and image layer updates (before nudging SB, SB layer is continuosly updated; after nudging SB, 
	   image layer is continuosly updated), could think about setting different handles in app.py

# calibration.py:
	- check what parameters are needed for centroiding here
	- check whether image needs to be continuosly displayed for each voltage