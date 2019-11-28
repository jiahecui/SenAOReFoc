# centroiding.py: 
	- check how the size of the S-H spots affect the effectiveness of the dynamic range and thresholding

# SHViewer.py:
	- change how SB layer and image layer updates (before nudging SB, SB layer is continuosly updated; after nudging SB, 
	   image layer is continuosly updated), could think about setting different handles in app.py

# calibration.py:
	- check what parameters are needed for centroiding and finish centroiding function here, and whether it is better to perform centroiding at the end or 	     while grabbing images
	- check whether image needs to be continuosly displayed for each voltage
	- write algorithm for acquiring influence function using Zernikes