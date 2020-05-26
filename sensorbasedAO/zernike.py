import numpy as np
import math

def zern(xx_orig, yy_orig, j):
    """
    Generates the value of Zernike polynomials for wavefront reconstruction at a particular pupil location

    Args:
        xx - normalised x value meshgrid
        yy - normalised y value meshgrid
        j - single index, ignores piston, starts with tip as 1
    """
    # Choose coordinates within unit circle
    xx = xx_orig * (np.sqrt(xx_orig ** 2 + yy_orig ** 2) <= np.ones([np.shape(xx_orig)[0], np.shape(yy_orig)[0]]))
    yy = yy_orig * (np.sqrt(xx_orig ** 2 + yy_orig ** 2) <= np.ones([np.shape(xx_orig)[0], np.shape(yy_orig)[0]]))

    # Convert single index to double index form for both radial order and angular frequency
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))
    m = 2 * j - n * (n + 2)
    mabs = abs(m)

    # Calculate rho and theta (add eps to prevent from dividing by zero)
    eps = 2 ** (-52)
    xx = xx + eps
    rho     = np.sqrt(xx ** 2 + yy ** 2) + eps
    theta   = np.arctan2(yy, xx) + eps

    # Calculate the radial dependent component and derivative
    Rnm = 0
    for s in range((n - mabs) // 2 + 1):

        constant = ((-1) ** s * math.factorial(n - s)) / (math.factorial(s) * math.factorial((n + mabs) / 2 - s) * math.factorial((n - mabs) / 2 - s))
        Rnm      = Rnm + constant * rho ** (n - 2 * s)

    # Calculate normalisation constant
    if m == 0:
        N = np.sqrt(n + 1)
    else:
        N = np.sqrt(2 * (n + 1))

    # Final result
    if m >= 0:
        Z = np.mean(N * Rnm * np.cos(m * theta))
    else:
        Z = -np.mean(N * Rnm * np.sin(m * theta))

    return Z

def zern_gen(xx_orig, yy_orig, j):
    """
    Generates zernike phase profile for the jth zernike polynomial

    Args:
        xx - normalised x value meshgrid
        yy - normalised y value meshgrid
        j - single index, ignores piston, starts with tip as 1
    """
    # Choose coordinates within unit circle
    xx = xx_orig * (np.sqrt(xx_orig ** 2 + yy_orig ** 2) <= np.ones([np.shape(xx_orig)[0], np.shape(yy_orig)[0]]))
    yy = yy_orig * (np.sqrt(xx_orig ** 2 + yy_orig ** 2) <= np.ones([np.shape(xx_orig)[0], np.shape(yy_orig)[0]]))

    # Convert single index to double index form for both radial order and angular frequency
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))
    m = 2 * j - n * (n + 2)
    mabs = abs(m)

    # Calculate rho and theta (add eps to prevent from dividing by zero)
    eps = 2 ** (-52)
    xx = xx + eps
    rho     = np.sqrt(xx ** 2 + yy ** 2) + eps
    theta   = np.arctan2(yy, xx) + eps
            
    # Calculate the radial dependent component and derivative
    Rnm = 0
    for s in range((n - mabs) // 2 + 1):

        constant = ((-1) ** s * math.factorial(n - s)) / (math.factorial(s) * math.factorial((n + mabs) / 2 - s) * math.factorial((n - mabs) / 2 - s))
        Rnm      = Rnm + constant * rho ** (n - 2 * s)

    # Calculate normalisation constant
    if m == 0:
        N = np.sqrt(n + 1)
    else:
        N = np.sqrt(2 * (n + 1))

    # Final result
    if m >= 0:
        Z = N * Rnm * np.cos(m * theta)
    else:
        Z = -N * Rnm * np.sin(m * theta)
            
    return Z
                
def zern_diff(xx_orig, yy_orig, j, x_flag = True):
    """
    Generates the slopes of Zernike polynomials for wavefront reconstruction using Zernikes

    Args:
        xx - normalised x value meshgrid
        yy - normalised y value meshgrid
        j - single index, ignores piston, starts with tip as 1
        x_flag - flag for differentiating relative to x or y
    """
    # Choose coordinates within unit circle
    xx = xx_orig * (np.sqrt(xx_orig ** 2 + yy_orig ** 2) <= np.ones([np.shape(xx_orig)[0], np.shape(yy_orig)[0]]))
    yy = yy_orig * (np.sqrt(xx_orig ** 2 + yy_orig ** 2) <= np.ones([np.shape(xx_orig)[0], np.shape(yy_orig)[0]]))

    # Convert single index to double index form for both radial order and angular frequency
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))
    m = 2 * j - n * (n + 2)
    mabs = abs(m)

    # Calculate rho and theta (add eps to prevent from dividing by zero)
    eps = 2 ** (-52)
    xx = xx + eps
    rho     = np.sqrt(xx ** 2 + yy ** 2) + eps
    theta   = np.arctan2(yy, xx) + eps

    # Calculate the derivatives of rho and theta
    if x_flag:
        drho_dx     = xx / rho
        dtheta_dx   = -yy / rho ** 2
    else:
        drho_dx     = yy / rho
        dtheta_dx   = xx / rho ** 2

    # Calculate the radial dependent component and derivative
    Rnm         = 0
    dRnm_drho   = 0
    
    for s in range((n - mabs) // 2 + 1):

        constant    = ((-1) ** s * math.factorial(n - s)) / (math.factorial(s) * math.factorial((n + mabs) / 2 - s) * math.factorial((n - mabs) / 2 - s))
        Rnm         = Rnm + constant * rho ** (n - 2 * s)
        dRnm_drho   = dRnm_drho + constant * (n - 2 * s) * rho ** (n - 2 * s - 1)

    # Calculate the angular dependent derivative
    dcos_dtheta = -m * np.sin(m * theta)
    dsin_dtheta = m * np.cos(m * theta)

    # Calculate normalisation constant
    if m == 0:
        N = np.sqrt(n + 1)
    else:
        N = np.sqrt(2 * (n + 1))

    # Final result using chain rule and differentiation of a complex function
    if m >= 0:
        Zn = (Rnm * dcos_dtheta * dtheta_dx) + (dRnm_drho * drho_dx * np.cos(m * theta))
        Z  = np.mean(N * Zn)
    else:
        Zn = (Rnm * dsin_dtheta * dtheta_dx) + (dRnm_drho * drho_dx * np.sin(m * theta))
        Z  = -np.mean(N * Zn)

    return Z
