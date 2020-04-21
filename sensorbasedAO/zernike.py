import numpy as np
import math

def zern(xx, yy, j):
    """
    Generates the value of Zernike polynomials for wavefront reconstruction at a particular pupil location

    Args:
        xx - normalised x value array
        yy - normalised y value array
        j - single index, ignores piston, starts with tip as 1
    """
    # Initialise instance variables
    count = 0
    val_sum = 0

    # Convert single index to double index form for both radial order and angular frequency
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))
    m = 2 * j - n * (n + 2)
    mabs = abs(m)

    # Calculate values of the jth Zernike polynomial
    for y in yy:
        for x in xx:

            if np.sqrt((x ** 2 + y ** 2)) <= 1:

                count += 1

                # Calculate rho and theta (add eps to prevent from dividing by zero)
                eps = 2 ** (-52)
                x = x + eps
                rho     = np.sqrt(x ** 2 + y ** 2)
                theta   = math.atan2(y, x)

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

                # Sum value for each element in search block
                val_sum += Z

    # Divide sum of zernike value by total number of counts:
    z_ave = val_sum / count

    return z_ave

def zern_gen(xx, yy, j):
    """
    Generates zernike phase profile for the jth zernike polynomial

    Args:
        xx - normalised x value meshgrid
        yy - normalised y value meshgrid
        j - single index, ignores piston, starts with tip as 1
    """
    # Convert single index to double index form for both radial order and angular frequency
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))
    m = 2 * j - n * (n + 2)
    mabs = abs(m)

    # Calculate rho and theta (add eps to prevent from dividing by zero)
    eps = 2 ** (-52)
    xx = (xx + eps) * (np.sqrt(xx ** 2 + yy ** 2) <= np.ones([np.shape(xx)[0], np.shape(xx)[0]]))
    rho     = np.sqrt(xx ** 2 + yy ** 2) * (np.sqrt(xx ** 2 + yy ** 2) <= np.ones([np.shape(xx)[0], np.shape(xx)[0]]))

    theta = np.zeros([np.shape(xx)[0], np.shape(xx)[1]])
    for k in range(np.shape(xx)[0]):
        for l in range(np.shape(xx)[1]):
            theta[k, l] = math.atan2(yy[k, l], xx[k, l]) * (np.sqrt(xx[k, l] ** 2 + yy[k, l] ** 2) <= 1)
            
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
                
def zern_diff(xx, yy, j, x_flag = True):
    """
    Generates the slopes of Zernike polynomials for wavefront reconstruction using Zernikes

    Args:
        xx - normalised x value array
        yy - normalised y value array
        j - single index, ignores piston, starts with tip as 1
        x_flag - flag for differentiating relative to x or y
    """
    # Initialise instance variables
    count = 0
    diff_sum = 0

    # Convert single index to double index form for both radial order and angular frequency
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))
    m = 2 * j - n * (n + 2)
    mabs = abs(m)

    # Calculate averaged derivative of the jth Zernike polynomial
    for y in yy:
        for x in xx:

            if np.sqrt((x ** 2 + y ** 2)) <= 1:

                count += 1

                # Calculate rho and theta (add eps to prevent from dividing by zero)
                eps = 2 ** (-52)
                x = x + eps
                rho     = np.sqrt(x ** 2 + y ** 2)
                theta   = math.atan2(y, x)

                # Calculate the derivatives of rho and theta
                if x_flag:
                    drho_dx     = x / rho
                    dtheta_dx   = -y / rho ** 2
                else:
                    drho_dx     = y / rho
                    dtheta_dx   = x / rho ** 2

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
                    Z  = N * Zn
                else:
                    Zn = (Rnm * dsin_dtheta * dtheta_dx) + (dRnm_drho * drho_dx * np.sin(m * theta))
                    Z  = -N * Zn

                # Sum derivative for each element in search block
                diff_sum += Z

    # Divide sum of derivative by total number of counts:
    z_ave = diff_sum / count

    return z_ave
