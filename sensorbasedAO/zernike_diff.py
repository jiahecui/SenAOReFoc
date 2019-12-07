import numpy as np
import math

def zern_diff(xx, yy, j, x_flag = True):
    """
    Generates the slopes of Zernike polynomials for wavefront reconstruction using Zernikes

    Args:
        xx - x value array
        yy - y value array
        j - single index, ignores piston, starts with tip as 1
        x_flag - flag for differentiating relative to x or y
    """
    # Initialise instance variables
    count = 0
    diff_sum = 0

    # Convert single index to double index form for both radial order and angular frequency
    n = int(np.ceil((-3 + np.sqrt(9 + 8 * j)) / 2))
    m = int(abs(2 * j - n * (n + 2)))

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
                Rnm         = np.zeros(np.size(rho))
                dRnm_drho   = np.zeros(np.size(rho))
                
                for s in range((n - m) // 2):

                    constant    = ((-1) ** s * math.factorial(n - s)) / (math.factorial(s) * math.factorial((n + m) / 2 - s) * math.factorial((n - m) / 2 - s))
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

    # if x_flag:
    #     print('Averaged derivative for the {}th polynomial is {}'.format(j, z_ave))
           
    return z_ave