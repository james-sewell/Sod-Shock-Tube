# Shock tube pressure ratio convergence
""" A newton's method convergence used to determine the pressure ratio p2/p1 from a normal shock wave based on
the ratio p4/p1 which is the pressure ratio between the driver and driven sections of a shocktube."""

import numpy as np
import sys
import matplotlib.pyplot as plt


def shock_press_ratio(p4_p1, p_in, gamma_4, gamma_1, r1, r4, t4, t1):
    """
    The function is based on equation .. from Introduction to Compressible Fluids
    pressure ratio p2/p1 for in front and behind a normal shock
    :param p4_p1: the pressure ratio between the driver and driven section
    :param p_in: the initial guess for the pressure ratio on either side of the shock, used for convergence
    :param gamma_4: the ratio of specific heat at constant volume to the specific heat at constant pressure
    (Cv/Cp) for the driver section
    :param gamma_1: the ratio of specific heat at constant volume to the specific heat at constant pressure
    (Cv/Cp) for the driven section
    :param r1:  driver section molecular gas constant (J/(kg*K))
    :param r4:  driven section molecular gas constant (J/(kg*K))
    :param t4:  driver section start temperature (K)
    :param t1: driven section start temperature (K)
    :return: p2_p1
    """
    a1 = np.sqrt(gamma_1 * r4 * t1)
    a4 = np.sqrt(gamma_4 * r1 * t4)
    spsnd_ratio = a1 / a4
    gamma_diff = gamma_4 - 1
    dbl_gamma = 2 * gamma_1
    c1 = gamma_diff * spsnd_ratio / dbl_gamma
    c2 = (gamma_1 - 1) / dbl_gamma
    c3 = 2 * gamma_4 / gamma_diff
    p_in_diff = p_in -1
    ex_sec_num = 1 - c1 * p_in_diff
    ex_sec_den = 1 + c2 * p_in_diff
    return p4_p1 * (1 - ex_sec_num / np.sqrt(ex_sec_den))**c3


def dfdp(p, delta_p):
    return 2 * delta_p


def newton(f, dfdx, x, eps):
    f_value = f(x)
    iteration_counter = 0
    while np.abs(f_value) > eps and iteration_counter < 100:
        try:
            x = x - f_value/dfdx(x)
        except ZeroDivisionError:
            print("Error! - derivative zero for x = ", x)
            sys.exit(1)     # Abort with error

        f_value = f(x)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(f_value) > eps:
        iteration_counter = -1
    return x, iteration_counter


solution, num_iterations = newton(f, dfdx, x=1000, eps=1.0e-6)

if num_iterations > 0:    # Solution found
    print("Number of function calls: %d" % (1 + 2*no_iterations))
    print("A solution is: %f" % (solution))
else:
    print("Solution not found!")
