import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../shocktubecalc')
import sod


def export_values(fname, vals):

    keys = vals.keys()
    arr = np.array([vals[key] for key in keys]).T
    header = " ".join('{:^17}'.format(str(key)) for key in keys)
    np.savetxt(fname, arr, header=header, fmt='% 1.10e')


def main():
    gamma = 1.4
    positions, regions, values = sod.solve(left_state=(1, 1, 0), right_state=(0.125, 0.1, 0.),
                                           geometry=(-1., 1., 0), t=0.2, gamma=gamma, npts=500)

    # print positions and states
    print('Positions:')
    for desc, vals in positions.items():
        print('{0:10} : {1}'.format(desc, vals))

    print('States:')
    for desc, vals in regions.items():
        print('{0:10} : {1}'.format(desc, vals))

    # calculate other state variables
    e = values['p'] / (gamma - 1) / values['rho']
    E = values['p']/(gamma-1.) + 0.5*values['rho']*values['u']**2
    T = values['p'] / values['rho']
    c = np.sqrt(gamma * values['p'] / values['rho'])
    M = values['u'] / c
    h = e + values['p']/values['rho']

    # export calculated values only
    export_values(fname='sod_calculated_only.txt', vals=values)

    # export all values
    values['e'] = e
    values['E'] = E
    values['c'] = c
    values['M'] = M
    values['T'] = T
    values['h'] = h
    export_values(fname='all_values.txt', vals=values)

    # plot values
    f, axarr = plt.subplots(4, 2, sharex='col')

    axarr[0, 0].plot(values['x'], values['p'], linewidth=1.5)
    axarr[0, 0].set_ylabel(r'$p$')
    axarr[0, 0].set_xlabel(r'$x$')

    axarr[0, 1].plot(values['x'], values['rho'], linewidth=1.5)
    axarr[0, 1].set_ylabel(r'$\rho$')
    axarr[0, 1].set_xlabel(r'$x$')

    axarr[1, 0].plot(values['x'], values['u'], linewidth=1.5)
    axarr[1, 0].set_ylabel(r'$u$')
    axarr[1, 0].set_xlabel(r'$x$')

    axarr[1, 1].plot(values['x'], M, linewidth=1.5)
    axarr[1, 1].set_ylabel(r'$M$')
    axarr[1, 1].set_xlabel(r'$x$')

    axarr[2, 0].plot(values['x'], e, linewidth=1.5)
    axarr[2, 0].set_ylabel(r'$e$')
    axarr[2, 0].set_xlabel(r'$x$')

    axarr[2, 1].plot(values['x'], E, linewidth=1.5)
    axarr[2, 1].set_ylabel(r'$E$')
    axarr[2, 1].set_xlabel(r'$x$')

    axarr[3, 0].plot(values['x'], h, linewidth=1.5)
    axarr[3, 0].set_ylabel(r'$h$')
    axarr[3, 0].set_xlabel(r'$x$')

    axarr[3, 1].plot(values['x'], T, linewidth=1.5)
    axarr[3, 1].set_ylabel(r'$T$')
    axarr[3, 1].set_xlabel(r'$x$')

    plt.show()


if __name__ == '__main__':
    main()
