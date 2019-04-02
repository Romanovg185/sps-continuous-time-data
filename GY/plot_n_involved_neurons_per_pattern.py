import numpy as np
import matplotlib.pyplot as plt

def main():
    with open('n_patterns_vs_phi.txt') as f:
        for line in f:
            z = line
    l = line.split(', ')[:-1]
    x = np.array([float(i) for i in l])
    shifts = np.arange(-4, 4.01, 0.01)
    plt.plot(shifts, x)
    plt.xlabel(r'$\varphi$' + ' [sec]')
    plt.ylabel(r'Total number of significant firing events')
    plt.show()
    

if __name__ == "__main__":
    main()
