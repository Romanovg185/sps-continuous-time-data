import numpy as np
import matplotlib.pyplot as plt
from convolve_with_kernel import get_indices_significant_overlap, convolve_with_kernel

def main():
    def plot1():
        scope1_raw = np.loadtxt('Scope1_raw.csv', delimiter=',') # sh[cell, ts]
        timecourse = 1/30*np.arange(scope1_raw.shape[1])
        for i, el in enumerate(scope1_raw):
            plt.plot(timecourse, el + i, c='C0', lw=0.1)
        plt.show()

    def plot2():
        scope1_raw = np.loadtxt('Scope1_raw.csv', delimiter=',') # sh[cell, ts]
        timecourse = 1/30*np.arange(scope1_raw.shape[1])
        du2dt2 = (scope1_raw[:, :-2] - 2*scope1_raw[:, 1:-1] + scope1_raw[:, 2:])/(timecourse[1] - timecourse[0])
        for i in du2dt2:
            plt.plot(timecourse[1:-1], i, c='C1', lw=0.1)
        plt.show()

    """
    Plot all areas considered to be peaks. If the sum looks like a peak, but it is not shaded, the most likely hypothesis is that it is a summation of simple spikes (i.e. spikes not detected by FRI)
    """
    def plot3():
        scope1_raw = np.loadtxt('Scope1_raw.csv', delimiter=',') # sh[cell, ts]
        scope1_sum = np.sum(scope1_raw, axis=0)
        timecourse = 1/30*np.arange(scope1_raw.shape[1])
        plt.plot(timecourse, scope1_sum)
        scope1_patterns = np.loadtxt('Scope1_denoised_mc_results.csv', delimiter=',')
        intervals = convolve_with_kernel(scope1_patterns)
        print(len(intervals))
        print(intervals)
        for waste, trash, start, end in intervals:
            time_mask = np.logical_and(timecourse < end, timecourse >= start)
            plt.fill_between(timecourse[time_mask], scope1_sum[time_mask], np.zeros_like(scope1_sum)[time_mask], color='C1')
        plt.show()

    def plot4():
        scope1_raw = np.loadtxt('Scope1_raw.csv', delimiter=',') # sh[cell, ts]
        scope1_sum = np.sum(scope1_raw, axis=0)
        timecourse = 1/30*np.arange(scope1_raw.shape[1])
        scope1_patterns = np.loadtxt('Scope1_denoised_mc_results.csv', delimiter=',')
        plt.plot(timecourse, scope1_sum, label='Sum of 213 transients')
        intervals = get_indices_significant_overlap(scope1_patterns)
        for start, end in intervals:
            time_mask = np.logical_and(timecourse < 0.001*end, timecourse >= 0.001*start)
            plt.fill_between(timecourse[time_mask], scope1_sum[time_mask], np.zeros_like(scope1_sum)[time_mask], color='C1')
        plt.fill_between(timecourse[time_mask], scope1_sum[time_mask], np.zeros_like(scope1_sum)[time_mask], color='C1', label='Peak found to be non-spurious')
        plt.title('Scope 1')
        plt.xlabel('Time [sec]')
        plt.ylabel(r'$\frac{\Delta F}{F}$', fontsize=20)
        plt.show()

    def plot5():
        scope1_raw = np.loadtxt('Scope1_raw.csv', delimiter=',') # sh[cell, ts]
        scope1_sum = np.sum(scope1_raw, axis=0)
        timecourse = 1/30*np.arange(scope1_raw.shape[1])
        scope1_patterns = np.loadtxt('Scope1_denoised_mc_results.csv', delimiter=',')
        intervals = get_indices_significant_overlap(scope1_patterns)
        fig = plt.figure()
        figure = fig.add_subplot(111)
        for start, end in intervals:
            i = 0
            time_mask = np.logical_and(timecourse < 0.001*end, timecourse >= 0.001*start)
            for cell in scope1_patterns.T: #Iterate per cell
                if np.sum(np.logical_and(cell >= 0.001*start, cell < 0.001*end)):
                    i += 1
            figure.bar(timecourse[time_mask], i, color='C0', edgecolor='C0')
        figure.bar(timecourse[time_mask], i, color='C0', edgecolor='C0', label='Number of cells in synchronous pattern')
        figure.plot(timecourse, scope1_sum, c='C1', label='Sum of transients')
        plt.title('Scope 1')
        plt.xlabel('Time [sec]')
        plt.ylabel(r'$\frac{\Delta F}{F}$', fontsize=20)
        plt.legend()
        plt.show()

    def plot6():
        scope1_raw = np.loadtxt('Scope2_raw.csv', delimiter=',') # sh[cell, ts]
        scope1_sum = np.sum(scope1_raw, axis=0)
        timecourse = 1/30*np.arange(scope1_raw.shape[1])
        scope1_patterns = np.loadtxt('Scope2_denoised_mc_results.csv', delimiter=',')
        intervals = get_indices_significant_overlap(scope1_patterns)
        fig = plt.figure()
        figure = fig.add_subplot(111)
        for start, end in intervals:
            time_mask = np.logical_and(timecourse < 0.001*end, timecourse >= 0.001*start)
            for ts in timecourse[time_mask]:
                i = 0
                for cell in scope1_patterns.T: #Iterate per cell
                    if np.sum(np.abs(cell - ts) < 0.1): # If at least 1 firing event is closer than 0.1 sec to the time point
                        i += 1
                figure.bar(ts, i, color='C0', edgecolor='C0')
        plt.title('Scope 2')
        plt.xlabel('Time [sec]')
        plt.ylabel('Number of cells partaking in pattern')
        plt.show()

    """
    Plot matrices but not with events on x-axis, but ts
    """
    def plot7():
        scope1_raw = np.loadtxt('Scope2_raw.csv', delimiter=',') # sh[cell, ts]
        scope1_sum = np.sum(scope1_raw, axis=0)
        timecourse = 1/30*np.arange(scope1_raw.shape[1])
        scope1_patterns = np.loadtxt('Scope2_denoised_mc_results.csv', delimiter=',')
        intervals = get_indices_significant_overlap(scope1_patterns)
        fig = plt.figure()
        figure = fig.add_subplot(111)
        total = []
        for start, end in intervals:
            time_mask = np.logical_and(timecourse < 0.001*end, timecourse >= 0.001*start)
            for ts in timecourse[time_mask]:
                l = np.zeros(scope1_patterns.shape[1])
                for i, cell in enumerate(scope1_patterns.T): #Iterate per cell
                    if np.sum(np.abs(cell - ts) < 0.1): # If at least 1 firing event is closer than 0.1 sec to the time point
                        l[i] = 1
                total.append(l)
        m_tot = np.vstack(total)
        m_tot = m_tot.T
        plt.imshow(m_tot)
        plt.title('Scope 2')
        plt.xlabel('Time step')
        plt.ylabel('Cell id')
        plt.show()

    plot7()



if __name__ == "__main__":
    main()
