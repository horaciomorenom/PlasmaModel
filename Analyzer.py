import numpy as np
from sortedcontainers import SortedDict
from matplotlib import pyplot as plt
from Plasma import Particle, Plasma_Stream, Plasma_Evolver
from math import floor, ceil
from scipy.fft import fft, fftfreq

class Plasma_Analyzer:

    def __init__(self, evolver: Plasma_Evolver):

        self.evolver = evolver
        self.dt = evolver.dt


    def calc_charge_dist(self, time: float = None, num: float = None):

        time = self.evolver.t if time is None else time
        num = self.evolver.N0 if num is None else num

        index = int(time / self.dt)

        charge_dists = []

        for plasma in self.evolver.plasma:
            pos_charge_pairs = np.array([(p.pos_hist[index] + p.period_hist[index], p.charge_hist[index])
                                         for p in plasma.stream.values() if p.pos_hist[index] is not None])

            minimum = floor(np.min(pos_charge_pairs.T[0]))
            maximum = ceil(np.max(pos_charge_pairs.T[0]))
            num_intervals = (maximum - minimum) * num
            charge_sum_array = [0] * num_intervals

            interval_size = (maximum - minimum) / num_intervals

            intervals = [(maximum - minimum) * (i / num_intervals) + minimum for i in range(num_intervals)]

            for position, charge in pos_charge_pairs:

                interval_index = min(int((position - minimum) / interval_size), num_intervals)

                charge_sum_array[interval_index] += charge
            
            

            charge_dists.append([intervals, charge_sum_array])

        return charge_dists

    def get_particle_pos(self, index: int, stream: int):

        chosen_stream = self.evolver.plasma[stream]
        key = chosen_stream.stream.keys()[index]
        p = chosen_stream.stream[key]

        pos = np.array(p.pos_hist)

        return pos
    
    def get_particle_vel(self, index: int, stream: int):

        chosen_stream = self.evolver.plasma[stream]
        key = chosen_stream.stream.get_keys()[index]
        p = chosen_stream.stream[key]

        pos = np.array(p.vel_hist)

        return pos

class Plasma_Plotter:

    def __init__(self, evolver: Plasma_Evolver):

        self.evolver = evolver
        self.dt = evolver.dt

    def make_fig_title(self, main_title: str, line_break: bool = False) -> str:

        title = r'$N = {}$, $\delta = {}$, $\Delta t = {}$, $\varepsilon = {}$'.format(self.evolver.N, self.evolver.delta, self.evolver.dt, self.evolver.epsilon)

        if self.evolver.insertion: 
            title = r'$N_0 = {}$, '.format(self.evolver.N0) + title
            title += r', $d_1 = {}$'.format(self.evolver.d1)

        title = '(' + title + ')'

        title = main_title + ' ' + '\n' + title if line_break else main_title + title 

        return title
    
    def get_phase_space_data(self, index):

        data = []

        for plasma_stream in self.evolver.plasma:

                # Extract positions and velocities of particles
                positions = np.array([(p.pos_hist[index] + p.period_hist[index]) for p in plasma_stream.stream.values() if p.pos_hist[index] is not None])
                velocities = np.array([p.vel_hist[index] for p in plasma_stream.stream.values() if p.vel_hist[index] is not None])

                # Add last particle to make sure streams connect
                positions = np.concatenate((positions, np.array([positions[0] + 1])))
                velocities = np.concatenate((velocities, np.array([velocities[0]])))

                data.append((positions, velocities))
        
        return data
    
    def init_phase_space_axis(self, ax: plt.Axes, title, x_label, y_label, 
                              x_lim, y_lim, title_fontsize, label_fontsize):
        
        ax.set(xlim=x_lim, ylim=y_lim)
        ax.set_title(title, fontsize=title_fontsize)
        ax.set_xlabel(x_label, fontsize=label_fontsize)
        ax.set_ylabel(y_label, fontsize=label_fontsize)
        
    
    def populate_phase_space_axis(self, ax: plt.Axes, 
                                  x_arr: np.array, v_arr: np.array,
                                  color = None, 
                                  lines: bool = True, 
                                  markers: bool = True, 
                                  marker_size: float = 5.0):
        
        if lines and not markers:
            ax.plot(x_arr, v_arr, alpha=1, c=color, linewidth=1.5)
            return
        if not lines and markers:
            ax.scatter(x_arr, v_arr, marker='.', s=marker_size, c=color, alpha=1)
            return 
        ax.plot(x_arr, v_arr, marker='.', markersize=marker_size, c=color, alpha=1, 
                linewidth=1.5)
        




    def plot_phase_space(self, periods: int = 1, times: tuple = None, 
                         x_lims: tuple = None, y_lims: tuple = None, 
                         markers_on: True = False, lines_on: bool = True, 
                         marker_size: float = 5.0, coloring: str = 'default'):

        assert(markers_on or lines_on)

        times = (self.evolver.t,) if times is None else times
        times = [t if t <= self.evolver.t else self.evolver.t for t in times]

        num_axes = len(times)

        width = 10 + 2 * (periods >= 3)
        height = 3 * num_axes

        if x_lims is None and y_lims is None: # Default window
            if len(times) == 1: # Only 1 time plotted
                width = 6 if periods == 1 else width
                height = 4
        else:
            periods = 1
            times = (times[-1],)
            width = 8
            height = 6

        x_lims = (0, periods) if x_lims is None else x_lims
        y_lims = (-0.3, 0.3) if y_lims is None else y_lims

        fig, axs = plt.subplots(num_axes, 1, figsize=(width, height), dpi=150)
        axs = np.atleast_1d(axs)

        fig.suptitle(self.make_fig_title("Particle Phase Space"))

        for ax, t in zip(axs, times):
            self.init_phase_space_axis(ax, r'$t = {}$'.format(t), "Position", "Velocity", x_lims,
                                       y_lims, 12, 10)

        indices = (int(t / self.evolver.dt)  for t in times)

        print(indices)
        
        colors = ('b', 'r')
        

        for i, t_index in enumerate(indices):
            data = self.get_phase_space_data(t_index)

            for stream_data, color in zip(data,colors):

                stream_pos = stream_data[0]
                stream_vel = stream_data[1]

                stream_min = np.min(stream_pos)
                stream_max = np.max(stream_pos)

                print(stream_min)
                print(stream_max)

                lower_floor = floor(np.min(stream_pos))
                lower_ceil = ceil(np.min(stream_pos))
                upper_floor = floor(np.max(stream_pos))
                upper_ceil = ceil(np.max(stream_pos))

                lower_comp = 0
                while lower_comp + stream_min > x_lims[0]:
                    lower_comp -= 1
                
                upper_comp = 0
                while upper_comp + stream_max < x_lims[1]:
                    upper_comp += 1

                #print(f'lower = {lower}, upper = {upper}, lc = {lower_comp}, uc = {upper_comp}')

                period_copies = range(lower_comp,upper_comp + 1)

                print(period_copies)

                period_copies = sorted(period_copies, key=lambda x: abs(x))


                for p in period_copies:
                    self.populate_phase_space_axis(axs[i], stream_pos + p, 
                                                   stream_vel,
                                                   lines = lines_on,
                                                   color=color,
                                                   markers=markers_on, 
                                                   marker_size=marker_size)

        fig.tight_layout()
        plt.show()



        
