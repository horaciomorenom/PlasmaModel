import numpy as np
from matplotlib import pyplot as plt
import math
from typing import Optional, List
from matplotlib.animation import FuncAnimation
from scipy import interpolate
from sortedcontainers import SortedDict

class Particle:
    """
    A particle in a one-dimensional electrostatic cold stream plasma.

    Attributes:
    -----------
    id : int
        Label to preserve order of particles
    alpha : float
        The Lagrangian coordinate of the particle, ranging from 1:N
    x : float
        The position of the particle in the domain [0,1).
    v : float
        The velocity of the particle.
    """
    alpha: float
    x: float
    v: float
    a: float
    active: bool
    pos_hist: list
    vel_hist: list
    periods : int
    period_hist: list 

    def __init__(self, alpha_in: float, x0: float, v0: float, 
                 active: bool = True, periods_in: int = 0, num_iter: int = None):              
        """Initialzie a particle in plasma

        Args:
            alpha_in (float): Lagrangian coordinate of the particle
            x0 (float): Initial position
            v0 (float): Initial velocity
            active (bool, optional): Boolean that determines whether a particle is active
            . Defaults to True.
            periods_in (int, optional): Number of periods travelled by particle. Defaults to 0.
            num_iter (int, optional): Number of iterations that the particle has existed for. Defaults to None.
        """
        self.alpha = alpha_in
        self.x = x0
        self.v = v0
        self.a = 0
        self.active = active
        self.periods = periods_in

        if num_iter is None:
            self.pos_hist = []
            self.vel_hist = []
            self.period_hist = []
        else:
            self.pos_hist = [None] * (num_iter - 1)
            self.vel_hist = [None] * (num_iter - 1)
            self.period_hist = [None] * (num_iter - 1)
        
        self.pos_hist.append(self.x)
        self.vel_hist.append(self.v)
        self.period_hist.append(self.periods)
        

    def __lt__(self, other):
        """
        Compare two particles with respect to their Lagrangian Coordinates

        Parameters:
        -----------
        other : Particle
            The other particle that the current one will be compared to

        """
        return self.alpha < other.alpha
        

    def update_position(self, new_x):
        """
        Update the position of the particle based on its current velocity and 
        the elapsed time using Euler's Method.
        
        Args:
            dt (float): The elapsed time in seconds.
        
        Returns:
            None
        """
        if self.alpha != 0:
            self.x = new_x
        if self.x >= 1: self.periods += 1
        if self.x < 0: self.periods -= 1
        self.x = self.x % 1
        self.pos_hist.append(self.x)
        self.period_hist.append(self.periods)


    def update_velocity(self, new_v):
        """
        Update the velocity of the particle based on the given acceleration and
        the elapsed time using Euler's Method.
        
        Args:
            acceleration (float): The acceleration of the particle.
            dt (float): The elapsed time in seconds.
        
        Returns:
            None
        """
        if self.alpha != 0:
            self.v = new_v
        self.vel_hist.append(self.v)

    def update_acceleration(self, acceleration):
        """
        Update the acceleration of the particle based on the given acceleration
        function.
        
        Args:
            acceleration (float): The acceleration of the particle.
        
        Returns:
            None
        """
        self.a = acceleration(self)
        self.acc_hist.append(self.a)

def init_cold_pert(N: int, epsilon: float, 
                   active_only: bool = True) -> SortedDict:
    """Initialize a cold plasma with perturbation

    Args:
        N (int): Number of particles in the plasma
        epsilon (float): Magniture of perturbations
        active_only (bool, optional): Determines wether all particles are active or not. Defaults to True.

    Returns:
        SortedDict: Sorted dictionary with particle objects as items and lagrangian coordinates as keys
    """
    initial_plasma = SortedDict()

    for i in range(1, N+1):
        alpha = (i - 0.5) / N if active_only else (i - 1) / N
        x0 = alpha + epsilon * np.sin(2 * np.pi * alpha)
        v0 = 0

        if not active_only:
            is_active = i % 2 == 0
        else:
            is_active = True

        p = Particle(alpha, x0, v0, active = is_active)
        initial_plasma[alpha] = p
    
    return initial_plasma


class Plasma_Evolver:

    N: int
    dt: float
    epsilon: float
    delta: float
    ion_density: float = 1
    plasma : SortedDict
    t: float = 0
    kernel: callable
    green: callable
    
    def __init__(self, N_in: int, dt_in: float, epsilon_in: float = 0.05, 
                 delta_in: float = 0.002, insertion: bool = False, 
                 d1: float = None, d2: float = None, rk: bool = False):
        """Initializes an instance of PlasmaEvolver class with specified parameters

        Args:
            N_in (int): Number of starting particles in the plasma
            dt_in (float): Timestep parameter for integration
            epsilon_in (float, optional): Initial spatial perturbation parameter. Defaults to 0.05.
            delta_in (float, optional): E-field regularization parameter. Defaults to 0.002.
            insertion (bool, optional): True if adaptive particle insetion is used. Defaults to False.
            d1 (float, optional): Distance threshold parameter for particle insertion. Defaults to None.
            d2 (float, optional): Curvature threshold parameter for particle insertion. Defaults to None.
            rk (bool, optional): True for RK4 integration to be used, False for Euler's method. Defaults to False.
        """
        self.N0 = N_in
        self.N = N_in
        self.dt = dt_in
        self.epsilon = epsilon_in
        self.delta = delta_in
        self.weights = SortedDict()
        self.insertion = insertion
        self.rk = rk
        self.Ep_hist = []
        self.Ek_hist = []
        self.sym_hist = []
        self.ins_hist = []
        self.sym_profiles = []
        self.epsilon = epsilon_in
        self.current_t = 0

        self.d1 = 2/self.N if d1 is None else d1

        self.plasma = init_cold_pert(self.N, self.epsilon, not insertion)
        
        for p in self.plasma.values():
            if self.insertion:
                if not p.active: 
                    self.weights[p.alpha] = self.get_w(p)
            else:
                self.weights[p.alpha] = self.get_w(p)

        if not self.insertion:
            self.Ep_hist.append(self.calc_Ep())
            self.Ek_hist.append(self.calc_Ek())
        self.sym_hist.append(self.check_symmetry())
        self.ins_hist.append(0)

    def get_prev(self, p: Particle, same_type: bool = True) -> tuple:
        """Returns the previous particle in the self.plasma array. If same_type 
        is true, then it returns the first previous particle of the same type 
        (active/pasive). Assumes N is even 

        Args:
            p (Particle): reference particle
            same_type (bool, optional): same_type flag. Defaults to True.

        Returns:
            Particle: previous particle matching type if same_type is True
            wrap_around: flag to determine if next particle is on the other side
                of interval
        """

        if self.insertion: assert(self.N % 2 == 0)

        index = self.plasma.bisect_left(p.alpha)

        if same_type:
            prev_index = index - 2
        else:
            prev_index = index -1

        wrap_around = True if prev_index < 0 else False

        return list(self.plasma.values())[prev_index], wrap_around
    
    def get_next(self, p: Particle, same_type: bool = True) -> tuple:
        """Returns the next particle in the self.plasma array. If same_type 
        is true, then it returns the first next particle of the same type 
        (active/pasive). 
        Assumes N is even.

        Args:
            p (Particle): reference particle
            same_type (bool, optional): same_type flag. Defaults to True.

        Returns:
            Particle: next particle matching type if same_type is True
            wrap_around: flag to determine if next particle is on the other side of interval
        """

        if self.insertion: assert(self.N % 2 == 0)

        index = self.plasma.bisect_left(p.alpha)

        if same_type:
            prev_index = index + 2
        else:
            prev_index = index + 1

        wrap_around = True if (prev_index >= len(self.plasma.values())) else False
            
        prev_index = prev_index % len(self.plasma.values())

        return list(self.plasma.values())[prev_index], wrap_around

    def calc_next_dist(self, p: Particle) -> float:

        p_next, wrap = self.get_next(p)
        
        if not wrap:
            p_coords = np.array([p.x, p.v])
            p_next_coords = np.array([p_next.x, p_next.v])

            dist = np.linalg.norm(p_next_coords - p_coords)
        else:
            p_coords = np.array([p.x, p.v])
            p_next_coords = np.array([p_next.x + 1, p_next.v])

            dist = np.linalg.norm(p_next_coords - p_coords)

        return dist
       

    def get_active_weights(self):
        active_alphas = [p.alpha for p in self.plasma.values() if p.active]

        weights = np.array([self.weights[a] for a in active_alphas])

        return weights

    def calc_dist2chord(self, p1: Particle):

        p2, wrap2 = self.get_next(p1, same_type=False) # next particle
        p3, wrap3 = self.get_next(p1) # next particle of same type

        x1, y1 = p1.x, p1.v
        x2, y2 = p2.x, p2.v
        x3, y3 = p3.x, p3.v

        # Fix wrap around to ensure distances are right
        if wrap2:
            x2 += 1
        if wrap3:
            x3 += 1

        # Calculate the slope of the line connecting p1 and p3
        slope = (y3 - y1) / (x3 - x1)

        # Calculate the y-intercept of the line connecting p1 and p3
        intercept = y1 - slope * x1

        # Calculate the perpendicular distance from p2 to the line
        distance = abs(slope * x2 - y2 + intercept) / math.sqrt(slope ** 2 + 1)

        return distance
    
    def check_symmetry(self, tol=1e-3):
        pos = self.get_pos_array()
        vel = self.get_vel_array()
        

        if self.insertion:
            pos = np.delete(pos, 0)
            vel = np.delete(vel, 0)
            pos_left = pos[0:(pos.size // 2) + 1]
            pos_right = np.flip(pos[(pos.size // 2):] - 1)
    
            vel_left = vel[0:(vel.size // 2) + 1]
            vel_right = np.flip(vel[(vel.size // 2):])
        else:
            pos_left = pos[0:(pos.size // 2)]
            pos_right = np.flip(pos[(pos.size // 2):] - 1)
    
            vel_left = vel[0:(vel.size // 2)]
            vel_right = np.flip(vel[(vel.size // 2):])


        pos_sym = pos_left + pos_right
        vel_sym = vel_left + vel_right

        profile = pos_sym + vel_sym

        self.sym_profiles.append(profile)

        total = (np.sum(np.abs(pos_sym)) + np.sum(np.abs(vel_sym))) / self.N

        """ if total > 0.5: 
            print(pos_sym)
            print(vel_sym) """

        return total


        #if np.any(pos_sym > tol) or np.any(vel_sym > tol):
            #print("Symmetry broken! t = {}".format(self.current_t))
            #print(np.nonzero(pos_sym > tol))
            #print(np.nonzero(vel_sym > tol))
            #print(pos_sym)
    
    def insert_particles(self):
        
        # Get position and velocity arrays
        init_pos = self.get_pos_array()
        init_vel = self.get_vel_array()

        # Add periodic image of point at x=0
        pos = np.empty(len(init_pos) + 1, dtype=init_pos.dtype)
        pos[:len(init_pos)] = init_pos
        pos[len(init_pos)] = init_pos[0] + 1

        vel = np.empty(len(init_vel) + 1, dtype=init_vel.dtype)
        vel[:len(init_vel)] = init_vel
        vel[len(init_vel)] = init_vel[0]

        # Get spatial distances between every other pair of particles
        first_pos_diff = np.abs(np.diff(pos[::2]))
        # Consider alternate difference for points that have crossed periodic boundary
        alt_pos_diff = 1 - first_pos_diff

        # Make sure we get smallest distance between 2 points (account
        # for periodic boundaries)
        pos_diff = np.minimum(first_pos_diff, alt_pos_diff,)

        # Calculate velocity distances between passive particles
        vel_diff = np.diff(vel[::2])

        # Calculate phase space distance between passive particles
        total_diff = np.linalg.norm(np.array([pos_diff, vel_diff]).T, axis=1)

        # Calculate indices for particles that exceed threshold d1
        indices = np.transpose(np.argwhere(total_diff > self.d1))
        indices = 2 * indices

        # Track number of particles inserted this iteration
        self.ins_hist.append(np.size(indices))

        # Stop if no particles need to be inserted
        if np.size(indices) == 0:
            return

        # Save lagrangian coordinates for intervas that need insertion
        keys = np.array(self.plasma.keys())[indices[0]]    

        # Begin insertion for each interval
        for key in keys:
            # Get 3 particles in interval for insertion
            p1 = self.plasma[key]
            assert(not p1.active)
            p2, _ = self.get_next(p1, same_type=False)
            p3, wrap = self.get_next(p1, same_type=True)

            # Get lagrangian coordinates
            alphas = np.array([p1.alpha, p2.alpha, p3.alpha])
            # Make sure all alphas are together so interpolation works
            alphas[2] = p3.alpha if not wrap else p3.alpha + 1
            # Calculate new alphas for inserted particles
            a_left = 0.5 * (alphas[0] + alphas[1])
            a_right = 0.5 * (alphas[1] + alphas[2])

            # Get # of periods traversed by 3 ref particles
            periods = np.array([p1.periods, p2.periods, p3.periods])
            # Normalize so the array is one of the following: 
            #   (0,0,0), (1,0,0), (1,1,0), (0,1,1), (0,0,1)
            periods = periods - np.min(periods)

            # Get ref particle positions and fix w/ periods to ensure all 
            # particles are on the same interval
            x_vals = np.array([p1.x, p2.x, p3.x])
            
            # Ensure all particle positions are increasing
            if wrap: x_vals[2] += 1

            # Normalize all x_values to make sure they are continuous according to periods travelled
            x_vals = x_vals + periods
            # Get ref particle velocities
            v_vals = np.array([p1.v, p2.v, p3.v])

            # Get interpolating functions
            x_func = interpolate.interp1d(alphas, x_vals, kind='quadratic')
            v_func = interpolate.interp1d(alphas, v_vals, kind='quadratic')

            # Calculate new coordinates for inserted particles
            x_left = x_func(a_left)
            x_right = x_func(a_right)

            v_left = v_func(a_left)
            v_right = v_func(a_right)

            # Logic to determine # of periods travelled for inserted particles
            # to ensure proper visualization when plotting
            if periods[0] == 1:

                if periods[1] == 1:
                    # Case where p1, p2 are both one period ahead, 
                    # so left particle will also be on that period
                    l_period = p1.periods
                    # if new coordinate for right particle falls outside of [0,1)
                    # then set to period ahead. Otherwise p3 period
                    r_period = p2.periods if (x_right < 0 or x_right >=1) else p3.periods
                    #print("a")
                else:
                    # Only p1 is a period ahead
                    l_period = p1.periods if (x_left < 0 or x_left >=1) else p2.periods
                    r_period = p2.periods
                    #print("b")
            
            else:
                if periods[1] == 1:

                    if periods[2] == 1:
                        # Case where p2, p3 are both one period ahead, 
                        # so right particle will also be on that period 
                        l_period = p2.periods if (x_left < 0 or x_left >=1) else p1.periods
                        r_period = p2.periods
                    else:
                        # Only p3 is a period ahead 
                        l_period = p1.periods
                        r_period = p2.periods if (x_right < 0 or x_right >=1) else p3.periods
                else:
                    if periods[2] == 1:
                        l_period = p1.periods
                        r_period = p3.periods if (x_right < 0 or x_right >=1) else p2.periods
                    else:
                        # All particles on the same period
                        l_period = r_period = p1.periods 

            # Make sure new particle positions fall in interval [0,1)
            x_left = x_left % 1
            x_right = x_right % 1 

            # Create new particle objects with specified values
            p_left = Particle(a_left, x_left, v_left, periods_in=l_period, num_iter=len(p1.pos_hist))
            p_right = Particle(a_right, x_right, v_right, periods_in=r_period, num_iter=len(p1.pos_hist))

            # Set middle particle to passive
            self.plasma[alphas[1]].active = False

            # Add particles to plasma dictionary
            self.plasma[a_left] = p_left
            self.plasma[a_right] = p_right

            # Increase particle count
            self.N += 2

        # Recalculate quadrature weigths
        self.weights.clear()

        for p in self.plasma.values():
            if not p.active: self.weights[p.alpha] = self.get_w(p)
 
    def get_w(self, p: Particle) -> float:
        """Get quadrature weight of a given particle

        Args:
            p (Particle): reference particle

        Returns:
            float: quadrature weight value
        """

        p_next, wrap = self.get_next(p, same_type=self.insertion)

        if not wrap:
            return p_next.alpha - p.alpha
        else:
            return p_next.alpha - p.alpha + 1
        
    def get_pos_array(self, active_only = False) -> np.array:
        """Extract all particle positions from plasma dictionary

        Args:
            active_only (bool, optional): True if only active particle positions should be extracted. False for all other particles. Defaults to False.

        Returns:
            [np.array]: Array of particle positions in ascending order with respect to lagrangian coordinates
        """
        if active_only:
            return np.array([p.x for p in self.plasma.values() if p.active],
                            dtype=float)

        return np.array([p.x for p in self.plasma.values()],dtype=float)
    
    def get_vel_array(self)-> np.array:

        return np.array([p.v for p in self.plasma.values()],dtype=float)
    
    def update_particles(self, new_x: np.array, new_v: np.array) -> None:

        for i, p in enumerate(self.plasma.values()):
            p.update_position(new_x[i])
            p.update_velocity(new_v[i])

    def calc_Ek(self):

        vel = self.get_vel_array()
        w = np.array(self.weights.values(), dtype=float)
        Ek = 0.5 * np.sum(np.power(vel, 2))
        return Ek


    def calc_Ep(self):

        # Get positions
        pos = self.get_pos_array()
        w = self.weights.values()
        # Calculate absoolute value of differences between all positions
        abs_diff_matrix = - self.gd(pos[:, np.newaxis], pos[np.newaxis, :])
        square_diff_matrix = 0.5 * np.power(pos[:, np.newaxis] 
                                            - pos[np.newaxis, :], 2)
        potential = (abs_diff_matrix - square_diff_matrix) * w
        total_potential = np.sum(np.triu(potential, k=1))

        


        return total_potential


    def kd(self, p1: float, p2: float) -> float:

        p2 = p2 - p1
        p1 = 0 
    
        p1 = (p1 - 0.5) % 1 + 0.5
        p2 = (p2 - 0.5) % 1 + 0.5

        diff = p1 - p2
        c = np.sqrt(1 + 4*self.delta**2)

        if self.delta == 0:
            return 0.5 * np.sign(diff) - diff

        return np.where(diff == 0, 0, 0.5 * c * diff / 
                        np.sqrt(diff**2 + self.delta**2) - diff)


    def gd(self, p1: float, p2: float) -> float:

        diff = p1 - p2
        c = np.sqrt(1 + 4*self.delta**2)

        if self.delta == 0:
            return - 0.5 * np.abs(diff)
        else:
            return -c * 0.5 * np.sqrt(diff**2 + self.delta**2)
    
    def calc_Efield(self, num_samples = None):

        if num_samples is None:
            num_samples = self.N
        
        weights = np.array(self.weights.values(), dtype=float)
        particles = self.get_pos_array()
        x_arr = np.linspace(0, 1, num_samples, endpoint=False) - (1/(2*num_samples))

        # Broadcast positions to get a 2D array of values of Green's Function kd
        # over all particles. i,j entry corresponds to the values of kd(xi, xj) 
        # for i,j = 1:N
        k_map = np.array([self.kd(x_arr[:, np.newaxis], particles)])[0]
        # Multiply each row by the corresponding weights
        weighted_k_map = k_map * weights
        # Add up each row to get electric field force felt by each particle
        k_contribution = np.sum(weighted_k_map, axis=1)

        return -k_contribution, x_arr
    
    def calc_acceleration_reg(self, x_arr: np.array, v_arr: np.array) -> float:

        """
        Calculates the acceleration on a given particle due to rest of the stream
        with regularization

        Returns:
            float: acceleration
        """

        weights = self.weights.values()


        # Broadcast positions to get a 2D array of values of Green's Function kd
        # over all particles. i,j entry corresponds to the values of kd(xi, xj) 
        # for i,j = 1:N
        if self.insertion:
            k_map = np.array([self.kd(x_arr[:, np.newaxis], x_arr[1::2])])[0]
        else:
            k_map = np.array([self.kd(x_arr[:, np.newaxis], x_arr)])[0]
        # Multiply each row by the corresponding weights
        weighted_k_map = k_map * weights
        # Add up each row to get electric field force felt by each particle
        k_contribution = np.sum(weighted_k_map, axis=1)

        """ print("K map")
        print(k_map)
        print("\nweighted k map")
        print(weighted_k_map)
        print("\n Acceleration") """



        first = - k_contribution

        acc = first
 
        return acc
    
    def euler_update(self, fx: callable, fv: callable , x0: np.array, v0: np.array) -> tuple:

        dx = fx(x0, v0)
        dv = fv(x0, v0)

        new_x = x0 + dx * self.dt
        new_v = v0 + dv * self.dt

        return new_x, new_v
    
    def rk4_update(self, fx: callable, fv: callable , x0: np.array, v0: np.array) -> tuple:

        k1x = fx(x0, v0)
        k1v = fv(x0, v0)
        k2x = fx(x0 + 0.5 * self.dt * k1x, v0 + 0.5 * self.dt * k1v)
        k2v = fv(x0 + 0.5 * self.dt * k1x, v0 + 0.5 * self.dt * k1v)
        k3x = fx(x0 + 0.5 * self.dt * k2x, v0 + 0.5 * self.dt * k2v)
        k3v = fv(x0 + 0.5 * self.dt * k2x, v0 + 0.5 * self.dt * k2v)
        k4x = fx(x0 + self.dt * k3x, v0 + self.dt * k3v)
        k4v = fv(x0 + self.dt * k3x, v0 + self.dt * k3v)

        new_x = x0 + (1/6) * self.dt * (k1x + 2 * k2x + 2 * k3x + k4x)
        new_v = v0 + (1/6) * self.dt * (k1v + 2 * k2v + 2 * k3v + k4v)

        return new_x, new_v
    
    def evolve_plasma(self, time: float):
        """
        Evolves the plasma for the given amount of time

        Args:
            time (float): Time to evolve plasma over
        """

        self.t += time

        for _ in range(int(time / self.dt)):

            self.current_t += self.dt
            
            self.sym_hist.append(self.check_symmetry())

            x_arr = self.get_pos_array()
            v_arr = self.get_vel_array()

            if self.rk: new_x, new_v = self.rk4_update(lambda x,v: v, self.calc_acceleration_reg, x_arr, v_arr)
            else: new_x, new_v = self.euler_update(lambda x,v: v, self.calc_acceleration_reg, x_arr, v_arr)

            self.update_particles(new_x, new_v)
            
            if not self.insertion:
                self.Ep_hist.append(self.calc_Ep())
                self.Ek_hist.append(self.calc_Ek())

            if self.insertion: self.insert_particles()

            assert(np.all(np.array([p.active for p in self.plasma.values()])[1::2] == 1))
            
                
    def plot_particles(self,times: tuple = (-1,), periods: int = 1, zoom: bool = False, markers_on = True, 
                       lines_off = False, bigplot = False, midzoom: bool = False):
        """
        Plots particles stored in a sorted dictionary in phase space as points where
        the x-coordinate is position and the y-coordinate is velocity, and lines that 
        connect neighboring particles on the sorted dictionary.
        
        Args:
        - zoom (bool): boolean to zoom in into the center.
        
        Returns:
        - None
        """

        cmap = plt.get_cmap('tab10')

        colors = [cmap(i) for i in range(periods)]

        size = (5*periods,3.5*len(times)) if bigplot else (4*periods,2*len(times))

        if periods == 1 and len(times) == 1:
            size = (5,3)


        if zoom or midzoom:
            times = (times[-1],)
            periods = 1 if zoom else periods
            size = (8,6) if zoom else (8,8)

        fig, axs = plt.subplots(len(times), 1, figsize=size, dpi=150)
        

        # Convert axs to an iterable if it contains a single subplot
        axs = np.atleast_1d(axs)

        param_title = r'$N = {}$, $\delta = {}$, $\Delta t = {}$, $\varepsilon = {}$'.format(self.N, self.delta, self.dt, self.epsilon)

        if self.insertion: 
            param_title = r'$N_0 = {}$, '.format(self.N0) + param_title
            param_title += r', $d_1 = {}$'.format(self.d1)

        param_title = '(' + param_title + ')'

        title = 'Particle Phase Space ' + param_title if len(times) < 3 else 'Particle Phase Space' + '\n' + param_title
        

        if periods == 1 and len(times) == 1 and not zoom:
            fig.suptitle(title, fontsize=10)
        else: 
            fig.suptitle(title, fontsize=12)

        #fig.text(0.5, 0.01, 'Position', ha='center')
        #fig.text(0.04, 0.5, 'Velocity', va='center', rotation='vertical')
        


        for num, t in enumerate(times):

            if t == -1 or t > self.t:
                t = self.t
            
            index = int(t / self.dt)
            
            # Extract positions and velocities of particles
            positions = np.array([(p.pos_hist[index] + p.period_hist[index]) for p in self.plasma.values() if p.pos_hist[index] is not None])
            
            velocities = np.array([p.vel_hist[index] for p in self.plasma.values() if p.vel_hist[index] is not None])

            #positions = np.concatenate((np.array([positions[-1] - 1]), positions, np.array([positions[0] + 1])))
            #velocities = np.concatenate((np.array([velocities[-1]]), velocities, np.array([velocities[0]])))
            positions = np.concatenate((positions, np.array([positions[0] + 1])))
            velocities = np.concatenate((velocities, np.array([velocities[0]])))


            # Plot particles as points in phase space
            axs[num].set_xlim((0,periods))
            axs[num].set_ylim((-0.5,0.5))
            if t == times[-1]: axs[num].set_xlabel("Position")
            axs[num].set_ylabel("Velocity")

            if zoom or midzoom: 
                if zoom: axs[num].hlines(y=0, xmin=0, xmax=periods, linewidth = 0.5, color = 'r', linestyle = '--')
                visperiods = range(-3, periods + 3) if midzoom else range(-1, periods + 1)
                for j in visperiods:
                    if markers_on:
                        axs[num].plot(positions + j, velocities, marker='.', markersize=3,  alpha=1, linewidth=0.7)
                    else:
                        axs[num].plot(positions + j, velocities, alpha=1, linewidth=0.7)
                axs[num].set_title("t = " + str(t))
                if zoom:
                    axs[num].set_xlim((0.45, 0.55))
                    axs[num].set_ylim((-0.1, 0.1))
                if midzoom:
                    axs[num].set_xlim((0.5, 1.5))
                    axs[num].set_ylim((-0.4, 0.4))
                #axs[num].set_yticks((-0.1, -0.05, 0, 0.05, 0.1 ))

            else:

                # Add lines connecting neighboring particles on the sorted dictionary
                for j in range(-1, periods + 1):
                    if markers_on and not lines_off:
                        axs[num].plot(positions + j, velocities, marker='.', markersize=2.5,  alpha=1, linewidth=0.7)
                    elif lines_off: 
                        axs[num].scatter(positions + j, velocities, marker='.', s=1,  alpha=1)
                    else:
                        axs[num].plot(positions + j, velocities, alpha=1, linewidth=0.7)
                    axs[num].set_title("t = " + str(t))
                    if zoom: axs[num].hlines(y=0, xmin=0, xmax=periods, linewidth = 1, color = 'b', linestyle = '--')
                    """ for i, (k, p) in enumerate(self.plasma.items()):
                        if i == 0:
                            if j != 0:
                                prev_p = list(self.plasma.values())[-1]
                                print([prev_p.pos_hist[index] + j - 1, p.pos_hist[index] + j ])
                                #axs[num].plot([prev_p.pos_hist[index] + j - 1, p.pos_hist[index] + j ], 
                                #        [prev_p.vel_hist[index], p.vel_hist[index]], color=colors[j], alpha=0.3)
                            continue
                        prev_p = list(self.plasma.values())[i-1]
                        axs[num].plot([prev_p.pos_hist[index] + j + prev_p.period_hist[index], 
                                    p.pos_hist[index] + j + p.period_hist[index]], 
                                    [prev_p.vel_hist[index], p.vel_hist[index]], color=colors[j], alpha=0.3) """                
                
        fig.tight_layout()
        plt.show()

    def makeMovie(self,tmax: float, fileName: str):

        size = (8,4)

        fps = 30

        num_frames = tmax * fps

        fig, ax = plt.subplots(1, 1, figsize=size, dpi=250)

        ax.set_xlim((0,2))
        ax.set_ylim((-0.5,0.5))
        ax.set_xlabel("Position")
        ax.set_ylabel("Velocity")
        ax.set_title(r'$t = {}$'.format(0))

        param_title = r'$N = {}$, $\delta = {}$, $\Delta t = {}$, $\varepsilon = {}$'.format(self.N, self.delta, self.dt, self.epsilon)

        if self.insertion: 
            param_title = r'$N_0 = {}$, '.format(self.N0) + param_title
            param_title += r', $d_1 = {}$'.format(self.d1)

        param_title = '(' + param_title + ')'

        title =  'Particle Phase Space ' + param_title
        

        fig.suptitle(title)

        positions = np.array([(p.pos_hist[0] + p.period_hist[0]) for p in self.plasma.values() if p.pos_hist[0] is not None])
            
        velocities = np.array([p.vel_hist[0] for p in self.plasma.values() if p.vel_hist[0] is not None])

        positions = np.concatenate((positions, np.array([positions[0] + 1])))
        velocities = np.concatenate((velocities, np.array([velocities[0]])))

        period_2, = ax.plot(positions + -3, velocities, marker=None, alpha=0.8, linewidth=0.7, color='C6')
        period_1, = ax.plot(positions + -2, velocities, marker=None, alpha=0.8, linewidth=0.7, color='C4')
        period0, = ax.plot(positions + -1, velocities, marker=None, alpha=0.8, linewidth=0.7, color='C0')
        period1, = ax.plot(positions + 0, velocities, marker=None, alpha=0.8, linewidth=0.7, color='C1')
        period2, = ax.plot(positions + 1, velocities, marker=None, alpha=0.8, linewidth=0.7, color='C2')
        period3, = ax.plot(positions + 2, velocities, marker=None, alpha=0.8, linewidth=0.7, color='C3')
        period4, = ax.plot(positions + 3, velocities, marker=None, alpha=0.8, linewidth=0.7, color='C5')


        periods = [period0, period1, period2, period3, period_1, period4, period_2]

        fig.tight_layout()

        def update(frame):

            # Compute time in order to fit all data in one video of specified length
            t = int(round(frame*tmax / num_frames, 2) / self.dt)

            if t * self.dt <= tmax:

                ax.set_xlim((0,2))
                ax.set_ylim((-0.5,0.5))
                ax.set_xlabel("Position")
                ax.set_ylabel("Velocity")
                ax.set_title(r'$t = {}$'.format(round(frame / fps, 1)))


                positions = np.array([(p.pos_hist[t] + p.period_hist[t]) for p in self.plasma.values() if p.pos_hist[t] is not None])
                
                velocities = np.array([p.vel_hist[t] for p in self.plasma.values() if p.vel_hist[t] is not None])

                positions = np.concatenate((positions, np.array([positions[0] + 1])))
                velocities = np.concatenate((velocities, np.array([velocities[0]])))

                period_1.set_data(positions + -2, velocities)
                period_2.set_data(positions + -3, velocities)
                period0.set_data(positions + -1, velocities)
                period1.set_data(positions + 0, velocities)
                period2.set_data(positions + 1, velocities)
                period3.set_data(positions + 2, velocities)
                period4.set_data(positions + 3, velocities)

                for period in periods:

                    period.set_marker(None)
                    #period.set_markersize(0)
                    period.set_alpha(0.8)
                    period.set_linewidth(0.7)
                
                periods[0].set_color('C0')
                periods[1].set_color('C1')
                periods[2].set_color('C2')
                periods[3].set_color('C3')
                periods[4].set_color('C4')
                periods[5].set_color('C5')
                periods[6].set_color('C6')

                fig.tight_layout()

            
            # Return a tuple of the scatterplots to be updated
            return (*periods,)


        
        # Generate animation object
        anim = FuncAnimation(fig, update, frames=int(num_frames) + fps * 5, interval=int(1000/fps))

        # Save animation object
        anim.save(fileName, writer='ffmpeg',savefig_kwargs={"pad_inches":0})

        

