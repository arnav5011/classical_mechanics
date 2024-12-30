import numpy as np
from scipy import special as sc
from math import factorial
from matplotlib.animation import FuncAnimation
from matplotlib import pyplot as plt

def bessel_series(n,x,terms = 20):
    """Use a series approximation for small x, and small n.
    Use this function to evaluate bessel function for small x"""
    J = 0 # Initialize Bessel Function for given node n and point x
    for m in range(terms):
        a = (-1)**m
        b = 1/(factorial(m)*factorial(m+n))
        c = (x/2) ** (2*m+n)
        J = J + a*b*c
    return J

def bessel_recursive(n,x,J0,J1):
    """Use this method to find the bessel function for small and medium x,n
    Initialize J0 and J1 based on size of x
    J_n = 2*n/x * J_n-1 - J_n-2"""
    if n == 0:
        return J0
    elif n == 1:
        return J1
    else:
        J_prev, J_prev_2 = J1, J0
        for i in range(2, n + 1):
            J = (2*n)/x * J_prev - J_prev_2
            J_prev, J_prev_2 = J, J_prev
    return J

def bessel_integral(n,x, num_points = 1000):
    """Use this for medium x and small n"""
    #Use simpsons method : I = 3h/8 (f(x0)+f(xn) + 3(y1+y2+y4 ...) + 2(y3+y6...))

    def integrand_function(theta):
        integrand = np.cos(n*theta - x *np.sin(theta))
        return integrand
    lower_bound = 0
    upper_bound = np.pi
    delta_x = (upper_bound - lower_bound)/(num_points-1)
    
    theta_values = np.linspace(lower_bound, upper_bound, num_points)
    integral_values = integrand_function(theta_values)
    J = integral_values[0] + integral_values[-1]

    weights = np.ones_like(integral_values)*3
    weights[3::3] = 2

    J = J + np.sum(integral_values[1:-1]*weights[1:-1])
    return 3*delta_x/8 * J

def bessel_asymptotic(n,x):
    """Use this for large x"""
    return np.sqrt(2/(np.pi*x))*np.cos(x-n*np.pi/2 - np.pi/4)

def compute_bessel(n,x):
    """Decide which method to use for computing bessel function based on distance from origin
    This is to optimize efficiency and accuracy"""
    x_small_threshold = 5 
    x_large_threshold = 50
    n_small_threshold = 7
    if x<=x_small_threshold:
        if n<n_small_threshold:
            return bessel_series(n,x)
        else:
            J0 = bessel_series(0,x)
            J1 = bessel_series(1,x)
            return bessel_recursive(n,x,J0,J1)
    elif x<x_large_threshold:
        if n<n_small_threshold:
            return bessel_integral(n,x)
        else:
            J0 = bessel_integral(0,x)
            J1 = bessel_integral(1,x)
            return bessel_recursive(n,x,J0,J1)
    else:
        return bessel_asymptotic(n,x)
        
def get_instant_wave_function(n,m,a,c,t, R, theta): 
    """Solution to Wave Funtion u = cos(m*theta) * cos(c*lamda*t) * J_m(lamda*r)
    n->number of circular nodes
    m->number of azimuthal nodes
    a->radius of circular membrance
    c->speed of wave in medium
    t->instantaneous time
    R->mesh grid of radial and angular intervals
    theta->mesh grid of angular and radial intervals"""
    
    l = sc.jn_zeros(m,n)[-1]/a 
    print(sc.jn_zeros(m,n)[-1])
    u = np.cos(m*theta) * np.cos(c*l*t)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            u[i,j] = u[i,j] * compute_bessel(n,l * R[i,j])
    
    print(u[R.shape[0]-1,R.shape[1]-1])
    return u

def plot_wave(n, m, a, c, t_max, dt):
    """Plot animation of wave function
    n->number of circular nodes
    m->number of azimuthal nodes
    a->radius of circular membrance
    c->speed of wave in medium
    t_max->end time. This shall be converted to non dimensional time by factor 2pi
    dt-> Interval step size"""
    
    r_array = np.linspace(0, a, 200) #Initialize array of radial positions
    theta_array = np.linspace(0, 2 * np.pi, 200) #Initailize array of angular positions

    #Create a grid for plotting in polar coordinates 
    R, theta = np.meshgrid(r_array, theta_array)
    
    #Convert polar to cartesian coordinate
    X = R * np.cos(theta)
    Y = R * np.sin(theta)
    
    #Create a figure plot
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    ax.set_zlim(-1, 1)
    
    #Initalize Wave function evaluate to 0 for all positions on membrane
    U = np.zeros_like(X)
    
    #Plot the surface
    surf = ax.plot_surface(X, Y, U, cmap='viridis')
    
    #Find Framems required in animation
    frames = int(t_max / dt)

    def update(frame):
        nonlocal surf
        t = frame * dt/(2*np.pi)

        #Find the instanteous wave function for the given frame
        U = get_instant_wave_function(n, m, a, c, t, R, theta)

        #Removes the previous frame and plots the new one
        surf.remove() 
        surf = ax.plot_surface(X, Y, U, cmap='viridis')
        #Return New surface
        return [surf]

    #Create anumation
    ani = FuncAnimation(fig, update, frames=frames, interval=10, blit=False)

    #Show animation
    plt.show()

#Test
plot_wave(2,2,1,1,10,1)


