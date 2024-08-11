import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.transforms import Affine2D

def distance(vector):
    """Helper function to find the length of a vector"""
    return np.linalg.norm(vector) #Use this form to prevent floating point error

def Verlet(M, position0, velocity0, time_max, dt): #Using Verlet integrator
    
    """IMPORTANT: position0 and velocity0 should be defined as np.array([[x,y,z]]) so that it is array of vectors
       M is mass of the planet, time_max is end time for simulation, dt is step size
       Function returns a 2D array of position and velocity vectors"""
    
    G = 6.67 * 10 ** (-11)
    time_array = np.arange(0,time_max,dt) #Creating time axis
    
    #Convert into 2D array of transposed vectors
    position_vector_list_Verlet= np.array(position0)
    velocity_vector_list_Verlet= np.array(velocity0)

    #Since using Verlet and we use the 2 previous values, evelaute position1 and velocity1 directly
    position1 = position0 + velocity0 * dt -0.5 * (G*M/distance(position0)**3)*position0 * dt * dt #x_1 = x0 + v0 *dt + 0.5 * a0 * dt**2
    velocity1 = velocity0 - (G*M/distance(position0)**3)*position0 * dt #v1 = v0 + a0 * dt

    #Convert output into array
    position1 = np.array(position1)
    velocity1 = np.array(velocity1)

    #Stack array
    position_vector_list_Verlet= np.vstack((position_vector_list_Verlet,position1))
    velocity_vector_list_Verlet= np.vstack((velocity_vector_list_Verlet,velocity1))

    #Verlet Integration
    for i in range(2,len(time_array)):
        current_position=position_vector_list_Verlet[i-1] #Store current position
        r = distance(current_position) #Calculate distance from origin
        a = -G*M/r**3 * current_position #Acceleration due to Gravity
        new_position = 2 * position_vector_list_Verlet[i-1] - position_vector_list_Verlet[i-2] + a*dt**2 #Verlet formula x_(n+1) = 2*x_(n) - x_(n-1) + a(n)*dt**2
        position_vector_list_Verlet = np.vstack((position_vector_list_Verlet, new_position)) #Stack new position in list

    #Evaluate Velocity using central difference method
    for i in range(1, len(position_vector_list_Verlet) - 1):
        previous_position = position_vector_list_Verlet[i-1]
        next_position = position_vector_list_Verlet[i+1]
        new_velocity = (next_position-previous_position)/(2*dt)
        velocity_vector_list_Verlet = np.vstack((velocity_vector_list_Verlet, new_velocity))
    
    return position_vector_list_Verlet,velocity_vector_list_Verlet

def plot_3D_position(position_vector_list, R = 0):
    """position_vector_list is a 2D array of coordinates that is a result of verlet integrator
    R is an optional input which is the radius of the planet
    Function returns a plot of the orbit"""
    
    fig = plt.figure() #Create a plot object
    ax = fig.add_subplot(111, projection='3d') #add a 3D projected figure to plot

    # Plot the position vectors
    ax.plot(position_vector_list[:,0], position_vector_list[:,1], position_vector_list[:,2], label='Position Path')

    # Label the origin
    ax.scatter([0], [0], [0], color='red', s=100)
    ax.text(0, 0, 0, 'Origin', color='red')

    #Label the initial point
    ax.scatter(position_vector_list[0][0], position_vector_list[0][1], position_vector_list[0][2], color='red', s=100)  # origin point
    ax.text(position_vector_list[0][0], position_vector_list[0][1], position_vector_list[0][2], 'Initial Position', color = 'blue')
    
    # Plot a sphere of radius R to show the planet 
    if R > 0:
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j] 
        #Spherical coordinates
        x = R * np.cos(u) * np.sin(v) 
        y = R * np.sin(u) * np.sin(v)
        z = R * np.cos(v)
        ax.plot_surface(x, y, z, color="red", alpha=0.4, shade=True)

    # Set labels
    ax.set_xlabel('X /m')
    ax.set_ylabel('Y /m')
    ax.set_zlabel('Z /m')
    ax.legend()
    
    plt.show()

def animation_plot(position_vector_list, velocity_vector_list, R=0):
    """position_vector_list is a 2D array of coordinates that is a result of verlet integrator
    velocity_vector_list is a 2D array of velocity vectors that is a result of verlet integrator
    R is an optional input which is the radius of the planet
    Function returns a plot of the orbit, an animation of motion along the orbit with the velocity vector"""
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter([0], [0], [0], color='red', s=100)  # origin point
    ax.text(0, 0, 0, 'Origin', color='red')
    
    position_path, = ax.plot(position_vector_list[:,0], position_vector_list[:,1], position_vector_list[:,2], label='Position Path')
    
    ax.scatter(position_vector_list[0][0], position_vector_list[0][1], position_vector_list[0][2], color='red', s=100)  # initial position
    ax.text(position_vector_list[0][0], position_vector_list[0][1], position_vector_list[0][2], 'Initial Position', color='blue')
    plt.gca().set_aspect('equal', adjustable='box')
    ax.set_xlabel('X /m')
    ax.set_ylabel('Y /m')
    ax.set_zlabel('Z /m')
    l1 = position_vector_list[:, 0].max() - position_vector_list[:, 0].min()
    l2 = position_vector_list[:, 1].max() - position_vector_list[:, 1].min()
    l3 = position_vector_list[:, 2].max() - position_vector_list[:, 2].min()
    a = max(l1,l2,l3)
    # Plot a sphere of radius R
    if R > 0:
        u, v = np.mgrid[0:2*np.pi:100j, 0:np.pi:50j]
        x = R * np.cos(u) * np.sin(v)
        y = R * np.sin(u) * np.sin(v)
        z = R * np.cos(v)
        ax.plot_surface(x, y, z, color="red", alpha=0.6, shade=True)

    vector_arrow = ax.quiver(
        position_vector_list[0,0], position_vector_list[0,1], position_vector_list[0,2],
        velocity_vector_list[0,0], velocity_vector_list[0,1], velocity_vector_list[0,2],
        color='green', length=a/5, normalize=True)
    
    
    # Initialize a point that will move along the path
    point, = ax.plot([position_vector_list[0,0]], [position_vector_list[0,1]], [position_vector_list[0,2]], 'go')
    def update_point(frame):
        point.set_data(position_vector_list[frame,0], position_vector_list[frame,1])
        point.set_3d_properties(position_vector_list[frame,2])
        return point,
    
    def update_arrowhead(frame):
        nonlocal vector_arrow
        if vector_arrow:
            vector_arrow.remove()
        vector_arrow = ax.quiver(
            position_vector_list[frame,0], position_vector_list[frame,1], position_vector_list[frame,2],
            velocity_vector_list[frame,0], velocity_vector_list[frame,1], velocity_vector_list[frame,2],
            color='green', length=a/5, normalize=True)
        return vector_arrow,
    
    def updateALL(frameNum):
        p = update_point(frameNum)
        d = update_arrowhead(frameNum)
        return p + d
    
    arrow_marker = TextPath((0, 0), u'\u2192', size=12)
    arrow_patch = PathPatch(arrow_marker, transform=Affine2D().scale(1, 1) + ax.transData, color='green')
    proxy_velocity = arrow_patch
    ani = animation.FuncAnimation(fig, updateALL, frames=len(position_vector_list), interval=30, blit=False)
    plt.legend([position_path, point, proxy_velocity], ['Position Path', 'Particle', 'Velocity Vector'+ u'\u2192'])
    plt.show()

def velocity_field(position_vector_list, velocity_vector_list, interval):
    """position_vector_list is a 2D array of coordinates that is a result of verlet integrator
    velocity_vector_list is a 2D array of velocity vectors that is a result of verlet integrator
    interval is step between 2 considered velocity vectors
    Function returns a plot of the orbit"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X /m')
    ax.set_ylabel('Y /m')
    ax.set_zlabel('Z /m')
    
    l1 = position_vector_list[:, 0].max() - position_vector_list[:, 0].min()
    l2 = position_vector_list[:, 1].max() - position_vector_list[:, 1].min()
    l3 = position_vector_list[:, 2].max() - position_vector_list[:, 2].min()
    a = max(l1, l2, l3)

    for i in range(0, len(position_vector_list), interval):
        ax.quiver(
            position_vector_list[i, 0], position_vector_list[i, 1], position_vector_list[i, 2],
            velocity_vector_list[i, 0], velocity_vector_list[i, 1], velocity_vector_list[i, 2],
            color='green', length=a/10, normalize=True)
    plt.show()

#Example Execution using satellite around Mars at altitude 20000km 
M = 6.42 * 10**23 #Mass of Mars
R = 3400 * 10**3 #Radius of Mars
h = 20000 * 10**3 #Altitude
G = 6.67 * 10**(-11) #Gravitational Constant
v0 = np.sqrt(G*M/(R+h)) #Velocity for circular Orbit
time_max = 2000000 #Max time for simulation considered
dt = 1000 #Step size such that 2000 points considered
position0 = np.array([[(R+h),0,(R+h)]]) #Initialize initial position SPECIFICALLY in this format
velocity0 = np.array([[-v0/4, v0, v0/4]]) #Initialize initial velocity SPECIFICALLY in this format
position,velocity = Verlet(M, position0, velocity0, time_max, dt) #Find positions and velocity
plot_3D_position(position, R)
animation_plot(position, velocity, R)
velocity_field(position, velocity, 25)