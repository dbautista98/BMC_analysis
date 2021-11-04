import trackpy as tp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pims
import cv2
import astropy.units as u
import astropy.constants as c

def color_plot(img, show_frac=.1, hsv=True, pshow=1, mask=None):

    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib import colors
    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1.,vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors)

    if hsv:
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    else:
        h, s, v = cv2.split(img)

    if mask is not None:
        h,s,v = h[mask], s[mask], v[mask]
    else:
        h,s,v = h.flatten(), s.flatten(), v.flatten()
    idx = np.random.choice(h.size, int(h.size*show_frac), replace=False)

    fig = plt.figure()
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    # print(len(pixel_colors))
    # raise Exception()
    axis.scatter(h[idx], s[idx], v[idx], facecolors=pixel_colors[idx].tolist(), marker=".")
    if hsv:
        axis.set_xlabel("Hue")
        axis.set_ylabel("Saturation")
        axis.set_zlabel("Value")
    else:
        axis.set_xlabel("Red")
        axis.set_ylabel("Green")
        axis.set_zlabel("Blue")

    if pshow:
        plt.show()

    return fig, axis

def annotate(filepath, diameter=5, minmass=50, maxsize=3, show=False, crop=((0,0), (0,0))):
    """
    takes a filepath and returns a trackpy pipeline for the images

    Arguments:
    -----------
    filepath : str
        path to the folder containing the .bmp files
    diameter : int
        odd integer setting the minimum radius of a particle
    minmass : float
        minimum detected brightness
    maxsize : float
        maximum allowed radius of a particle
    show : bool
        display image of tracked particles
    
    Returns: 
    ---------
    frames : slicerator.Pipeline
        pipeline of images to be used in tracking particles across images
    """
    @pims.pipeline
    def crop_img(image, crop=crop):
        sh = image.shape
        return image[crop[0][0]:(sh[0]-crop[0][1]), crop[1][0]:(sh[1]-crop[1][1]), :]

    frames = pims.as_grey(crop_img(pims.open(filepath)))#pims.open(path)

    loc = tp.locate(frames[0], diameter=diameter, minmass=minmass, maxsize=maxsize,)
    if show:
        tp.annotate(loc, frames[0])
    return frames

def get_trajectories(frames, search_range=10, diameter=5, minmass=50, maxsize=3, show=False, clip_stubs=0):
    """
    tracks the particles movement across images

    Arguments: 
    -----------
    frames : slicerator.Pipeline
        pipeline of images to be used in tracking particles across images
    search_range : int
        the maximum distance features can move between frames
    diameter : int
        odd integer setting the minimum radius of a particle
    minmass : float
        minimum detected brightness
    maxsize : float
        maximum allowed radius of a particle
    show : bool
        display image of particle trajectories
    clip_stubs : int
        number of frames a particle must be present in to be kept

    Returns:
    ---------
    linked : pandas.core.frame.DataFrame
        dataframe containing trajectory information for all tracked particles
    """
    all_imgs = tp.batch(frames, diameter=diameter, minmass=minmass, maxsize=maxsize)
    linked = tp.link(all_imgs, search_range=search_range)
    linked = tp.filtering.filter_stubs(linked, threshold=clip_stubs)
    if show:
        tp.plot_traj(linked)
    return linked

def clear_drift(df, show=False, correct=False):
    """
    checks and removes net drift of all particles

    Arguments:
    -----------
    df : pandas.core.frame.DataFrame
        dataframe containing trajectory information
    show : bool
        option to show plot of average drifts
    correct : bool
        option to remove any net drift of all particles
    
    Returns:
    ---------
    df : pandas.core.frame.DataFrame
        dataframe containing the trajectory information
    """
    d = tp.compute_drift(df)
    if show:
        d.plot()
        plt.show()
    if correct:
        df = tp.subtract_drift(df, d)
    return df

def diffusion_coeff(df, um_per_px, fps, show=False):
    """
    calculate the diffusion coefficient from a set of trajectories

    Arguments:
    -----------
    df : pandas.core.frame.DataFrame
        dataframe containing the trajectory information
    um_per_px : float
        conversion ratio between micrometers and pixels
    fps : float
        framerate of the images
    show : bool
        option to plot the power law fit to the data

    Returns:
    ---------
    D : float
        dispersion coefficient derved from trajectory data
    """
    em = tp.emsd(df, um_per_px, fps)
    fit = tp.utils.fit_powerlaw(em, plot=show)
    D = fit["A"][0]/4
    return D

def particle_velocity(tbl, particle_number, um_per_px, fps):
    """
    returns total velocity and x,y components of a single 
    particle's velocity in microns per second

    Arguments:
    -----------
    tbl : pandas.core.frame.DataFrame
        dataframe containing the trajectory information
    particle_number : int
        um_per_px : float
        conversion ratio between micrometers and pixels
    fps : float
        framerate of the images
    
    Returns:
    ---------
    velocity : numpy.ndarray
        an array of all velocities for the particle
    x_ velocity : numpy.ndarray
        an array of all x velocities for the particle
    y_velocity : numpy.ndarray
        an array of all y velocities for the particle
    """
    tbl = tbl[tbl["particle"] == particle_number]
    x_disp = np.diff(tbl['x'].values)
    y_disp = np.diff(tbl['y'].values)
    r_disp = np.sqrt(x_disp**2 + y_disp**2)
    velo = r_disp * fps / um_per_px
    x_velo = x_disp * fps / um_per_px
    y_velo = y_disp * fps / um_per_px
    return velo, x_velo, y_velo

def viscosity(tbl, particle_number, um_per_px, fps):
    """
    returns viscosity for each point in its trajectory
    under the assumption of spherical particles

    Arguments:
    -----------
    tbl : pandas.core.frame.DataFrame
        dataframe containing the trajectory information
    particle_number : int
        um_per_px : float
        conversion ratio between micrometers and pixels
    fps : float
        framerate of the images
    
    Returns:
    ---------
    mean_viscosity : float
        average viscosity based on the trajectory data
    sigma_visc : float
        uncertainty in the viscosity
    """

    tbl = tbl[tbl["particle"] == particle_number]
    D = diffusion_coeff(tbl, um_per_px, fps, show=False)*u.um**2/u.s
    R_gyration = tbl["size"].values*u.um
    kT = 300*u.K*c.k_B

    # calculate the radii under assumption of spherical blobs
    r = np.sqrt(5/3 * R_gyration**2)
    visc = (kT  / (6*np.pi*r*D)).to(u.mPa * u.s).value
    return visc

def work(tbl, particle_number, um_per_px, fps, x):
    """
    returns the amount of total work needed to transport a vesicle from 
    the perimeter of a cell to its center using Stokes Law

    Arguments:
    -----------
    tbl : pandas.core.frame.DataFrame
        dataframe containing the trajectory information
    particle_number : int
        index of particle in dataframe
    um_per_px : float
        conversion ratio between micrometers and pixels
    fps : float
        framerate of the images
    x : float
        distance traveled by the vesicle
    
    Returns:
    ---------
    work : float
       total amount of work
    """
    
    tbl = tbl[tbl["particle"] == particle_number]
    
    n = viscosity(tbl, particle_number, um_per_px, fps)
    
    R_gyration = tbl["size"].values
    mean_R_gyration = np.mean(R_gyration)*u.um
    r = np.sqrt(5/3 * mean_R_gyration**2)
    
    v = particle_velocity(tbl, particle_number, um_per_px, fps)[0]
    
    # WIP: determine proper distance to cell perimeter, x
    # for placeholder, set x = 20 um 
    
    return 3*np.pi*n*r*v*x #units of joules

def myosin_num(tbl, particle_number, um_per_px, fps, x):
    """
    returns the the minimum number of myosin motors required
    to transport a vesicle from the perimeter of the cell to its center.

    Arguments:
    -----------
    tbl : pandas.core.frame.DataFrame
        dataframe containing the trajectory information
    particle_number : int
        index of particle in dataframe
    um_per_px : float
        conversion ratio between micrometers and pixels
    fps : float
        framerate of the images
    x : float
        distance traveled by the vesicle
    
    Returns:
    ---------
    myosin_num : int
       number of myosin motors, rounded to the nearest integer
       (can't have one half motor, etc.)
    """
    
    work_calc = work(tbl, particle_number, um_per_px, fps, x)
    free_e = 57.5 # units of kJ/mol, amount of free energy that is released when ATP is hydrolyzed to ADP
                  # if this doesn't work, try 30.5 kJ/mol
    
        # per https://courses.lumenlearning.com/boundless-biology/chapter/atp-adenosine-triphosphate/
        # WIP: UNIT CONVERSION REQUIRED b/n REQUIRED WORK, FREE ENERGY RELEASED
    
    return np.round(work_calc / free_e) #expected value should be in the teens for a given cell

def theory_D(diameter, viscosity, temperature):
    """
    calculates the theoretical diffusion coefficient for a set of parameters

    Arguments:
    -----------
    diameter : float
        diameter of a particle in um 
    viscosity : float
        fluid viscosity in units of cP [mPa s]
    temperature : float
        temperature in Kelvin

    Returns:
    ---------
    D : float
        diffusion coefficient in units of um^2 s^-1
    """
    d = diameter*u.um                 # diameter in mocrometers
    eta = viscosity*u.mPa*u.second    # viscosity in cP [mPa * s]
    T = temperature*u.Kelvin          # Temperature in Kelvin

    D = (T*c.k_B / (3*np.pi*eta*d)).to(u.um**2/u.second)
    return D

def estimate_D(d_squared_displacement, dimensions, tau):
    simulated_D = np.mean(d_squared_displacement/(2*dimensions*tau))
    return simulated_D.to(u.um**2/u.second)

def uncertainty(d_squared_displacement, dimensions, tau, N, D, simulated_D):
    standard_error = np.std(d_squared_displacement) / (2*dimensions*tau*np.sqrt(N))
    actual_error = D - simulated_D
    return standard_error.to(u.um**2/u.second), actual_error.to(u.um**2/u.second)

def simulate_experiment(diameter, viscosity, temperature, tau=0.1, N=1000):
    D_predicted = theory_D(diameter, viscosity, temperature)
    dimensions = 2
    tau = tau*u.second
    time = tau*np.arange(N)
    k = np.sqrt(D_predicted*dimensions*tau)
    diameter = diameter*u.um
    viscosity = viscosity*u.mPa*u.second
    dx = k * np.random.randn(N)
    dy = k * np.random.randn(N)

    x = np.cumsum(dx)
    y = np.cumsum(dy)
    d_squared_displacement = dx**2 + dy**2
    squared_displacement = x**2 + y**2
    
    D_estimated = estimate_D(d_squared_displacement, dimensions, tau)
    stand_err, actual_err = uncertainty(d_squared_displacement, dimensions, tau, N, D_predicted, D_estimated)
    
    return squared_displacement.value, D_predicted.value, D_estimated.value, stand_err.value, actual_err.value

def simulation_wrapper(bead_sizes, viscosities, room_temp=300, n_points=1000, show=False):
    """
    simulates brownian motion with beads of given sizes in fluids of given viscosities

    Arguments:
    -----------
    bead_sizes : list
        list of bead sized in um
    viscosities : list
        list of fluid viscosities in units of cP (centi Poise) [mPa s]
    room_temp : float
        temperature in Kelvin
    n_points : int
        number of steps a particle should take in the simulation
    show : bool
        option to display plot of the squared displacement

    Returns:
    ---------
    analysis_results : pandas.core.frame.DataFrame
        dataframe containing the theoretical and simulated diffusion coefficients
    """
    #n_points = 1000
    all_sims = []
    
    if show:
        plt.figure(figsize=(10,7))
        for diam in bead_sizes:
            for visc in viscosities:
                simulation_results = simulate_experiment(diam, visc, room_temp, N=n_points)
                all_sims.append(simulation_results[1:])
                plt.plot(simulation_results[0], label=r"d = %s $\mu m$, $\eta$ = %s cP"%(diam, visc))
        plt.legend()
        plt.ylabel(r"displacement squared $[\mu m^2]$");
        plt.show()
    else:
        for diam in bead_sizes:
            for visc in viscosities:
                simulation_results = simulate_experiment(diam, visc, room_temp, N=n_points)
                all_sims.append(simulation_results[1:])

    all_sims = np.asarray(all_sims)
    labels = []
    analysis_results = pd.DataFrame()
    for diam in bead_sizes:
        for visc in viscosities:
            labels.append("d = %s um, visc = %s cP"%(diam, visc))

    analysis_results["parameters"] = labels

    column_names = ["D theory [mPa s]", "D measured [mPa s]", "standard error [mPa s]", "actual error [mPa s]"]
    for i in range(len(column_names)):
        analysis_results[column_names[i]] = all_sims[:, i]
    return analysis_results

