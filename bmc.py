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

def get_trajectories(frames, search_range=10, diameter=5, minmass=50, maxsize=3, show=False):
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

    Returns:
    ---------
    linked : pandas.core.frame.DataFrame
        dataframe containing trajectory information for all tracked particles
    """
    all_imgs = tp.batch(frames, diameter=diameter, minmass=minmass, maxsize=maxsize)
    linked = tp.link(all_imgs, search_range=search_range)
    if show:
        tp.plot_traj(linked)
    return linked

def drift(df, show=False, correct=False):
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
    returns average velocity of a single particle in microns per second

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
    mean : float
        average velocity of the particle
    st_dev : float
        the standard deviation of the particle's velocity
    velocity_vector : numpy.ndarray
        a vector of the particle's average velocity
    all_velocity : numpy.ndarray
        an array of all velocities for the particle
    """
    tbl = tbl[tbl["particle"] == particle_number]
    x_disp = np.diff(tbl['x'].values)
    y_disp = np.diff(tbl['y'].values)
    r_disp = np.sqrt(x_disp**2 + y_disp**2)
    velo = r_disp * fps / um_per_px

    x_vec = np.mean(x_disp)
    y_vec = np.mean(y_disp)
    vector = np.asarray([x_vec, y_vec])
    velocity_vector = velo * (vector/np.linalg.norm(vector))
    return np.mean(velo), np.std(velo), velocity_vector, velo

def viscosity(tbl, particle_number, um_per_px, fps):
    """
    returns average viscosity as well as the uncertainty bounds
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
    over_viscosity : float
        upper bound on the viscosity based on the uncertainty
    under_viscosity : float
        lower bound on the viscosity based on the uncertainty
    """

    tbl = tbl[tbl["particle"] == particle_number]
    D = diffusion_coeff(tbl, um_per_px, fps, show=False)*u.um**2/u.s
    R_gyration = tbl["size"].values
    sigma_R_gyration = np.std(R_gyration)*u.um
    mean_R_gyration = np.mean(R_gyration)*u.um
    kT = 300*u.K*c.k_B

    # calculate the radii under assumption of spherical blobs
    r = np.sqrt(5/3 * mean_R_gyration**2)
    r_over = np.sqrt(5/3 * (mean_R_gyration + sigma_R_gyration)**2)
    r_under = np.sqrt(5/3 * (mean_R_gyration - sigma_R_gyration)**2)

    mean_visc =  (kT  / (6*np.pi*r*D)).to(u.mPa * u.s)
    over_visc = (kT  / (6*np.pi*r_under*D)).to(u.mPa * u.s)
    under_visc = (kT  / (6*np.pi*r_over*D)).to(u.mPa * u.s)
    return mean_visc.value, over_visc.value, under_visc.value
