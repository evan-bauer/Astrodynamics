from copy import copy, deepcopy

import numpy as np    
import scipy.special as sp
from scipy.optimize import newton
from scipy.integrate import odeint
from scipy.spatial.transform import Rotation
from scipy.signal import argrelmin

from mpl_toolkits import mplot3d
import matplotlib as mpl
import matplotlib.pyplot as plt

AU = 149597871000              #  m
G = 6.67430e-11                #  m^3 s^-2
cos, sin = sp.cosdg(np.linspace(0, 360)), sp.sindg(np.linspace(0, 360))

sun = {"name": "Sun",
       "color": "yellow",
       "e": 0,
       "a": 0,
       "inc": 0,
       "long_asc_node": 0,
       "arg_periapsis": 0,
       "radius": 6.957e+8,
       "mass": 1.98847e+30}
mercury = {"name": "Mercury",
           "color": "gray",
           "e": 0.205630,
           "a": 0.387098 * AU,
           "inc": {"sun_eq": 3.38,
                   "ecliptic": 7.005,
                   "invariable_plane": 6.35},
           "long_asc_node": 48.331,
           "arg_periapsis": 29.124,
           "radius": 2.4397e+6,
           "mass": 3.3011e+23}
venus = {"name": "Venus",
         "color": "yellow",
         "e": 0.006772,
         "a": 0.723332 * AU,
         "inc": {"sun_eq": 3.86,
                 "ecliptic": 3.39458,
                 "invariable_plane": 2.15},
         "long_asc_node": 76.680,
         "arg_periapsis": 54.884,
         "radius": 6.0518e+6,
         "mass": 4.8675e+24}
earth = {"name": "Earth", 
         "color": "g",
         "e": 0.0167086,
         "a": 149598023000, 
         "inc": {"sun_eq": 7.155,
                 "ecliptic": 0.00005,
                 "invariable_plane": 1.57869},
         "long_asc_node": -11.26064, 
         "arg_periapsis": 114.20783, 
         "radius": 6371000, 
         "mass": 5.97217e+24}
moon = {"name": "Moon",
        "color": "grey",
        "e": 0.0549, 
        "a": 384399000, 
        "inc": {"sun_eq": 0,
                "ecliptic": 5.145,
                "invariable_plane": 0},
        "long_asc_node": 0, 
        "arg_periapsis": 0,
        "radius": 1737400,
        "mass": 7.342e22}
mars = {"name": "Mars",
        "color": "red",
        "e": 0.0934, 
        "a": 2.27939366e+11, 
        "inc": {"sun_eq": 5.65,
                "ecliptic": 1.850,
                "invariable_plane": 1.63},
        "long_asc_node": 49.57854, 
        "arg_periapsis": 286.5,
        "radius": 3.3895e+6,
        "mass": 6.4171e+23}
jupiter = {"name": "Jupiter", 
           "color": "orange",
           "e": 0.0489,
           "a": 5.2038 * AU, 
           "inc": {"sun_eq": 6.09,
                   "ecliptic": 1.303,
                   "invariable_plane": 0.32},
           "long_asc_node": 100.464, 
           "arg_periapsis": 273.867, 
           "radius": 69911000, 
           "mass": 1.8982e+27}
saturn = {"name": "Saturn", 
          "color": "yellow",
          "e": 0.0565,
          "a": 9.5826 * AU, 
          "inc": {"sun_eq": 5.51,
                  "ecliptic": 2.485,
                  "invariable_plane": 0.93},
          "long_asc_node": 113.665, 
          "arg_periapsis": 339.392, 
          "radius": 58.232e+6, 
          "mass": 5.6834e+26}
uranus = {"name": "Uranus", 
          "color": "cyan",
          "e": 0.04717,
          "a": 19.19126 * AU, 
          "inc": {"sun_eq": 6.48,
                  "ecliptic": 0.773,
                  "invariable_plane": 0.99},
          "long_asc_node": 74.006, 
          "arg_periapsis": 96.998857, 
          "radius": 25.362e+6, 
          "mass": 8.681e+25}
neptune = {"name": "Neptune", 
           "color": "blue",
           "e": 0.008678,
           "a": 30.07 * AU, 
           "inc": {"sun_eq": 6.43,
                   "ecliptic": 1.770,
                   "invariable_plane": 0.74},
           "long_asc_node": 131.783, 
           "arg_periapsis": 273.187, 
           "radius": 24.622e+6, 
           "mass": 1.024e+26}

timestep_conversion = {"seconds": 1, 
                       "minutes": 60, 
                       "hours": 3600, 
                       "days": 86400, 
                       "years": 365 * 86400}
spatial_conversion = {"m": 1,
                      "km": 1e3,
                      "Mm": 1e6,
                      "Gm": 1e9,
                      "AU": AU}

unit = lambda vector: vector / np.linalg.norm(vector)

terrans, giants = [mercury, venus, earth, mars], [jupiter, saturn, uranus, neptune]
kepler, cartes = ["a", "e", "inc", "arg_periapsis", "long_asc_node", "true_anomaly"], ["position", "velocity"]

def check_sign(a: float, e: float) -> list:
    """Function verifying that the sign of `a` is consistent with the magnitude of eccentricity."""
    if (a < 0) and (e < 1): a *= -1
    if (a > 0) and (e > 1): a *= -1
    return a, e
    
def can_keep_position(position: np.ndarray, a: float, e: float) -> bool:
    """Function for checking if a new trajectory can be established to pass through an existing position vector."""
    if e >= 1: return False
    r = np.linalg.norm(position)
    return (a * (1 - e) <= r) and (a * (1 + e) >= r)
    
def sphere(radius):
    """Generate points along the surface of a sphere of specified radius."""
    u, v = np.linspace(0, 2 * np.pi, 15), np.linspace(0, np.pi, 15)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    return radius * np.array([x, y, z])
    
def radians(deg: float | int) -> float: 
    """Convert degrees to radians."""
    return (deg / 360) * (2 * np.pi) % (2 * np.pi)
    
def degrees(rad: float | int) -> float: 
    """Convert radians to degrees."""
    ang = (rad / (2 * np.pi)) * 360
    return ang % 360

def semi_circ(theta: float) -> float:
    """Function (primarily for inclination) to make an angle more readable."""
    theta = theta % 180
    return theta - 180 if theta > 90 else theta
    
def reduced_mass(M: float, m: float) -> float:
    """Returns reduced mass of two massive bodies."""
    return (M * m) / (M + m)

def U_gravitational(r: float, M: float, m: float) -> float:
    """Computes the gravitational potential of a two-body system at a specific radial distance."""
    return - G * M * m / r
    
def U_centrifugal(r: float, M: float, m: float, h: float) -> float:
    """Computes the centrifugal potential of a two-body system at a specific radial distance."""
    return (reduced_mass(M, m) * h) ** 2 / (2 * reduced_mass(M, m) * r ** 2)

def r_soi(r: float, M: float, m: float) -> float: 
    """Returns the radius of a massive object's sphere of influence at a radial distance."""
    return r * (m / M) ** (2 / 5)
    
def grav_param(m1: float, m2: float) -> float: 
    """Determines the standard gravitational parameter of a two-body system."""
    return G * (m1 + m2)
    
def radial_velocity(position: np.ndarray, velocity: np.ndarray) -> float: 
    """Calculate the radial component of the velocity vector."""
    return np.dot(velocity, unit(position))
    
def eccentricity_vector(position: np.ndarray, velocity: np.ndarray, std_grav_param: float) -> np.ndarray: 
    """Calculate eccentricity vector."""
    return np.cross(velocity, np.cross(position, velocity)) / std_grav_param - unit(position)
    
def semi_latus_rectum(a: float, e: float) -> float:
    """Determine semi-latus rectum of a semi-major axis and eccentricity pair."""
    if e == 0: return a
    if e < 1: return a * (1 - e**2)
    if e == 1: raise NotImplementedError("Process of defining parabolic trajectories unavailable.")
    if e > 1: return - a * (e**2 - 1)
        
def mean_angular_motion(std_grav_param: float, a=None) -> float:
    """Computes the mean angular motion of a satellite."""
    if a == None: return np.sqrt(std_grav_param)
    return np.sqrt(std_grav_param / np.abs(a ** 3))

def hyperbolic_2_true_anom(hyp_anom: float, e: float) -> float: 
    """Convert hyperbolic anomaly to true anomaly."""
    hyp_anom = radians(hyp_anom)
    return degrees(2 * np.arctan(np.sqrt((e + 1) / (e - 1)) * np.tanh(hyp_anom / 2)))
    
def true_2_hyperbolic_anom(true_anom: float, e: float) -> float: 
    """Convert true anomaly to hyperbolic anomaly."""
    return degrees(2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * sp.tandg(true_anom / 2)))
    
def true_2_eccentric_anom(true_anom: float, e: float) -> float: 
    """Convert true anomaly to eccentric anomaly."""
    return degrees(np.arctan((np.sqrt(1 - e**2) * sp.sindg(true_anom)) / (e + sp.cosdg(true_anom))))
    
def eccentric_2_true_anom(ecc_anom: float, e: float) -> float:
    """Convert eccentric anomaly to true anomaly."""
    beta = e / (1 + np.sqrt(1 - e**2))
    return ecc_anom + degrees(2 * np.arctan((beta * sp.sindg(ecc_anom)) / (1 - beta * sp.cosdg(ecc_anom))))

def mean_2_eccentric_anom(mean_anom: float, e: float) -> float:
    """Convert the mean anomaly to the eccentric anomaly using summation of Bessel functions."""
    return mean_anom + degrees(2 * sum((sp.jv(n, n * e) / n * sp.sindg(n * mean_anom)) for n in range(1, 250)))

def mean_2_true_anom(mean_anom: float, e: float) -> float:
    """Convert the mean anomaly to the true anomaly using summation of Bessel functions."""
    beta = (1 - np.sqrt(1 - e**2)) / e
    inner = lambda s: sum((beta**p * (sp.jv(s - p, s * e) + sp.jv(s + p, s * e))) for p in range(1, 150))
    return mean_anom + degrees(2 * sum((sp.sindg(s * mean_anom) / s * (sp.jv(s, s * e) + inner(s))) for s in range(1, 250)))  

def mean_2_hyperbolic_anom(mean_hyp_anom: float, e: float) -> float:
    """Function for estimating the hyperbolic anomaly from mean hyperbolic anomaly."""
    mean_hyp_anom = radians(mean_hyp_anom)
    def f(F): return e * np.sinh(F) - F - mean_hyp_anom
    return degrees(newton(func=f, x0=mean_hyp_anom, fprime=lambda F: e * np.cosh(F) - 1, fprime2=lambda F: e * np.sinh(F), maxiter=200))

def true_anom_from_vectors(position: np.ndarray, velocity: np.ndarray, std_grav_param: float) -> float:
    """Function returning true anomaly calculated from state vectors and standard gravitational parameter."""
    ecc_vector = eccentricity_vector(position, velocity, std_grav_param)
    nu = degrees(np.arccos(np.dot(ecc_vector, position) / np.multiply(*np.linalg.norm((ecc_vector, position), axis=1))))
    if np.linalg.norm(ecc_vector) >= 1: return nu
    return nu if np.dot(position, velocity) < 0 else (360 - nu)

def mean_anomaly_shift(true_anom: float, e: float) -> float:
    """Function computing the phase shift of the mean anomaly due to a nonzero true anomaly at the start of the simulation."""
    return degrees(np.arctan2(-np.sqrt(1 - e**2) * sp.sindg(true_anom), -e - sp.cosdg(true_anom)) + np.pi - e * (np.sqrt(1 - e**2) * sp.sindg(true_anom)) / (1 + e * sp.cosdg(true_anom)))

def mass_calculator(obj) -> float:
    """Calculates the total mass of the given body and its satellites."""
    total_mass = 0
    if isinstance(obj, Satellite):
        total_mass += obj.body.mass
    elif isinstance(obj, Body):  
        total_mass += obj.body_mass
    elif isinstance(obj, System):
        total_mass += obj.primary.mass
    if len(obj) > 0: total_mass += sum(mass_calculator(satellite) for key, satellite in obj)
    return total_mass
    
def satellite_root(obj) -> str:
    """Returns the name of the primary objects of which `obj` is a satellite of."""
    root_trace = ""
    if hasattr(obj, "primary"): root_trace += f"{satellite_root(obj.primary)}"
    if isinstance(obj, Satellite):
        root_trace += obj.body.name
    elif isinstance(obj, Body):
        root_trace += obj.body_name
    return root_trace

def namer(obj) -> str:
    """Returns the name of the system with primary `obj`."""
    full_name = f"{obj.body.name if isinstance(obj, Body) else satellite_root(obj)}"
    if len(obj) > 0: full_name += f"[{', '.join(namer(satellite) for key, satellite in obj)}]system"
    return full_name
    
def plot_energy(obj, y_window=None, r_max=1e+8):
    """Function for plotting the effective potential of a two-body system."""
    plt.style.use('default')    
    fig, ax = plt.figure(figsize=(6, 6)), plt.axes()

    # ESTIMATE APSE DISTANCES
    mag_h = np.linalg.norm(obj.specific_angular_momentum)
    if obj.e > 1: r_peri, r_apo = -obj.a * (obj.e - 1), np.inf
    if obj.e == 1: r_peri, r_apo = mag_h**2 / (2 * G * (obj.primary.mass + obj.mass)), np.inf
    if obj.e == 0: r_peri, r_apo = obj.a, obj.a
    if obj.e < 1: r_peri, r_apo = obj.a * (1 - obj.e), obj.a * (1 + obj.e)
    if np.isinf(r_apo):
        if r_max == None: r_max = 1e+8
        r, rs = np.linspace(10, r_max * 1.1), np.linspace(r_peri, r_max * 1.1)
    else:
        r, rs = np.linspace(10, r_apo * 1.1), np.linspace(r_peri, r_apo)

    # CALCULATE POTENTIALS
    U_cf, U_gr = U_centrifugal(r, obj.primary.mass, obj.mass, mag_h),  U_gravitational(r, obj.primary.mass, obj.mass)
    U_cf2, U_gr2 = U_centrifugal(rs, obj.primary.mass, obj.mass, mag_h), U_gravitational(rs, obj.primary.mass, obj.mass)
    U_eff, U_eff2 = U_cf + U_gr, U_gr2 + U_cf2
    r_0 = rs[U_eff2.argmin()]

    ax.grid()
    ax.plot(r, U_gr, c='r', label=r'$U_{grav}$')
    ax.plot(r, U_cf, c='g', label=r'$U_{cf}$')
    ax.plot(r, U_eff, c='b', label=r'$U_{eff}$')
    ax.axhline(obj.specific_energy, c='k', label=r"$E$")
    ax.axvline(r_peri, dashes=(1, 1), c='orange', label=r"$r_{peri}$")
    ax.axvline(r_apo, dashes=(1, 1), c='cyan', label=r"$r_{apo}$") if not r_apo == np.inf else None
    ax.axvline(r_0, dashes=(1, 1), c='magenta', label=r"$r_{circ}$")
    ax.set_title(f"Effective Potential of {obj.body.name} About {obj.primary.body_name}")
    ax.set_xlabel(r"Radial Distance (m)")
    ax.set_ylabel(r"Effective Potential")
    ax.set_xlim(0, r.max())
    if type(y_window) == list:
        ax.set_ylim(*y_window)
    elif type(y_window) == int:
        ax.set_ylim(-y_window, y_window)
    ax.fill_between(rs, np.full_like(rs, obj.specific_energy), U_eff2, alpha=.25)
    if np.isinf(r_apo):
        ax.set_xticks([0, r_peri, r_0, r.max()], [0, r"$r_{peri}$", r"$r_{circ}$", "max"])
    else:
        ax.set_xticks([0, r_peri, r_0, r_apo, r.max()], [0, r"$r_{peri}$", r"$r_{circ}$", r"$r_{apo}$", "max"])
    ax.set_axisbelow(True)
    plt.legend(loc='lower left')
    plt.show()

def makefig(graphic_type: str, scale: str = "AU"):
    """
    Internal function to create the framework for plotting a simulable system.
    
    Parameters
    ----------
    graphic_type : str
        Type of graphic to produce, '3d' makes a 3-dimensional graphic, 'planar' plots the projection of a trajectory onto the XY, XZ, and YZ planes.
    scale : str
        Spatial scale to make results more readable (default: 'AU').
        
    Returns
    -------
    list
        List of length 3 of format `[plt.Figure, plt.Axes, str]`.
        
    Raises
    ------
    KeyError
        If the provided spatial scale is undefined.
    """
    
    plt.style.use('dark_background')
    if not scale in spatial_conversion.keys(): raise KeyError(f"Spatial scale {scale} not defined.")
    scale_factor = spatial_conversion[scale]
    
    if graphic_type == "planar":
        fig, axes = plt.subplots(nrows=2, ncols=2, sharex="col", figsize=(10, 10), gridspec_kw={'hspace': 0.035, 'wspace': 0.025})
        ((ax1, ax2), (ax3, ax4)) = axes
        X, Y, Z = (f"{coord} [{scale}]" for coord in "XYZ")
        for ax in [ax1, ax2]:
            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")
        for ax in [ax2, ax4]:
            ax.set_xlabel(Z)
            ax.set_ylabel(Y)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        ax1.set_title(f'XY Plane')
        ax1.set_xlabel(X)
        ax1.set_ylabel(Y)
        ax1.yaxis.set_label_position("left")
        ax2.set_title(f'YZ Plane')
        ax3.set_title(f'XZ Plane', y=0, pad=-45)
        ax3.set_xlabel(X)
        ax3.xaxis.set_label_position("bottom")
        ax3.set_ylabel(Z)
        ax4.set_title(f'YZ Plane', y=0, pad=-45)
    elif graphic_type == "3d":
        fig = plt.figure()
        axes = fig.add_subplot(projection="3d")
        axes.set_xlabel(f"X [{scale}]")
        axes.set_ylabel(f"Y [{scale}]")
        axes.set_zlabel(f"Z [{scale}]")    
    fig.suptitle("Predicted Trajectories of Provided System", y=.95)
    return [fig, axes, graphic_type]

def central_obj(fig, axes, graphic_type: str, primary_obj, scale: str = "AU"):
    """
    Internal function to draw the static position of the system primary (e.g. the Sun).
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.
    axes : numpy.ndarray (matplotlib axes object)
        Matplotlib axes object.
    graphic_type : str
        Type of graphic to produce, '3d' makes a 3-dimensional graphic, 'planar' plots the projection of a trajectory onto the XY, XZ, and YZ planes.
    primary_obj : Satellite or System
        Primary body of the satellite to plot (assumed x=y=z=0).
    scale : str
        Spatial scale to make results more readable (default: 'AU'). Must be the same string passed to `makefig`.
    position : np.ndarray
        Position to place the central object at.**needs revision
        
    Returns
    -------
    list
        List of length 3 of format `[plt.Figure, plt.Axes, str]`.
    """
    
    scale_factor = spatial_conversion[scale]
    radius = primary_obj.radius / scale_factor
    if graphic_type == "planar":
        ((ax1, ax2), (ax3, ax4)) = axes
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.add_patch(plt.Circle((0, 0), radius, color=primary_obj.color, alpha=.75))
            if isinstance(primary_obj, Satellite): ax.add_patch(plt.Circle((0, 0), primary_obj.SOI / scale_factor, color="red", fill=False, linestyle=":"))
    elif graphic_type == "3d":
        if isinstance(primary_obj, Satellite): axes.text(0, 0, 0, "%s" % primary_obj.body.name, size=8, zorder=1, color="white") 
        if isinstance(primary_obj, System): axes.text(0, 0, 0, "%s" % primary_obj.name, size=8, zorder=1, color="white") 
        axes.plot_surface(*sphere(radius), color=primary_obj.color, alpha=1)
        if isinstance(primary_obj, Satellite): axes.plot_wireframe(*sphere(primary_obj.SOI / scale_factor), alpha=.25, color="red")
    return [fig, axes, graphic_type]

def plot_satellite(fig, axes, graphic_type: str, sat_obj, relative_to=0, scale: str = "AU", markevery: int = None, timestep: str = "days", cbar: bool = False):
    """
    Internal function to draw the motion of a secondary body orbiting a primary (e.g. the Earth about the Sun).
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.
    axes : numpy.ndarray (matplotlib axes object)
        Matplotlib axes object.
    graphic_type : str
        Type of graphic to produce, '3d' makes a 3-dimensional graphic, 'planar' plots the projection of a trajectory onto the XY, XZ, and YZ planes.
    sat_obj : Satellite
        Satellite object in orbit about the primary.
    relative_to : Satellite or int
        The Satellite body the new Satellite is to be plotted with respect to (default: 0).
    scale : str
        Spatial scale to make results more readable (default: 'AU'). Must be the same string passed to `makefig`.
    markevery : int or None
        Number of points to skip when plotting, primarily used to speed up and simplify the plotting process by plotting fewer points (default : None).
    timestep : str
        Timestep to re-scale the colorbar representing the simulation time to (default: 'days').
    cbar : bool
        If `True`, a colorbar will be added to the plot (default: `False`). Intended to avoid multiple identical colorbars.
        
    Returns
    -------
    list
        List of length 3 of format `[plt.Figure, plt.Axes, str]`.
    """

    scale_factor = spatial_conversion[scale]
    time = sat_obj.prediction.time_array[::markevery] / timestep_conversion[timestep]
    sat_coords = sat_obj.prediction.global_coords[::markevery, :].T / scale_factor
    if isinstance(relative_to, int):
        pass
    else:
        if relative_to.body.name == sat_obj.primary.body.name: sat_coords = sat_obj.prediction.local_coords[::markevery, :].T / scale_factor
            
    r0, rf = sat_obj.prediction.local_coords[0, :] / AU, sat_obj.prediction.local_coords[-1, :] / AU
    x0, y0, z0, xf, yf, zf = *sat_coords[:, 0], *sat_coords[:, -1]
    if isinstance(sat_obj.primary, Satellite):
        soi_0, soi_f = np.apply_along_axis(r_soi, 0, np.linalg.norm((r0, rf), axis=1), *[sat_obj.primary.body.mass, sat_obj.body.mass])
    elif isinstance(sat_obj.primary, System):
        soi_0, soi_f = np.apply_along_axis(r_soi, 0, np.linalg.norm((r0, rf), axis=1), *[sat_obj.primary.body_mass, sat_obj.body.mass])
    if graphic_type == "planar":
        ((ax1, ax2), (ax3, ax4)) = axes
        cb = ax1.scatter(*sat_coords[[0, 1], :], c=time, cmap="viridis", s=2)
        ax2.scatter(*sat_coords[[2, 1], :], c=time, cmap="viridis", s=2)
        ax3.scatter(*sat_coords[[0, 2], :], c=time, cmap="viridis", s=2)
        ax4.scatter(*sat_coords[[2, 1], :], c=time, cmap="viridis", s=2)
        pos_0, pos_f = [(x0, y0), (z0, y0), (x0, z0), (z0, y0)], [(xf, yf), (zf, yf), (xf, zf), (zf, yf)]
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.add_patch(plt.Circle(pos_0[i], soi_0, fill=False, linestyle=":", color="magenta"))
            ax.add_patch(plt.Circle(pos_f[i], soi_f, fill=False, linestyle=":", color="red"))
    elif graphic_type == "3d":
        cb = axes.scatter3D(*sat_coords, c=time, cmap="viridis", s=.5)
        xs, ys, zs = sphere(soi_0)
        axes.plot_wireframe(xs + x0, ys + y0, zs + z0, alpha=.25, color="magenta")
        xs, ys, zs = sphere(soi_f)
        axes.plot_wireframe(xs + xf, ys + yf, zs + zf, alpha=.25, color="red")
        axes.text(xf, yf, zf, "%s" % sat_obj.body.name, size=8, zorder=1, color="white")
    if cbar: fig.colorbar(cb, ax=axes, orientation="horizontal", label=f"Estimated satellite position after X {timestep}", fraction=.0335)    
    return [fig, axes, graphic_type]

def window_polish(fig, axes, graphic_type, x_window=None, y_window=None, z_window=None) -> list:
    """
    Internal function to draw the motion of a secondary body orbiting a primary (e.g. the Earth about the Sun) and specify axis sharing and gridlines.
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Matplotlib figure object.
    axes : numpy.ndarray (matplotlib axes object)
        Matplotlib axes object.
    graphic_type : str
        Type of graphic to produce, '3d' makes a 3-dimensional graphic, 'planar' plots the projection of a trajectory onto the XY, XZ, and YZ planes.
    x_window : list or None
        Window of values along the X axis to display (default: None) in units of `scale`.
    y_window : list or None
        Window of values along the Y axis to display (default: None) in units of `scale`.
    z_window : list or None
        Window of values along the Z axis to display (default: None) in units of `scale`.
        
    Returns
    -------
    list
        List of length 2 of format `[plt.Figure, plt.Axes]`.
    """
    
    if graphic_type == "planar":
        ((ax1, ax2), (ax3, ax4)) = axes
        if isinstance(x_window, list):
            for ax in [ax1, ax3]: ax.set_xlim(x_window[0], x_window[1])
        if isinstance(y_window, list):
            for ax in [ax1, ax2, ax4]: ax.set_ylim(y_window[0], y_window[1])
        if isinstance(z_window, list):
            for ax in [ax2, ax4]: ax.set_xlim(z_window[0], z_window[1])
            ax3.set_ylim(z_window[0], z_window[1])
        # AXIS SHARING, GRIDLINES
        ax1.sharey(ax2)
        for i, ax in enumerate([ax1, ax2, ax3, ax4]):
            ax.grid(alpha=.5)
            ax.set_axisbelow(True)
            ax.set_aspect('equal', anchor=["SE", "SW", "NE", "NW"][i])
    elif graphic_type == "3d":
        None if x_window is None else axes.axes.set_xlim3d(left=x_window[0], right=x_window[1])
        None if y_window is None else axes.axes.set_ylim3d(bottom=y_window[0], top=y_window[1])
        None if z_window is None else axes.axes.set_zlim3d(bottom=z_window[0], top=z_window[1])
        axes.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axes.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axes.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axes.grid()
        axes.set_axisbelow(True)
        axes.set_aspect("equal")
    return [fig, axes]

def model_2BP(state: list, t: np.ndarray, mu: float) -> list:
    """Function to be integrated to predict the interaction of two celestial bodies."""
    x, y, z, r = [*state[:3], np.linalg.norm(state[:3])]
    x_ddot = -mu * x / r ** 3
    y_ddot = -mu * y / r ** 3
    z_ddot = -mu * z / r ** 3
    return [*state[3:], x_ddot, y_ddot, z_ddot]

def keplerian_2_cartesian(std_grav_param: float, kwargs: dict[str, float], inclination: str = "ecliptic") -> list:
    """
    Function to convert Keplerian orbital elements to Cartesian state vectors.

    Parameters
    ----------
    std_grav_param : float 
        The standard gravitational parameter of the system (m^3 s^-2).
    kwargs : dict
        Dictionary of trajectory parameters including combinations of the following:
        a : float 
            Semi-major axis (m).
        e : float 
            Eccentricity.
        inc : float 
            Inclination (degrees).
        arg_periapsis : float 
            The argument of periapsis (degrees).
        long_asc_node : float 
            The longitude of the ascending node (degrees).
        true_anomaly : float 
            The true anomaly (degrees).
    inclination : str
        The reference plane for inclination measurements (default: 'ecliptic').
 
    Returns
    -------
    np.ndarray
        Cartesian position state vector (m).
    np.ndarray
        Cartesian velocity state vector (m s^-1).

    Raises
    ------
    TypeError
        If the argument containing Keplerian orbital elements is not a dictionary.
    NotImplementedError
        If an orbital element implies the trajectory is parabolic and therefore ambiguous.
    
    Adapted from https://orbital-mechanics.space/classical-orbital-elements/orbital-elements-and-the-state-vector.html#orbital-elements-state-vector
    """

    if not isinstance(kwargs, dict): raise TypeError("Keplerian elements must be defined in a dictionary.")
    a, e = check_sign(kwargs["a"], kwargs["e"])
    if (a == 0) or (e == 1): raise NotImplementedError("Process for defining parabolic trajectories from Keplerian elements not yet available, try providing state vectors instead.")
    try:
        angles = (kwargs["arg_periapsis"], kwargs["inc"][inclination], kwargs["long_asc_node"])
    except:
        angles = (kwargs[angle] for angle in ["arg_periapsis", "inc", "long_asc_node"])
    finally: 
        h = np.sqrt(std_grav_param * np.abs(a) * (e**2 - 1 if e > 1 else 1 - e**2))
        r = semi_latus_rectum(a, e) / (1 + e * sp.cosdg(kwargs["true_anomaly"]))
        r_w = r * np.array([sp.cosdg(kwargs["true_anomaly"]), sp.sindg(kwargs["true_anomaly"]), 0])
        v_w = std_grav_param / h * np.array([-sp.sindg(kwargs["true_anomaly"]), e + sp.cosdg(kwargs["true_anomaly"]), 0])
        return Rotation.from_euler("ZXZ", [-angle for angle in angles], degrees=True).apply([r_w, v_w], inverse=True) 

def cartesian_2_keplerian(std_grav_param: float, vectors: dict[str, float]) -> list:
    """
    Function to convert Cartesian state vectors to Keplerian orbital elements.
    
    Parameters
    ----------
    std_grav_param : float 
        The standard gravitational parameter of the system (m^3 s^-2).
    vectors : dict[str, np.ndarray]
        Dictionary of trajectory parameters including combinations of the following:
        position : np.ndarray
            Cartesian position state vector (m).
        velocity : np.ndarray
            Cartesian velocity state vector (m s^-1).
            
    Returns
    -------
    args : list
        List of Keplerian orbital elements of order:
        a : float 
            Semi-major axis (m).
        e : float 
            Eccentricity.
        inc : float 
            Inclination (degrees).
        arg_periapsis : float 
            The argument of periapsis (degrees).
        long_asc_node : float 
            The longitude of the ascending node (degrees).
        true_anomaly : float 
            The true anomaly (degrees).

    Raises
    ------
    TypeError
        If the argument containing state vectors is not a dictionary.
    ValueError
        If the shape of either the position or velocity is incompatible.
                
    Adapted from https://orbital-mechanics.space/classical-orbital-elements/orbital-elements-and-the-state-vector.html#orbital-elements-state-vector
    """

    if not isinstance(vectors, dict): raise TypeError("State vectors must be defined in a dictionary")
    if (vectors["position"].shape != (3, )) or (vectors["velocity"].shape != (3, )): raise ValueError(f"State vector shapes must be (3, )")
    position, velocity = vectors["position"], vectors["velocity"]
    h_vector = np.cross(position, velocity)
    specific_angular_momentum, r, v = np.linalg.norm((h_vector, position, velocity), axis=1)
    specific_energy = 0.5 * v**2 - std_grav_param / r
    e_vector = eccentricity_vector(position, velocity, std_grav_param)
    a, e = check_sign(-0.5 * std_grav_param / specific_energy, np.linalg.norm(e_vector))
    inc = np.arccos(h_vector[2] / specific_angular_momentum)
    N = np.cross(np.array([0, 0, 1]), h_vector)

    if N[1] >= 0: long_asc = np.arccos(N[0] / np.linalg.norm(N))
    if N[1] < 0: long_asc = 2 * np.pi - np.arccos(N[0] / np.linalg.norm(N))
    if e_vector[2] >= 0: arg_periapsis = np.arccos(np.dot(e_vector, N) / (e * np.linalg.norm(N)))
    if e_vector[2] < 0: arg_periapsis = 2 * np.pi - np.arccos(np.dot(e_vector, N) / (e * np.linalg.norm(N)))
    if radial_velocity(position, velocity) >= 0: true_anom = np.arccos(np.dot(e_vector, position) / (e * r))
    if radial_velocity(position, velocity) < 0: true_anom = 2 * np.pi - np.arccos(np.dot(e_vector, position) / (e * r))
    return a, e, semi_circ(degrees(inc)), degrees(arg_periapsis), degrees(long_asc), degrees(true_anom)

class Body:
    """
    A class used to represent a celestial body.

    ...

    Attributes
    ----------
    body_name : str
        Name of the body.
    body_mass : float
        Mass of the body (kg).
    body_radius : float, optional
        The radius of the body (m) (default 1000).
    negligible : bool
        Whether or not the body's mass and radius is considered to be negligible (for use with artificial satellites) (default: `False`).
    """
    
    def __init__(self, data: dict[str, float], negligible: bool = False):
        """
        Parameters
        ----------
        data : dict
            Dictionary containing values for at least `name` and `mass`.
        negligible : bool
            Whether or not the body's mass and radius is considered to be negligible (for use with artificial satellites) (default: `False`).
        """

        if not "name" in data.keys(): raise KeyError("`name` must be specified.")
        self.body_name = data["name"]
        if not "mass" in data.keys(): raise KeyError("`mass` must be specified.")
        self.body_mass = data["mass"]
        self.body_radius = 1e3 if not "radius" in data.keys() else data["radius"]
        self.__color = None if not "color" in data.keys() else data["color"]
        self.negligible = negligible

    @property
    def name(self) -> str: 
        """Returns the name of the body."""
        return self.body_name
    @property
    def mass(self) -> float: 
        """Returns the mass of the body."""
        if self.negligible: return 1
        return self.body_mass
    @property
    def radius(self) -> float: 
        """Returns the radius of the body."""
        if self.negligible: return 0.0
        return self.body_radius
    @property
    def color(self) -> str: return self.__color
    
    def __round__(self, n): return Body(self.name, round(self.mass, n), round(self.radius, n))
    def __repr__(self): return self.name
    def __str__(self):
        rounded = round(self, 2)
        return f"Body - {rounded.name}: Mass-{rounded.mass}kg, Radius-{rounded.radius / 1e3}km"

    def __ne__(self, other): 
        if self.negligible or other.negligible: return False if self.negligible and other.negligible else True
        return (self.mass != other.mass)
    def __eq__(self, other): 
        if self.negligible or other.negligible: return True if self.negligible and other.negligible else False
        return (self.mass == other.mass)
    def __gt__(self, other): 
        if self.negligible or other.negligible:
            if self.negligible and other.negligible: return False
            return False if self.negligible else True
        return (self.mass > other.mass)
    def __lt__(self, other): 
        if self.negligible or other.negligible:
            if self.negligible and other.negligible: return False
            return True if self.negligible else False
        return (self.mass < other.mass)
    def __ge__(self, other): 
        if self.negligible or other.negligible:
            if self.negligible and other.negligible: return True
            return False if self.negligible else True
        return (self.mass >= other.mass)
    def __le__(self, other): 
        if self.negligible or other.negligible:
            if self.negligible and other.negligible: return True
            return True if self.negligible else False
        return (self.mass <= other.mass)

    def keys(self): return ["name", "mass", "radius"]
    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, val): sef[key] = value
        
class System:
    """
    A class used to represent a celestial body.

    ...

    Attributes
    ----------
    primary : Body
        Primary gravitational presence of the system.
    satellites : dict
        Dictionary of all satellites in the system.
    system_mass : float
        Total mass of the system (kg).
    system_name : str
        The name of the system and its constituent satellites.

    Methods
    -------
    add_satellite(secondary, args)
        Adds a satellite to the system.
    simulate(duration, timestep="days", resolution=10000, solver="best", check_escape=True, check_collision=False)
        Predicts the motion of the satellite about the primary celestial body, and returns results of the simulation as an attribute of the simulated satellite.
    plot_simulation(graphic_type, scale="AU", markevery=5, encounter=None, timestep="days", x_window=None, y_window=None, z_window=None)
        Plots the motion of the system satellites according to simulation results.
    """
    
    def __init__(self, primary: Body, inclination: str = "ecliptic"):
        """
        Parameters
        ----------
        primary : Body
            Primary gravitational presence of the system.
        inclination : str
            The reference plane for inclination measurements (default: 'ecliptic').
        """
        
        self.primary = primary
        self.satellites = dict()
        self.system_mass = self.mass
        self.system_name = self.name
        self.__inclination = inclination

    @property
    def name(self) -> str: 
        """Returns the name of the whole system."""
        return namer(self)
    @property
    def body_name(self) -> str:
        """Return primary body name."""
        return self.primary.body_name
    @property
    def mass(self) -> float: 
        """Returns the total mass of the system."""
        return mass_calculator(self)
    @property
    def body_mass(self) -> float:
        """Return primary body mass."""
        return self.primary.body_mass
    @property
    def color(self) -> str: return self.primary.color
    @property
    def inclination(self) -> str: return self.__inclination

    def __iter__(self): return iter(self.satellites.items())
    def __next__(self):
        while True:
            try:
                return next(self.satellites.items())
            except StopIteration:
                break
    
    def __len__(self): return len(self.satellites)
    def __repr__(self): return self.name
    def __str__(self): return self.name
        
    def __ne__(self, other): return (self.mass != other.mass)
    def __eq__(self, other): return (self.mass == other.mass)
    def __gt__(self, other): return (self.mass > other.mass)
    def __lt__(self, other): return (self.mass < other.mass)
    def __ge__(self, other): return (self.mass >= other.mass)
    def __le__(self, other): return (self.mass <= other.mass)
        
    def __contains__(self, item):
        if isinstance(item, str):
            for sat_name, sat_obj in self:
                if (sat_name == item) or (sat_obj.name == item): return True
                if len(sat_obj) > 0: return item in sat_obj
        elif isinstance(item, Body) or isinstance(item, Satellite):
            for sat_name, sat_obj in self:
                if (sat_obj.body_name == item.body_name) or (sat_obj.name == item.body_name): return True
                if len(sat_obj) > 0: item in sat_obj
        return False
    
    def __get_pred_lengths(self) -> np.ndarray:
        pred_lengths = np.array([])
        for satname, sat_obj in self:
            pred_lengths = np.append(pred_lengths, len(sat_obj.prediction))
            if len(sat_obj) > 0: pred_lengths = np.concatenate((pred_lengths, sat_obj.get_pred_lengths()))
        return pred_lengths.astype(int)

    def __truncate(self, lengths: np.ndarray):
        """Internal class method to ensure the simulated data arrays are of the same length."""
        for satname, sat_obj in self: sat_obj.force_truncate(lengths.min())
            
    def __check_encounters(self):
        """Class method to systematically compare each pair of satellite's coordinates for an encounter."""
        _satellites = [sat_obj for satname, sat_obj in self]
        encounter_indeces = []
        while len(_satellites) > 1:
            # Popping the list of satellites avoids duplicate comparison between satellite pairs
            satA = _satellites.pop()
            satA_coords = satA.prediction.global_coords
            for satX in _satellites:
                satX_coords = satX.prediction.global_coords
                distance = np.linalg.norm(np.subtract(satA_coords, satX_coords), axis=1)
                if not satA.negligible: distance = np.subtract(distance, satA.SOI)
                if not satX.negligible: distance = np.subtract(distance, satX.SOI)
                if np.any(np.nonzero(distance < 0.0)):
                    idx = np.where(distance < 0.0)[0][0]
                    print(f"Encounter found between {satA.body.name} and {satX.body.name} at time {satA.prediction.time_array[idx] / timestep_conversion['days']} days")
                    encounter_indeces.append(idx)
        if len(encounter_indeces) > 0: self.__truncate(np.array(encounter_indeces))
            
    def add_satellite(self, secondary: Body, kwargs: dict[str, float]):
        """
        Adds a satellite to the system.

        Parameters
        ----------
        secondary : Body
            The celestial body to assign a trajectory to.
        kwargs : dict
            Dictionary of trajectory parameters including combinations of the following:
            a : float 
                Semi-major axis (m).
            e : float 
                Eccentricity.
            inc : float 
                Inclination (degrees).
            arg_periapsis : float 
                The argument of periapsis (degrees).
            long_asc_node : float 
                The longitude of the ascending node (degrees).
            true_anomaly : float 
                The true anomaly (degrees).
            position : array
                Cartesian position state vector (m).
            velocity : array
                Cartesian velocity state vector (m s^-1).
                
        Raises
        ------
        ValueError
            If the satellite is more massive than the primary body.
        """
        
        if self.primary > secondary:
            self.satellites[secondary.name] = Satellite(secondary, kwargs, primary=self, inclination=self.inclination)
            self.system_name = self.name
            self.system_mass = self.mass
        else:
            raise ValueError("Satellite more massive than System primary.")

    def simulate(self, duration: float | int, timestep: str = "days", resolution: int = 10000, solver: str = "best", check_escape: bool = True, check_collision: bool = False, check_encounter=True):
        """
        Predicts the motion of the satellite about the primary celestial body, and returns results of the simulation as an attribute of the simulated satellite.
        
        Parameters
        ----------
        duration : float or int
            Length of time to simulate a trajectory in `timestep`.
        timestep : str
            Conversion factor from duration in `timestep` to duration in seconds. Defined timesteps: `seconds`, `minutes`, `hours`, `days`, `years` (default: `days`).
        resolution : int
            Number of equally-spaced times between zero and `duration`.
        solver : str
            Method used to predict the motion of the satellite (default: 'best').
        check_escape : bool
            If `True`, all radial distances will be compared to the system's sphere of influence. If the radial distance exceeds the SOI, all data up to the time where the satellite leaves the system's SOI will be returned (default: `True`).
        check_escape : bool
            If `True`, all radial distances will be compared to the radius of the primary body. If the radial distance does not exceed the primary body's radius, all data up to the time where the satellite collides with the primary will be returned (default: `False`).
        check_encounter : bool
            If `False`, an encounter will not be checked for.

        Raises
        ------
        ValueError
            If the length of time to simulate a trajectory is non-finite or negative.
        KeyError
            If the specified timestep is not defined.
        """

        if (not np.isfinite(duration)) or (0 > duration): raise ValueError("Simulation duration must be a finite, nonnegative number.")
        if not timestep in timestep_conversion.keys(): raise KeyError(f"Timestep conversion not defined for timestep {timestep}.")
        times = np.linspace(0, duration * timestep_conversion[timestep], resolution)

        for satname, sat_obj in self:
            sat_obj.prediction = Predict(sat_obj, times, solver=solver, check_escape=check_escape, check_collision=check_collision)
            if len(sat_obj) > 0: sat_obj.simulate(duration, timestep=timestep, resolution=resolution, solver=solver, check_escape=check_escape, check_collision=check_collision)
        self.__truncate(self.__get_pred_lengths())
        if check_encounter: self.__check_encounters()
    
    def plot_simulation(self, graphic_type, scale: str = "AU", markevery: int = 5, timestep: str = "days", x_window=None, y_window=None, z_window=None):
        """
        Function to draw the motion of a simulated system.
        Should only be called after `simulate`.
        
        Parameters
        ----------
        graphic_type : str
            Type of graphic to produce, '3d' makes a 3-dimensional graphic, 'planar' plots the projection of a trajectory onto the XY, XZ, and YZ planes.
        scale : str
            Spatial scale to make results more readable (default: 'AU'). Must be the same string passed to `makefig`.
        timestep : str
            Timestep to re-scale the colorbar representing the simulation time to (default: 'days').
        markevery : int or None
            Number of points to skip when plotting, primarily used to speed up and simplify the plotting process by plotting fewer points (default : 5).
        x_window : list or None
            Window of values along the X axis to display (default: None) in units of `scale`.
        y_window : list or None
            Window of values along the Y axis to display (default: None) in units of `scale`.
        z_window : list or None
            Window of values along the Z axis to display (default: None) in units of `scale`.

        Raises
        ------
        AttributeError
            If a satellite has not been simulated.
        ValueError
            If `graphic_type` is not '3d' or 'planar'.
        KeyError
            If either `scale` or `timestep` conversions are undefined.
        """
        
        plt.close()
        has_cbar = False
        if not graphic_type in ["3d", "planar"]: raise ValueError("Graphic type not '3d' or 'planar'.")
        if not scale in spatial_conversion.keys(): raise KeyError(f"Spatial conversion scale {scale} not found.")
        if not timestep in timestep_conversion.keys(): raise KeyError(f"Timestep conversion {timestep} not found.")

        figobj = makefig(graphic_type, scale=scale)
        figobj = central_obj(*figobj, self.primary, scale=scale)
        for satname, sat_obj in self:
            if not hasattr(sat_obj, "prediction"): raise AttributeError(f"Satellite {satname} has not been simulated.")
            figobj = plot_satellite(*figobj, sat_obj, scale=scale, markevery=markevery, timestep=timestep, cbar= not has_cbar)
            if not has_cbar: has_cbar = True
            if len(sat_obj) > 0: figobj = sat_obj.plot_satsys(*figobj, scale=scale, markevery=markevery, timestep=timestep, has_cbar=True)
        figobj = window_polish(*figobj, x_window=x_window, y_window=y_window, z_window=z_window)
        plt.show()

class Trajectory(Body):
    """
    A class used to represent a static trajectory.

    ...

    Attributes
    ----------
    orbit : str
        Type of trajectory (circular, elliptical, parabolic, or hyperbolic) based on the eccentricity.
    a : float 
        Semi-major axis (m).
    e : float 
        Eccentricity.
    inc : float 
        Inclination (degrees).
    arg_periapsis : float 
        The argument of periapsis (degrees).
    long_asc_node : float 
        The longitude of the ascending node (degrees).
    true_anomaly : float 
        The true anomaly (degrees).
    position : array
        Cartesian position state vector (m).
    velocity : array
        Cartesian velocity state vector (m s^-1).
        
    Methods
    -------
    plot_potential(y_window, r_max)
        Method to calculate and display the effective potential of an orbiting body.
    """
    
    def __init__(self, kwargs: dict[str, float], to_set: dict = {}, inclination: str = "ecliptic"):
        """
        Parameters
        ----------
        kwargs : dict
            Dictionary of trajectory parameters including combinations of the following:
            a : float 
                Semi-major axis (m) (default 1).
            e : float 
                Eccentricity (default 0).
            inc : float 
                Inclination (degrees) (default 0).
            arg_periapsis : float 
                The argument of periapsis (degrees) (default 0).
            long_asc_node : float 
                The longitude of the ascending node (degrees) (default 0).
            true_anomaly : float 
                The true anomaly (degrees) (default 0).
            position : array
                Cartesian position state vector (m) (default [1e6, 0, 0]).
            velocity : array
                Cartesian velocity state vector (m s^-1) (default [0, 1e3, 0]).
        to_set : dict
            Dictionary for seperately storing elements to be changed apart from elements from an original trajectory, not intended to be used directly.
        inclination : str
            Type of inclination reference plane to be used (default: ecliptic), though if the dictionary supplied does not specify the reference plane, it is assumed that the angle provided is measured with respect to the default plane.
                
        Raises
        ------
        TypeError
            If the input parameters are not formatted as a dictionary.
        NotImplementedError
            If the input parameters contain both Keplerian and Cartesian elements and the trajectory is ambiguous.
        """

        if not isinstance(kwargs, dict): raise TypeError("Trajectory object can only be created using a dictionary of parameters")
        self.__inclination = inclination
        default = {"a": 1.0,
                   "e": 0.0,
                   "inc": 0.0,
                   "arg_periapsis": 0.0,
                   "long_asc_node": 0.0,
                   "true_anomaly": 0.0,
                   "position": np.array([1e6, 0, 0]),
                   "velocity": np.array([0, 1e3, 0])}
        if len(to_set) == 0:
            if any(element in kepler for element in kwargs) and any(element in cartes for element in kwargs):
                raise NotImplementedError("Process to set a trajectory using both Keplerian and Cartesian parameters does not exist yet.")
            elif any(element in kepler for element in kwargs):
                self._from_kepler_({element: default[element] if not element in kwargs else kwargs[element] for element in kepler})
            elif any(element in cartes for element in kwargs):
                self._from_state_vectors_({element: default[element] if not element in kwargs else kwargs[element] for element in cartes})
        elif len(to_set) == 1:
            if "a" in to_set.keys():
                a, e = check_sign(to_set["a"], kwargs["e"])
                if can_keep_position(kwargs["position"], a, e):
                    self.__align_and_rotate(kwargs["position"], a, e)
                else:
                    self.__placeholder(kwargs, to_set)
            elif "e" in to_set.keys():
                a, e = check_sign(kwargs["a"], to_set["e"])
                if can_keep_position(kwargs["position"], a, e):
                    self.__align_and_rotate(kwargs["position"], a, e)
                else:
                    self.__placeholder(kwargs, to_set)
            else:
                self.__placeholder(kwargs, to_set)
        elif len(to_set) == 2:
            if ("a" in to_set.keys()) and ("e" in to_set.keys()):
                a, e = check_sign(to_set["a"], to_set["e"])
                if can_keep_position(kwargs["position"], a, e):
                    self.__align_and_rotate(kwargs["position"], a, e)
                else:
                    self.__placeholder(kwargs, to_set)
            else:
                self.__placeholder(kwargs, to_set)
        else:
            self.__placeholder(kwargs, to_set)

    def __repr__(self): return f"{self.orbit} Trajectory - a:{self.a}m, e:{self.e}, inc:{self.inc}deg"
    def __str__(self): return f"{self.orbit} Trajectory - a:{self.a}m, e:{self.e}, inc:{self.inc}deg"
    
    def keys(self): return ["orbit", *kepler, *cartes]
    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, val): sef[key] = value
        
    def _from_kepler_(self, kwargs: dict[str, float]):
        if kwargs["e"] < 0: raise ValueError("Eccentricity cannot be less than zero.")
        a, e = check_sign(kwargs["a"], kwargs["e"])
        position, velocity = keplerian_2_cartesian(self.gparam, kwargs, inclination=self.__inclination)
        if isinstance(kwargs["inc"], dict):
            inc = kwargs["inc"][self.__inclination]
        else:
            inc = kwargs["inc"]
        self.orbit, self.a, self.e, self.inc, self.arg_periapsis, self.long_asc_node, self.true_anomaly, self.position, self.velocity = self._orbit(eccentricity=e), a, e, semi_circ(inc), kwargs["arg_periapsis"], kwargs["long_asc_node"], kwargs["true_anomaly"], position, velocity

    def _from_state_vectors_(self, kwargs: dict[str, np.ndarray]):
        a, e, inc, arg_periapsis, long_asc_node, true_anomaly = cartesian_2_keplerian(self.gparam, kwargs)
        self.orbit, self.a, self.e, self.inc, self.arg_periapsis, self.long_asc_node, self.true_anomaly, self.position, self.velocity = self._orbit(eccentricity=e), a, e, semi_circ(inc), arg_periapsis, long_asc_node, true_anomaly, kwargs["position"], kwargs["velocity"]

    def _orbit(self, eccentricity=None) -> str:
        if eccentricity == None: eccentricity = self.e
        if eccentricity < 0: raise ValueError("Eccentricity cannot be less than zero.")
        if eccentricity == 0: return "Circular"
        if eccentricity < 1: return "Elliptical"
        if eccentricity == 1: return "Parabolic"
        if eccentricity > 1: return "Hyperbolic"

    def __placeholder(self, original, to_set):
        """Functional placeholder method for updating trajectory parameters without extra consideration for maintaining original elements."""
        original.update(to_set)
        if any(element in kepler for element in to_set) and any(element in cartes for element in to_set):
            raise NotImplementedError("Process to set a trajectory using both Keplerian and Cartesian parameters does not exist yet.")
        elif any(element in kepler for element in to_set):
            self._from_kepler_({element: original[element] for element in kepler})
        elif any(element in cartes for element in to_set):
            self._from_state_vectors_({element: original[element] for element in cartes})

    def __align_and_rotate(self, position, a, e):
        r = np.linalg.norm(position)
        self.true_anomaly = degrees(np.arccos((a * (1 - e**2) - r) / (e * r)))
        r_w = r * np.array([sp.cosdg(self.true_anomaly), sp.sindg(self.true_anomaly), 0])
        v_w = self.gparam / np.linalg.norm(self.specific_angular_momentum) * np.array([-sp.sindg(self.true_anomaly), e + sp.cosdg(self.true_anomaly), 0])
        align = self.rotation.align_vectors(np.array([r_w]), np.array([position]))[0]
        self.velocity = align.apply([v_w], inverse=True)
        
    @property
    def radial(self) -> float: return np.linalg.norm(self.position)
    @property
    def radial_velocity(self) -> float: return radial_velocity(self.position, self.velocity)
    @property
    def speed(self) -> float: return np.linalg.norm(self.velocity)
    @property
    def kepler(self) -> dict[str, float]:
        """Return dictionary of the 6 Keplerian orbital elements."""
        return {element: getattr(self, element) for element in kepler}
    @property
    def state(self) -> dict[str, np.ndarray]:
        """Return dictionary of the state vectors."""
        return {"position": self.position, "velocity": self.velocity}
    @property
    def elements(self) -> dict:
        """Return dictionary of all orbital elements."""
        return {element: getattr(self, element) for element in [*kepler, *cartes]}
    @property
    def gparam(self) -> float: 
        """Returns the standard gravitational parameter (m^3 s^-2) of the satellite and its primary"""
        if isinstance(self.primary, System): return grav_param(self.primary.primary.mass, self.body.mass)
        return grav_param(self.primary.body.mass, self.body.mass)
    @property
    def excess_velocity(self) -> float: 
        """Returns the hyperbolic excess velocity of the trajectory if the trajectory is hyperbolic or pararbolic, otherwise returns NaN."""
        if self.orbit == "Hyperbolic": return np.sqrt(-self.gparam / self.a)
        if self.orbit == "Parabolic": return 0
        return np.nan
    @property
    def impact_parameter(self) -> float: 
        """Returns the impact parameter (semi-minor axis) of a hyperbolic trajectory, otherwise returns NaN."""
        if self.orbit == "Hyperbolic": return self.a * np.sqrt(self.e**2 - 1)
        return np.nan
    @property
    def mean_angular_motion(self) -> float:
        """Returns the mean angular motion of the trajectory."""
        if self.e == 1: return mean_angular_motion(self.gparam)
        return mean_angular_motion(self.gparam, a=self.a)
    @property
    def eccentricity_vector(self) -> np.ndarray: 
        """Returns the eccentricity vector of the trajectory."""
        return eccentricity_vector(self.position, self.velocity, self.gparam)
    @property
    def period(self) -> float: 
        """Returns the orbital period of the trajectory, or +inf for unbound orbits."""
        return np.inf if self.e >= 1 else 2 * np.pi * np.sqrt(self.a**3 / self.gparam)
    @property
    def eccentric_anomaly(self) -> float: 
        """Returns the eccentric anomaly (in degrees) of the trajectory."""
        return true_2_eccentric_anom(self.true_anomaly, self.e)
    @property
    def flight_path_angle(self) -> float: 
        """Returns the flight path angle (in degrees) of the trajectory."""
        if self.orbit == "Parabolic": return self.true_anomaly / 2
        true_anomaly = radians(self.true_anomaly)
        return degrees(np.arctan((self.e * sp.sindg(true_anomaly)) / (1 + self.e * sp.cosdg(true_anomaly))))
    @property
    def rotation(self):
        """Return the SciPy Rotation object corresponding to the defined Euler angles."""
        return Rotation.from_euler("ZXZ", [-getattr(self, angle) for angle in ["arg_periapsis", "inc", "long_asc_node"]], degrees=True)
    @property
    def specific_energy(self) -> float:
        """Computes the specific orbital energy of the trajectory."""
        if self.e < 0: raise ValueError("Eccentricity less than zero")
        if self.e == 1: return 0
        epsilon = self.gparam / (2 * self.a)
        if self.e < 1: return -epsilon
        if self.e > 1: return epsilon
    @property
    def specific_angular_momentum(self) -> np.ndarray:
        """Returns the specific angular momentum vector."""
        return np.cross(self.position, self.velocity)

    def plot_potential(self, y_window=None, r_max=None):
        """
        Function to swiftly plot the effective potential of an orbiting body.
        
        Parameters
        ----------
        y_window : list or None
            The window of energies to display in the plot (default: None).
        r_max : float or None
            The maximum radial distance to consider in potential estimation (default: None).
        """
        
        plt.close()
        plot_energy(self, y_window=y_window, r_max=r_max)
    
class Satellite(Trajectory):
    """
    A class used to represent a satellite and its trajectory in the system with respect to the immediate primary.

    ...

    Attributes
    ----------
    primary : System, Satellite
        The system or satellite serving as a primary for the satellite.
    body_name : float 
        Name of the satellite.
    satellites: dict
        Dictionary of all satellites of the satellite.
    body : Body
        Body object representing the satellite.
    orbit : str
        Type of trajectory (circular, elliptical, parabolic, or hyperbolic) based on the eccentricity.
    a : float 
        Semi-major axis (m).
    e : float 
        Eccentricity.
    inc : float 
        Inclination (degrees).
    arg_periapsis : float 
        The argument of periapsis (degrees).
    long_asc_node : float 
        The longitude of the ascending node (degrees).
    true_anomaly : float 
        The true anomaly (degrees).
    position : array
        Cartesian position state vector (m).
    velocity : array
        Cartesian velocity state vector (m s^-1).
        
    Methods
    -------
    add_satellite(secondary, args)
        Adds a satellite to the satellite.
    simulate(duration, timestep="days", resolution=10000, solver="best", check_escape=True, check_collision=False)
        Predicts the motion of the satellites about the satellite the method is called on.
    plot_simulation(graphic_type, scale="AU", markevery=5, encounter=None, timestep="days", x_window=None, y_window=None, z_window=None)
        Plots the motion of the system satellites according to simulation results.
    set_elements(args)
        Sets multiple orbital elements, keeping unspecified elements constant (if possible).
    set_semimajor_axis(a)
        Directly sets semi-major axis, keeping other elements constant (if possible).
    set_eccentricity(e)
        Directly sets eccentricity, keeping other elements constant (if possible).
    set_inclination(inc):
        Directly sets inclination, keeping other elements constant (if possible).
    set_arg_periapsis(arg_periapsis):
        Directly sets argument of periapsis, keeping other elements constant (if possible).
    set_long_asc_node(long_asc_node):
        Directly sets longitude of ascending node, keeping other elements constant (if possible).
    set_true_anomaly(anomaly)
        Directly sets true anomaly, keeping other elements constant (if possible).
    set_eccentric_anomaly(anomaly):
        Directly sets eccentric anomaly, keeping other elements constant (if possible).
    set_position(position):
        Directly sets position, keeping existing velocity vector.
    set_velocity(velocity): 
        Directly sets velocity, keeping existing position vector.
    """
    
    def __init__(self, body: Body, args: dict[str, float], primary=None, inclination: str = "ecliptic"):
        """
        A class used to represent a satellite and its trajectory in the system with respect to the immediate primary.
    
        ...
    
        Parameters
        ----------
        primary : System, Satellite
            The system or satellite serving as a primary for the satellite (default: None).
        body : Body
            The celestial object to assign a trajectory to.
        args : dict
            Dictionary of trajectory parameters including combinations of the following:
            a : float 
                Semi-major axis (m).
            e : float 
                Eccentricity.
            inc : float 
                Inclination (degrees).
            arg_periapsis : float 
                The argument of periapsis (degrees).
            long_asc_node : float 
                The longitude of the ascending node (degrees).
            true_anomaly : float 
                The true anomaly (degrees).
            position : array
                Cartesian position state vector (m).
            velocity : array
                Cartesian velocity state vector (m s^-1).
        inclination : str
            The reference plane for inclination measurements (default: 'ecliptic').
        """
        
        self.primary = primary
        self.body = body
        self.negligible = body.negligible
        self.satellites = dict()
        self.__inclination = inclination
        super().__init__(args, inclination=self.__inclination)

    @property
    def name(self) -> str: return namer(self)
    @property
    def mass(self) -> float: return mass_calculator(self)
    @property
    def radius(self) -> float: return self.body.radius
    @property
    def color(self) -> str: return self.body.color
    @property
    def inclination(self) -> str: return self.__inclination
    @property
    def SOI(self) -> float: 
        """Returns the radius of the gravitational sphere of influence at the satellite's radial distance from primary."""
        return r_soi(self.radial, self.primary.mass, self.body.mass)

    def __len__(self): return len(self.satellites)
    def __repr__(self): return f"Satellite of {satellite_root(self.primary)} - {self.body.name}"
    def __str__(self): return f"Satellite of {self.primary.name} - {self.body.name}"
        
    def keys(self): return ["body", "orbit", "primary", *kepler, *cartes]
    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, val): self[key] = value

    def __iter__(self): return iter(self.satellites.items())
    def __next__(self):
        while True:
            try:
                return next(self.satellites.items())
            except StopIteration:
                break
    
    def __contains__(self, item):
        if isinstance(item, str):
            for sat_name, sat_obj in self:
                if (sat_name == item) or (sat_obj.body.name == item): return True
                if len(sat_obj) > 0: return item in sat_obj
        elif isinstance(item, Body) or isinstance(item, Satellite):
            for sat_name, sat_obj in self:
                if (sat_obj.body.name == item.body.name) or (sat_obj.name == item.name): return True
                if len(sat_obj) > 0: return item in sat_obj
        return False
        
    def __check_encounters(self):
        """Class method to systematically compare each pair of satellite's coordinates for an encounter."""
        _satellites = [sat_obj for satname, sat_obj in self]
        encounter_indeces = []
        while len(_satellites) > 1:
            # Popping the list of satellites avoids duplicate comparison between satellite pairs
            satA = _satellites.pop()
            satA_coords = satA.prediction.global_coords
            for satX in _satellites:
                satX_coords = satX.prediction.global_coords
                distance = np.linalg.norm(np.subtract(satA_coords, satX_coords), axis=1)
                if not satA.negligible: distance = np.subtract(distance, satA.SOI)
                if not satX.negligible: distance = np.subtract(distance, satX.SOI)
                if np.any(np.nonzero(distance < 0.0)):
                    idx = np.where(distance < 0.0)[0][0]
                    print(f"Encounter found between {satA.body_name} and {satX.body_name} at index {idx}")
                    encounter_indeces.append(idx)
        if len(encounter_indeces) > 0: self.force_truncate(np.array(encounter_indeces).min())
            
    def set_elements(self, kwargs: dict[str, float]):
        """
        Simple method for setting either Keplerian orbital parameters or Cartesian state vectors based on provided and existing parameters.
        
        Parameters
        ----------
        kwargs : dict
            Dictionary of trajectory parameters including combinations of the following:
            a : float 
                Semi-major axis (m).
            e : float 
                Eccentricity.
            inc : float 
                Inclination (degrees).
            arg_periapsis : float 
                The argument of periapsis (degrees).
            long_asc_node : float 
                The longitude of the ascending node (degrees).
            true_anomaly : float 
                The true anomaly (degrees).
            position : array
                Cartesian position state vector (m).
            velocity : array
                Cartesian velocity state vector (m s^-1).
                
        Raises
        ------
        NotImplementedError
            If the trajectorial parameters to set include both Keplerian orbital parameters and Cartesian state vectors, and neither definition is fully calculable.
        """
        
        if any(element in kepler for element in kwargs) and any(element in cartes for element in kwargs):
            if all(element in kwargs for element in cartes):
                # if the element dict has both state vectors, use ONLY those vectors
                super().__init__({"position": kwargs["position"], "velocity": kwargs["velocity"]}, inclination=self.__inclination)
            elif all(element in kwargs for element in kepler):
                # if the element dict has all Keplerian elements, use ONLY those elements
                super().__init__(kwargs, inclination=self.__inclination)
            else:
                raise NotImplementedError("Process to set a trajectory using both Keplerian and Cartesian parameters currently unnavailable.")
        elif any(element in kepler for element in kwargs):
            super().__init__(self.elements, to_set=kwargs, inclination=self.__inclination)
        elif any(element in cartes for element in kwargs):
            super().__init__(self.elements, to_set=kwargs, inclination=self.__inclination)

    def set_semimajor_axis(self, a: float):
        """Method directly setting semi-major axis."""
        elements = self.kepler; elements["a"] = a
        if (self.e > 1) and (a > 0): a *= -1
        if (self.e < 1) and (a < 0): raise ValueError("Semi-major axis < 0 implies hyperbolic trajectory but eccentricity does not match.")
        super().__init__(self.elements, to_set={"a": a}, inclination=self.__inclination)
        
    def set_eccentricity(self, e: float):
        """Method directly setting eccentricity."""
        if e < 0: raise ValueError("Eccentricity less than zero.")
        if (self.a < 0) and (e < 1): self.a *= -1
        if (self.a > 0) and (e > 1): self.a *= -1
        super().__init__(self.elements, to_set={"e": e}, inclination=self.__inclination)
        
    def set_inclination(self, inc: float): 
        """Method directly setting inclination."""
        super().__init__(self.elements, to_set={"inc": inc}, inclination=self.__inclination)
        
    def set_arg_periapsis(self, arg_periapsis: float): 
        """Method directly setting argument of periapsis."""
        super().__init__(self.elements, to_set={"arg_periapsis": arg_periapsis}, inclination=self.__inclination)

    def set_long_asc_node(self, long_asc_node: float): 
        """Method directly setting longitude of ascending node."""
        super().__init__(self.elements, to_set={"long_asc_node": long_asc_node}, inclination=self.__inclination)
        
    def set_true_anomaly(self, anomaly: float):
        """Method directly setting true anomaly."""
        if self.orbit == "Hyperbolic":
            true_anom_inf = degrees(np.arccos(-1 / self.e))
            if np.abs(anomaly) > true_anom_inf: raise ValueError(f"Anomaly for this trajectory cannot not exceed {true_anom_inf} degrees")
        if self.orbit == "Parabolic":
            if np.abs(anomaly) > 180: raise ValueError("Anomaly must not exceed 180 degrees.")
        super().__init__(self.elements, to_set={"true_anomaly": anomaly}, inclination=self.__inclination)
        
    def set_eccentric_anomaly(self, anomaly: float): 
        """Method directly setting eccentric anomaly."""
        elements = self.kepler; elements["true_anomaly"] = eccentric_2_true_anom(anomaly, self.e)
        super().__init__(self.elements, to_set={"true_anomaly": eccentric_2_true_anom(anomaly, self.e)}, inclination=self.__inclination)
        
    def set_position(self, position: np.ndarray): 
        """Method directly setting position vector."""
        if position.shape != (3, ): raise ValueError(f"Shape of position must be (3, )")
        super().__init__({"position": position, "velocity": self.velocity}, inclination=self.__inclination)
        
    def set_velocity(self, velocity: np.ndarray): 
        """Method directly setting velocity vector."""
        if velocity.shape != (3, ): raise ValueError(f"Shape of velocity must be (3, )")
        super().__init__({"position": self.position, "velocity": velocity}, inclination=self.__inclination)

    def add_satellite(self, secondary_body: Body, kwargs: dict[str, float]):
        """
        Adds a satellite to the satellite.
        
        Parameters
        ----------
        secondary_body : Body
            The Body to add as a satellite of the current satellite.
        kwargs : dict
            Dictionary of trajectory parameters including combinations of the following:
            a : float 
                Semi-major axis (m).
            e : float 
                Eccentricity.
            inc : float 
                Inclination (degrees).
            arg_periapsis : float 
                The argument of periapsis (degrees).
            long_asc_node : float 
                The longitude of the ascending node (degrees).
            true_anomaly : float 
                The true anomaly (degrees).
            position : array
                Cartesian position state vector (m).
            velocity : array
                Cartesian velocity state vector (m s^-1).

        Raises
        ------
        ValueError
            If the secondary body is more massive than the satellite.
        """
        
        if not self.primary > secondary_body: raise ValueError("Secondary more massive than primary.")
        self.satellites[secondary_body.name] = Satellite(secondary_body, kwargs, primary=self, inclination=self.__inclination)
        self.satellite_system_name = self.name

    def simulate(self, duration: float | int, timestep: str = "days", resolution: int = 10000, solver: str = "best", check_escape: bool = True, check_collision: bool = False, check_encounter: bool = True):
        """
        Predicts the motion of the satellite about the primary celestial body, and returns results of the simulation as an attribute of the simulated satellite.
        
        Parameters
        ----------
        duration : float or int
            Length of time to simulate a trajectory in `timestep`.
        timestep : str
            Conversion factor from duration in `timestep` to duration in seconds. Defined timesteps: `seconds`, `minutes`, `hours`, `days`, `years` (default: `days`).
        resolution : int
            Number of equally-spaced times between zero and `duration`.
        solver : str
            Method used to predict the motion of the satellite (default: 'best').
        check_escape : bool
            If `True`, all radial distances will be compared to the system's sphere of influence. If the radial distance exceeds the SOI, all data up to the time where the satellite leaves the system's SOI will be returned (default: `True`).
        check_escape : bool
            If `True`, all radial distances will be compared to the radius of the primary body. If the radial distance does not exceed the primary body's radius, all data up to the time where the satellite collides with the primary will be returned (default: `False`).
        check_encounter : bool
            If `False`, will not check for an encounter.

        Raises
        ------
        ValueError
            If the length of time to simulate a trajectory is non-finite or negative.
        KeyError
            If the specified timestep is not defined.
        """

        if (not np.isfinite(duration)) or (0 > duration): raise ValueError("Simulation duration must be a finite, nonnegative number.")
        if not timestep in timestep_conversion.keys(): raise KeyError(f"Timestep conversion not defined for timestep {timestep}.")
        if len(self) == 0: return

        times = np.linspace(0, duration * timestep_conversion[timestep], resolution)
        for satname, sat_obj in self:
            sat_obj.prediction = Predict(sat_obj, times, solver=solver, check_escape=check_escape, check_collision=check_collision)
            if len(sat_obj) > 0: sat_obj.simulate(duration, timestep=timestep, resolution=resolution, solver=solver, check_escape=check_escape, check_collision=check_collision)
        self.force_truncate(self.get_pred_lengths().min())
        if check_encounter: self.__check_encounters()

    def plot_simulation(self, graphic_type: str, scale: str = "AU", markevery: int = 5, timestep: str = "days", x_window=None, y_window=None, z_window=None) -> list:
        """
        Function to draw the motion of a simulated system.
        Should only be called after `simulate`.
        
        Parameters
        ----------
        graphic_type : str
            Type of graphic to produce, '3d' makes a 3-dimensional graphic, 'planar' plots the projection of a trajectory onto the XY, XZ, and YZ planes.
        scale : str
            Spatial scale to make results more readable (default: 'AU'). Must be the same string passed to `makefig`.
        markevery : int or None
            Number of points to skip when plotting, primarily used to speed up and simplify the plotting process by plotting fewer points (default : 5).
        timestep : str
            Timestep to re-scale the colorbar representing the simulation time to (default: 'days').
        x_window : list or None
            Window of values along the X axis to display (default: None) in units of `scale`.
        y_window : list or None
            Window of values along the Y axis to display (default: None) in units of `scale`.
        z_window : list or None
            Window of values along the Z axis to display (default: None) in units of `scale`.

        Raises
        ------
        AttributeError
            If a satellite has not been simulated.
        ValueError
            If `graphic_type` is not '3d' or 'planar'.
        KeyError
            If either `scale` or `timestep` conversions are undefined.
        """
        
        plt.close()
        has_cbar = False
        if not graphic_type in ["3d", "planar"]: raise ValueError("Graphic type not '3d' or 'planar'.")
        if not scale in spatial_conversion.keys(): raise KeyError(f"Spatial conversion scale {scale} not found.")
        if not timestep in timestep_conversion.keys(): raise KeyError(f"Timestep conversion {timestep} not found.")
            
        figobj = makefig(graphic_type, scale=scale)
        figobj = central_obj(*figobj, self, scale=scale)
        figobj = self.plot_satsys(*figobj, relative_to=self, scale=scale, markevery=markevery, timestep=timestep, has_cbar=False)
        figobj = window_polish(*figobj, x_window=x_window, y_window=y_window, z_window=z_window)
        plt.show()

    def plot_satsys(self, fig: plt.Figure, axes: plt.Axes, graphic_type: str, relative_to=0, scale: str = "AU", markevery: int = None, timestep: str = "days", has_cbar: bool = True) -> list:
        """Internal, public method to plot all satellites of another satellite."""
        for satname, sat_obj in self:
            if not hasattr(sat_obj, "prediction"): raise AttributeError(f"Satellite {satname} has not been simulated.")
            figobj = plot_satellite(*[fig, axes, graphic_type], sat_obj, relative_to=relative_to, scale=scale, markevery=markevery, timestep=timestep, cbar=not has_cbar)
            if not has_cbar: has_cbar = True
            if len(sat_obj) > 0: figobj = sat_obj.plot_satsys(*figobj, relative_to=relative_to, scale=scale, markevery=markevery, timestep=timestep, has_cbar=True)
        return [fig, axes, graphic_type]

    def get_pred_lengths(self) -> np.ndarray:
        """Internal, public method to compile a list of the lengths of all simulated satellites, primarily for determination of truncation length."""
        predict_lengths = np.array([])
        for satname, sat_obj in self:
            predict_lengths = np.append(predict_lengths, len(sat_obj.prediction))
            if len(sat_obj) > 0: predict_lengths = np.concatenate((predict_lengths, sat_obj.get_pred_lengths()))
        return predict_lengths.astype(int)

    def force_truncate(self, idx: int):
        """Internal, public method to force all predictions to be of equal length."""
        if hasattr(self, "prediction"): self.prediction.truncate(idx)
        for satname, sat_obj in self:
            sat_obj.force_truncate(idx)
            if len(sat_obj) > 0: sat_obj.force_truncate(idx)
            
class Predict:
    """
    A class used to represent the motion of a satellite in the system with respect to the immediate primary.

    ...

    Attributes
    ----------
    object : str
        Name of the satellite.
    solver : str
        Method used to predict the state vectors of the orbiting satellite.
    time_array : array
        Span of time (in seconds) across which to simulate a trajectory.
    true_anomaly : array 
        Predicted true anomaly (degrees) for time in `time_array`.
    local_coords : array
        Predicted Cartesian position state vector (m) for time in `time_array` relative to the primary object.
    local_velocity : array
        Predicted Cartesian velocity state vector (m s^-1) for time in `time_array` relative to the primary object.

    Methods
    -------
    truncate(idx)
        Truncates prediction results to specified index.
    """
    
    def __init__(self, satellite: Satellite, time_array: np.ndarray, solver: str = "best", check_escape: bool = True, check_collision: bool = False):
        """
        Predicts the motion of the satellite about the primary celestial body.
        
        Parameters
        ----------
        primary : Body
            The primary gravitational object about which the satellite moves.
        satellite : Satellite
            Satellite object to predict motion of.
        time_array : array
            Span of time (in seconds) across which to simulate a trajectory.
        solver : str
            Method used to predict the motion of the satellite (default: 'best').
        check_escape : bool
            If `True`, all radial distances will be compared to the system's sphere of influence. If the radial distance exceeds the SOI, all data up to the time where the satellite leaves the system's SOI will be returned (default: `True`).
        check_escape : bool
            If `True`, all radial distances will be compared to the radius of the primary body. If the radial distance does not exceed the primary body's radius, all data up to the time where the satellite collides with the primary will be returned (default: `False`).

        Raises
        ------
        ValueError
            If the secondary body is more massive than the satellite.
        KeyError
            If the requested solver method is undefined. Defined solvers: 'best', 'integrate'.
        """
        
        if satellite.primary < satellite: raise ValueError("Primary body more massive than satellite.")
        self.__satellite = satellite
        self.__primary = satellite.primary
        self.object = satellite.name
        if solver == "best":
            if satellite.orbit == "Circular":
                self.solver = "Anomaly: Algebraic"
                self.__iternomaly(time_array, degrees(satellite.mean_angular_motion * time_array))
            elif satellite.orbit == "Elliptical":
                self.solver = "Anomaly: Bessel Summation"
                self.__iternomaly(time_array, mean_2_true_anom(mean_anomaly_shift(satellite.true_anomaly, satellite.e) + degrees(satellite.mean_angular_motion * time_array), satellite.e))
            elif satellite.orbit == "Parabolic":
                self.__integrate(time_array)
            elif satellite.orbit == "Hyperbolic":
                self.solver = "Anomaly: Newton Solver"
                # Adapted from https://orbital-mechanics.space/time-since-periapsis-and-keplers-equation/hyperbolic-trajectory-example.html
                h = np.linalg.norm(self.__satellite.specific_angular_momentum)
                t_0 = h**3 / self.__satellite.gparam**2 * 1 / (self.__satellite.e**2 - 1)**(3 / 2)
                Mh_arr = self.__satellite.mean_angular_motion * (time_array - t_0)
                hyp_anoms = np.full_like(time_array, np.nan)
                for idx, M_h in enumerate(Mh_arr):
                    hyp_anoms[idx] = degrees(newton(
                        func=lambda F, M_h, e: e * np.sinh(F) - F - M_h,
                        fprime=lambda F, M_h, e: e * np.cosh(F) - 1,
                        fprime2=lambda F, M_h, e: e * np.sinh(F),
                        x0=0, args=(M_h, self.__satellite.e)))
                # Need to store the indeces where the anomaly is negative to accurately convert the hyperbolic to true anomaly
                # This also serves to establish an intuitive interpretation of where the satellite is along its trajectory with respect to pericenter
                isneg = np.where(hyp_anoms > degrees(np.arccos(-1 / self.__satellite.e)))
                hyp_anoms[isneg] -= 360
                true_anoms = np.apply_along_axis(hyperbolic_2_true_anom, 0, np.abs(hyp_anoms), self.__satellite.e)
                true_anoms[isneg] *= -1
                self.__iternomaly(time_array, true_anoms)
        elif solver == "integrate":
            self.__integrate(time_array)
        else:
            raise KeyError(f"Prediction method {solver} not defined. Use 'best' or 'integrate'.")
        
        if check_escape:
            if isinstance(self.__primary, System):
                # Check the velocity WRT system escape velocity
                # Truncation is not necessary as the system SOI is assumed to approach infinity
                pass
            else:
                if np.any(np.nonzero(self.radial > np.full_like(self.time_array, self.__primary.SOI))):
                    wrt_soi = np.greater(self.radial[:-1], np.full((len(self) - 1,), self.__primary.SOI))
                    diff_radial = np.greater(np.diff(self.radial), np.zeros(len(self) - 1))
                    idx = np.nonzero(np.logical_and(wrt_soi, diff_radial))[0][0]
                    print(f"Satellite {self.object} escapes primary {self.__primary.body.name} at time {time_array[idx] / timestep_conversion['days']} days")
                    self.truncate(idx)
        if check_collision:
            if isinstance(self.__primary, System):
                if np.any(np.nonzero(self.radial < np.full_like(self.time_array, self.__primary.primary.radius))):
                    idx = np.where(self.radial < np.full_like(self.time_array, self.__primary.primary.radius))[0][0]
                    self.truncate(idx)
                    print(f"Satellite {self.object} impacts primary {self.__primary.body.name} at time {time_array[idx] / timestep_conversion['days']} days")
            else:
                if np.any(np.nonzero(self.radial < np.full_like(self.time_array, self.__primary.radius))):
                    idx = np.where(self.radial < np.full_like(self.time_array, self.__primary.radius))[0][0]
                    self.truncate(idx)
                    print(f"Satellite {self.object} impacts primary {self.__primary.body.name} at time {time_array[idx] / timestep_conversion['days']} days")

    def __len__(self): return len(self.time_array)
    def __repr__(self): return f"Predicted motion of {self.object}"
    def __str__(self): return f"Predicted motion of {self.object}"
        
    def keys(self): return ["object", "solver", "time_array", "true_anomaly", "local_coords", "local_velocity"]
    def __getitem__(self, key): return getattr(self, key)
    def __setitem__(self, key, val): self[key] = value
        
    @property
    def radial(self) -> np.ndarray: return np.linalg.norm(self.local_coords, axis=1)
    @property
    def speed(self) -> np.ndarray: return np.linalg.norm(self.local_velocity, axis=1)
    @property
    def global_coords(self) -> np.ndarray:
        """Compute the coordinates of the satellite relative to the system."""
        if isinstance(self.__primary, System): return np.full((len(self), 3,), self.local_coords)
        if not hasattr(self.__primary, "prediction"): return self.local_coords + np.full((len(self), 3,), self.__primary.position)
        return self.local_coords + self.__primary.prediction.global_coords
    @property
    def global_velocity(self) -> np.ndarray:
        """Compute the velocity of the body relative to the system."""
        if isinstance(self.__primary, System): return np.full((len(self), 3,), self.local_velocity)
        if not hasattr(self.__primary, "prediction"): return self.local_velocity + np.full((len(self), 3,), self.__primary.velocity)
        return self.local_velocity + self.__primary.prediction.global_velocity
        
    def __iternomaly(self, time_array: np.ndarray, true_anoms: np.ndarray):
        """Internal function for iteration through true anomalies to compute state vectors"""
        self.time_array, self.true_anomaly = time_array, true_anoms
        self.local_coords, self.local_velocity = np.full((len(self), 3,), np.nan), np.full((len(self), 3,), np.nan)
        keplerian_elements = {element:getattr(self.__satellite, element) for element in kepler}
        for idx, true_anom in enumerate(self.true_anomaly):
            keplerian_elements["true_anomaly"] = true_anom
            self.local_coords[idx, :], self.local_velocity[idx, :] = keplerian_2_cartesian(self.__satellite.gparam, keplerian_elements)

    def __integrate(self, time_array: np.ndarray):
        """Internal function for integration of state vectors"""
        self.solver = "ODE Integration"
        self.time_array = time_array
        state = np.array([*self.__satellite.position.T, *self.__satellite.velocity.T]) * 1e-3
        solution = odeint(model_2BP, state, self.time_array, args=(self.__satellite.gparam * 1e-9, )) * 1e3
        local_coords, local_velocity = solution[:, :3], solution[:, 3:]
        self.true_anomaly = np.array([true_anom_from_vectors(pos, vel, self.__satellite.gparam) for pos, vel in zip(local_coords, local_velocity)])
        radial_velocity = np.array([np.dot(pos, vel) for pos, vel in zip(unit(local_coords), local_velocity)])
        self.true_anomaly[np.where(radial_velocity < 0)] *= -1 
        self.local_coords, self.local_velocity = local_coords, local_velocity

    def truncate(self, idx: int):
        """Method to truncate a solution depending on whether a satellite impacts a primary or escapes a primary's SOI."""
        self.time_array = self.time_array[0:idx]
        self.true_anomaly = self.true_anomaly[0:idx]
        self.local_coords = self.local_coords[0:idx, :]
        self.local_velocity = self.local_velocity[0:idx, :]

def get_system(func):
    """Wrapper for obtaining the base System object and returning a deep copy of it and two specific satellites."""
    def get_primary(obj):
        while not isinstance(obj, System): return get_primary(getattr(obj, "primary"))
        return deepcopy(obj)
    def wrapper(satA, satB):
        # If necessary, flip satellites to always determine the motion of the less massive object relative to the more massive object
        if satB > satA: satA, satB = satB, satA
        return func(get_primary(satA), deepcopy(satA), deepcopy(satB))
    return wrapper

def update_system(func):
    """"Wrapper updating a deep copy of a System with the positions of satellites after simulation."""
    def update_satellite_system(satsys):
        for satname, sat_obj in satsys:
            elements = sat_obj.kepler; elements["true_anomaly"] = sat_obj.prediction.true_anomaly[-1] % 360
            state_vectors = {"position": sat_obj.prediction.local_coords[-1, :], "velocity": sat_obj.prediction.local_velocity[-1, :]}
            del sat_obj.prediction
            sat_obj.set_elements(elements if elements["e"] < 1 else state_vectors)
            if len(sat_obj) > 0: update_satellite_system(sat_obj)
    def wrapper(system, satA, satB):
        update_satellite_system(system)
        try: 
            system = func(system.satellites[satA.body.name], system.satellites[satB.body.name], system=system)
        except KeyError:
            system = func(system.satellites[satA.body.name], system.satellites[satA.body.name].satellites[satB.body.name], system=system)
        finally:
            # Displaying the returned System object tends to look wrong, setting it as a variable and showing the variable seems to "fix" it
            # The above does not affect functionality, just displays wrong at times
            system.system_name = namer(system)
            return system
    return wrapper

@get_system
@update_system
def encounter(satA, satB, system=None):
    """Function creating a replicate System object with updated parameters reflecting an encounter and the location of other satellites at the end of the simulated time."""
    rel_posB, rel_velB = satB.position - satA.position, satB.velocity - satA.velocity
    system.satellites[satA.body.name].add_satellite(satB.body, {"position": rel_posB, "velocity": rel_velB})
    del system.satellites[satB.body.name]
    return system

@get_system
@update_system
def escape(satA, satB, system=None):
    """Function creating a replicate System object with updated parameters reflecting an escaping satellite the location of other satellites at the end of the simulated time."""
    rel_posB, rel_velB = satB.position + satA.position, satB.velocity + satA.velocity
    system.add_satellite(satB.body, {"position": rel_posB, "velocity": rel_velB})
    del system.satellites[satA.body.name].satellites[satB.body.name]
    return system
    
def solar_system(inclination="sun_eq"):
    """Returns a System object with all eight planets and the Moon defined."""
    solsys = System(Body(sun), inclination=inclination)
    for planet in [*terrans, *giants]: solsys.add_satellite(Body(planet), planet)
    solsys.satellites["Earth"].add_satellite(Body(moon), moon)
    return solsys

def inner_planets(inclination="sun_eq"):
    """Returns a System object with the four terrestrial planets and the Moon defined."""
    solsys = System(Body(sun), inclination=inclination)
    for planet in terrans: solsys.add_satellite(Body(planet), planet)
    solsys.satellites["Earth"].add_satellite(Body(moon), moon)
    return solsys

def earth_moon(inclination="ecliptic"):
    """Returns a System object with the Earth, Sun, and Moon."""
    solsys = System(Body(sun), inclination=inclination)
    solsys.add_satellite(Body(earth), earth)
    solsys.satellites["Earth"].add_satellite(Body(moon), moon)
    return solsys

def outer_planets(inclination="sun_eq"):
    """Returns a System object with the four outer planets defined."""
    solsys = System(Body(sun), inclination=inclination)
    for planet in giants: solsys.add_satellite(Body(planet), planet)
    return solsys