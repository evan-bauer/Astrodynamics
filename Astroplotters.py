import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

G = 6.67430e-11                #  N m^2 kg^-2
m_probe = 900                  #  kg
AU = 149597871000              #  m
cos, sin = np.cos(np.linspace(-np.pi, np.pi)), np.sin(np.linspace(-np.pi, np.pi))

def reduced_mass(M, m):
    return (M * m) / (M + m)

def U_gravitational(r, M, m):
    return -G * M * m / r

def U_centrifugal(r, M, m, h):
    return h**2 / (2 * reduced_mass(M, m) * r**2)

def U_effective(r, M, m, h):
    return U_gravitational(r, M, m) + U_centrifugal(r, M, m, h)

def plot_energy(obj, y_window=1e13, r_max=1e8):
    plt.style.use('default')
    if y_window == None: y_window = 1e13
    
    # CREATE AXES
    fig, ax = plt.figure(figsize=(6, 6)), plt.axes()
    r_peri, r_apo = obj.periapsis.distance, obj.apoapsis.distance
    if np.isinf(obj.T):
        if r_max == None: r_max = 1e8
        r = np.linspace(10, r_max * 1.1)
        rs = np.linspace(r_peri, r_max * 1.1)
    else:
        r = np.linspace(10, r_apo * 1.1)
        rs = np.linspace(r_peri, r_apo)
    r_0 = r[U_effective(r, obj.about.mass, obj.secondary_mass, obj.h).argmin()]
    ax.plot(r, U_gravitational(r, obj.about.mass, obj.secondary_mass), c='r', label=r'$U_{grav}$')
    ax.plot(r, U_centrifugal(r, obj.about.mass, obj.secondary_mass, obj.h), c='g', label=r'$U_{cf}$')
    ax.plot(r, U_effective(r, obj.about.mass, obj.secondary_mass, obj.h), c='b', label=r'$U_{eff}$')
    ax.axhline(obj.energy, c='k', label=r"$E$")
    ax.axvline(r_peri, dashes=(1, 1), c='orange', label=r"$r_{peri}$")
    ax.axvline(r_apo, dashes=(1, 1), c='cyan', label=r"$r_{apo}$") if not r_apo == np.inf else None
    ax.axvline(r_0, c='magenta', dashes=(1, 1), label=r"$r_{circ}$")
    ax.set_title(f"Effective Potential of {obj}")
    ax.set_xlabel(r"Radial Distance (m)")
    ax.set_ylabel(r"Effective Potential")
    ax.grid()
    ax.set_xlim(0,  r.max())
    ax.set_ylim(-y_window, y_window)
    ax.fill_between(rs, np.zeros_like(rs) + obj.energy, U_effective(rs, obj.about.mass, obj.secondary_mass, obj.h), alpha=.25)
    if np.isinf(r_apo):
        ax.set_xticks([0, r_peri, r_0, r.max()], [0, r"$r_{peri}$", r"$r_{circ}$", "max"])
    else:
        ax.set_xticks([0, r_peri, r_0, r_apo, r.max()], [0, r"$r_{peri}$", r"$r_{circ}$", r"$r_{apo}$", "max"])
    ax.set_axisbelow(True)
    plt.legend(loc='upper right')
    plt.show()

def plot_multi_simulation(sim_obj, scale="AU", x_window=None, y_window=None, z_window=None, encounter=None, markevery=None):
    scale_factor = 1e3                        # m -> km
    if scale == "AU": scale_factor = AU       # m -> AU
        
    # DEFINE PLOT ELEMENT OBJECTS AND BASIC PROPERTIES
    plt.style.use('dark_background')
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex="col", figsize=(10, 10), gridspec_kw={'hspace': 0, 'wspace': 0})
    ((ax1, ax2), (ax3, ax4)) = axes
        
    # TITLES, LABELS, AND TICK LOCATIONS
    fig.suptitle("Predicted Trajectories of Provided System", y=.95)
    
    ax1.set_title(f'XY Plane')
    ax1.set_xlabel(f'X [{scale}]')
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.set_ylabel(f'Y [{scale}]')
    ax1.yaxis.set_label_position("left")

    ax2.set_title(f'YZ Plane')
    ax2.set_xlabel(f'Z [{scale}]')
    ax2.xaxis.tick_top()
    ax2.xaxis.set_label_position("top")
    ax2.set_ylabel(f'Y [{scale}]')
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    ax3.set_title(f'XZ Plane', y=0, pad=-45)
    ax3.set_xlabel(f'X [{scale}]')
    ax3.xaxis.set_label_position("bottom")
    ax3.set_ylabel(f'Z [{scale}]')

    ax4.set_title(f'YZ Plane', y=0, pad=-45)
    ax4.set_xlabel(f'Z [{scale}]')
    ax4.set_ylabel(f'Y [{scale}]')
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.tick_right()
    
    # CALCULATE ALL SOI COORDINATES
    # SUN COORDINATES
    x_sun = sim_obj.system.primary.radius * cos / scale_factor
    y_sun = sim_obj.system.primary.radius * sin / scale_factor

    ax1.plot(0, 0, color='orange', alpha=0.7)
    ax1.fill_between(x_sun, -y_sun, y_sun, color='orange')
    
    ax2.plot(0, 0, color='orange', alpha=0.7)
    ax2.fill_between(x_sun, -y_sun, y_sun, color='orange')
    
    ax3.plot(0, 0, color='orange', alpha=0.7)
    ax3.fill_between(x_sun, -y_sun, y_sun, color='orange')
    
    ax4.plot(0, 0, color='orange', alpha=0.7)
    ax4.fill_between(x_sun, -y_sun, y_sun, color='orange')

    # PLOT PREDICTED SECONDARY TRAJECTORY
    if hasattr(sim_obj, "secondary"):
        p_secondary = ax1.scatter(sim_obj.secondary.orbit.position[0][::markevery] / scale_factor, sim_obj.secondary.orbit.position[1][::markevery] / scale_factor, c=sim_obj.t[::markevery] / (24 * 3600), cmap="cool", s=2)
        ax2.scatter(sim_obj.secondary.orbit.position[2][::markevery] / scale_factor, sim_obj.secondary.orbit.position[1][::markevery] / scale_factor, c=sim_obj.t[::markevery] / (24 * 3600), cmap="cool", s=2)
        ax3.scatter(sim_obj.secondary.orbit.position[0][::markevery] / scale_factor, sim_obj.secondary.orbit.position[2][::markevery] / scale_factor, c=sim_obj.t[::markevery] / (24 * 3600), cmap="cool", s=2)
        ax4.scatter(sim_obj.secondary.orbit.position[2][::markevery] / scale_factor, sim_obj.secondary.orbit.position[1][::markevery] / scale_factor, c=sim_obj.t[::markevery] / (24 * 3600), cmap="cool", s=2)
        fig.colorbar(p_secondary, ax=axes, orientation="horizontal", label='Secondary', fraction=.0335)    
        
        # SECONDARY SOI
        soi = sim_obj.secondary.SOI
        x_soi_secondary_initial_cos = (soi * cos + sim_obj.secondary.orbit.position[0][0]) / scale_factor
        y_soi_secondary_initial_cos = (soi * cos + sim_obj.secondary.orbit.position[1][0]) / scale_factor
        y_soi_secondary_initial_sin = (soi * sin + sim_obj.secondary.orbit.position[1][0]) / scale_factor
        z_soi_secondary_initial_cos = (soi * cos + sim_obj.secondary.orbit.position[2][0]) / scale_factor
        z_soi_secondary_initial_sin = (soi * sin + sim_obj.secondary.orbit.position[2][0]) / scale_factor
        
        x_soi_secondary_final_cos = (soi * cos + sim_obj.secondary.orbit.position[0][-1 if encounter == None else encounter]) / scale_factor
        y_soi_secondary_final_cos = (soi * cos + sim_obj.secondary.orbit.position[1][-1 if encounter == None else encounter]) / scale_factor
        y_soi_secondary_final_sin = (soi * sin + sim_obj.secondary.orbit.position[1][-1 if encounter == None else encounter]) / scale_factor
        z_soi_secondary_final_cos = (soi * cos + sim_obj.secondary.orbit.position[2][-1 if encounter == None else encounter]) / scale_factor
        z_soi_secondary_final_sin = (soi * sin + sim_obj.secondary.orbit.position[2][-1 if encounter == None else encounter]) / scale_factor

        ax1.plot(x_soi_secondary_initial_cos, y_soi_secondary_initial_sin, color='magenta', dashes=(1, 1))
        ax1.plot(x_soi_secondary_final_cos, y_soi_secondary_final_sin, color='red', dashes=(1, 1))
        
        ax2.plot(z_soi_secondary_initial_sin, y_soi_secondary_initial_cos, color='magenta', dashes=(1, 1))
        ax2.plot(z_soi_secondary_final_sin, y_soi_secondary_final_cos, color='red', dashes=(1, 1))
        
        ax3.plot(x_soi_secondary_initial_cos, z_soi_secondary_initial_sin, color='magenta', dashes=(1, 1))
        ax3.plot(x_soi_secondary_final_cos, z_soi_secondary_final_sin, color='red', dashes=(1, 1))    
        
        ax4.plot(z_soi_secondary_initial_sin, y_soi_secondary_initial_cos, color='magenta', dashes=(1, 1))
        ax4.plot(z_soi_secondary_final_sin, y_soi_secondary_final_cos, color='red', dashes=(1, 1))

    # PLOT PREDICTED PRIMARY TRAJECTORY
    if hasattr(sim_obj, "primary"):
        p_primary = ax1.scatter(sim_obj.primary.orbit.position[0][::markevery] / scale_factor, sim_obj.primary.orbit.position[1][::markevery] / scale_factor, c=sim_obj.t[::markevery] / (24 * 3600), cmap="spring", s=2)
        ax2.scatter(sim_obj.primary.orbit.position[2][::markevery] / scale_factor, sim_obj.primary.orbit.position[1][::markevery] / scale_factor, c=sim_obj.t[::markevery] / (24 * 3600), cmap="spring", s=2)
        ax3.scatter(sim_obj.primary.orbit.position[0][::markevery] / scale_factor, sim_obj.primary.orbit.position[2][::markevery] / scale_factor, c=sim_obj.t[::markevery] / (24 * 3600), cmap="spring", s=2)
        ax4.scatter(sim_obj.primary.orbit.position[2][::markevery] / scale_factor, sim_obj.primary.orbit.position[1][::markevery] / scale_factor, c=sim_obj.t[::markevery] / (24 * 3600), cmap="spring", s=2)
        fig.colorbar(p_primary, ax=axes, orientation="horizontal", label='Primary', fraction=.04075)    

        # PRIMARY SOI
        soi = sim_obj.primary.SOI
        x_soi_primary_initial_cos = (soi * cos + sim_obj.primary.orbit.position[0][0]) / scale_factor
        y_soi_primary_initial_cos = (soi * cos + sim_obj.primary.orbit.position[1][0]) / scale_factor
        y_soi_primary_initial_sin = (soi * sin + sim_obj.primary.orbit.position[1][0]) / scale_factor
        z_soi_primary_initial_cos = (soi * cos + sim_obj.primary.orbit.position[2][0]) / scale_factor
        z_soi_primary_initial_sin = (soi * sin + sim_obj.primary.orbit.position[2][0]) / scale_factor
        
        x_soi_primary_final_cos = (soi * cos + sim_obj.primary.orbit.position[0][-1 if encounter == None else encounter]) / scale_factor
        y_soi_primary_final_cos = (soi * cos + sim_obj.primary.orbit.position[1][-1 if encounter == None else encounter]) / scale_factor
        y_soi_primary_final_sin = (soi * sin + sim_obj.primary.orbit.position[1][-1 if encounter == None else encounter]) / scale_factor
        z_soi_primary_final_cos = (soi * cos + sim_obj.primary.orbit.position[2][-1 if encounter == None else encounter]) / scale_factor
        z_soi_primary_final_sin = (soi * sin + sim_obj.primary.orbit.position[2][-1 if encounter == None else encounter]) / scale_factor
        
        ax1.plot(x_soi_primary_initial_cos, y_soi_primary_initial_sin, color='magenta', dashes=(1, 1))
        ax1.plot(x_soi_primary_final_cos, y_soi_primary_final_sin, color='red', dashes=(1, 1))
        
        ax2.plot(z_soi_primary_initial_sin, y_soi_primary_initial_cos, color='magenta', dashes=(1, 1))
        ax2.plot(z_soi_primary_final_sin, y_soi_primary_final_cos, color='red', dashes=(1, 1))
        
        ax3.plot(x_soi_primary_initial_cos, z_soi_primary_initial_sin, color='magenta', dashes=(1, 1))
        ax3.plot(x_soi_primary_final_cos, z_soi_primary_final_sin, color='red', dashes=(1, 1))

        ax4.plot(z_soi_primary_initial_sin, y_soi_primary_initial_cos, color='magenta', dashes=(1, 1))
        ax4.plot(z_soi_primary_final_sin, y_soi_primary_final_cos, color='red', dashes=(1, 1))
        
    # PLOT PREDICTED PROBE TRAJECTORIES AND DEFINE COLORBAR COLORMAP STARTING WITH PROBE AVAILABILITY
    if hasattr(sim_obj, "probe"):
        p_probe = ax1.scatter(sim_obj.probe.orbit.position[0][::markevery] / scale_factor, sim_obj.probe.orbit.position[1][::markevery] / scale_factor, c=sim_obj.t[::markevery] / (24 * 3600), cmap="viridis", s=2)
        ax2.scatter(sim_obj.probe.orbit.position[2][::markevery] / scale_factor, sim_obj.probe.orbit.position[1][::markevery] / scale_factor, c=sim_obj.t[::markevery] / (24 * 3600), cmap="viridis", s=2)
        ax3.scatter(sim_obj.probe.orbit.position[0][::markevery] / scale_factor, sim_obj.probe.orbit.position[2][::markevery] / scale_factor, c=sim_obj.t[::markevery] / (24 * 3600), cmap="viridis", s=2)
        ax4.scatter(sim_obj.probe.orbit.position[2][::markevery] / scale_factor, sim_obj.probe.orbit.position[1][::markevery] / scale_factor, c=sim_obj.t[::markevery] / (24 * 3600), cmap="viridis", s=2)
        fig.colorbar(p_probe, ax=axes, orientation="horizontal", label='Probe', fraction=.05)    
        
    # DISPLAY WINDOWS
    if not x_window == None:
        ax1.set_xlim(x_window[0], x_window[1])
        ax3.set_xlim(x_window[0], x_window[1])
    if not y_window == None:
        ax1.set_ylim(y_window[0], y_window[1])
        ax2.set_ylim(y_window[0], y_window[1])
        ax4.set_ylim(y_window[0], y_window[1])
    if not z_window == None:
        ax2.set_xlim(z_window[0], z_window[1])
        ax3.set_ylim(z_window[0], z_window[1])
        ax4.set_xlim(z_window[0], z_window[1])

    # AXIS SHARING, GRIDS
    ax1.sharey(ax2)
    
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax3.set_axisbelow(True)
    ax4.set_axisbelow(True)
    
    ax1.set_aspect('equal', anchor="SE")
    ax2.set_aspect('equal', anchor="SW")
    ax3.set_aspect('equal', anchor="NE")
    ax4.set_aspect("equal", anchor="NW")
        
    plt.show()