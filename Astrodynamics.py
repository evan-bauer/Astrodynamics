import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.signal import argrelmin
from Astroplotters import *

G = 6.67430e-11                #  N m^2 kg^-2
m_probe = 900                  #  kg
AU = 149597871000              #  m

sun = {"ecc":0,
       "sma":0,
       "inc":0,
       "long_asc":0,
       "arg_peri":0,
       "radius":6.957e8,
       "mass":1.98847e30}
earth = {"ecc":0.0167086,
         "sma":149598023000, 
         "inc":7.155 / 360 * (2 * np.pi),
         "long_asc":-11.26064 / 360 * (2 * np.pi), 
         "arg_peri":114.20783 / 360 * (2 * np.pi), 
         "radius":6371000, 
         "mass":5.97217e24}
moon = {"ecc":0.0549, 
        "sma":384399000, 
        "inc":5.145 / 360 * (2 * np.pi),
        "long_asc":0, 
        "arg_peri":0,
        "radius":1737400,
        "mass":7.342e22}
jupiter = {"ecc":0.0489,
           "sma":5.2038 * AU, 
           "inc":6.09 / 360 * (2 * np.pi),
           "long_asc":100.464 / 360 * (2 * np.pi), 
           "arg_peri":273.867 / 360 * (2 * np.pi), 
           "radius":69911000, 
           "mass":1.8982e27}

def period(a, primary_mass):
    return 2 * np.pi * np.sqrt(a**3 / (G * primary_mass))

def visviva(M, r, a):
    return np.sqrt(G * M * ((2 / r) - (1 / a)))

def hohmann(r1, r2, M):
    return np.sqrt(G * M / r1) * (np.sqrt(2 * r2 / (r1 + r2)) - 1)

def r_soi(M, m, r):
    return r * (m / M) ** (2 / 5)

def magnitude(vector):
    return np.linalg.norm(vector)

def eccentricity_vector(r_vector, v_vector, h_vector, mu):
    return np.cross(v_vector, h_vector) / mu - r_vector / magnitude(r_vector)

def orbit_energy(r_vector, v_vector, mu):
    return magnitude(v_vector)**2 / 2 - mu / magnitude(r_vector)

def true_2_eccentric_anom(true_anom, ecc):
    return np.arctan((np.sqrt(1 - ecc**2) * np.sin(true_anom))/(ecc + np.cos(true_anom)))
    
def model_2BP(state, t, M_primary):
    mu = G * 1e-9 * M_primary  # Gravitational parameter  [km^3/s^2]
    x, y, z, x_dot, y_dot, z_dot = state
    x_ddot = -mu * x / magnitude(np.array([x, y, z])) ** 3
    y_ddot = -mu * y / magnitude(np.array([x, y, z])) ** 3
    z_ddot = -mu * z / magnitude(np.array([x, y, z])) ** 3
    return [x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot]

def kep_2_cart(a, e, i, arg_peri, long_asc, ecc_anom, mu):
#     Function adapted from https://space.stackexchange.com/questions/19322/converting-orbital-elements-to-cartesian-state-vectors/19335#19335
    nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(ecc_anom / 2))
    r = a * (1 - e * np.cos(ecc_anom))
    h = np.sqrt(mu * a * (1 - e**2))
    r_peri = a * (1 - e**2)
    x = r * (np.cos(long_asc) * np.cos(arg_peri + nu) - np.sin(long_asc) * np.sin(arg_peri + nu) * np.cos(i))
    y = r * (np.sin(long_asc) * np.cos(arg_peri + nu) + np.cos(long_asc) * np.sin(arg_peri + nu) * np.cos(i))
    z = r * (np.sin(i) * np.sin(arg_peri + nu))
    v_x = (x * h * e / (r * r_peri)) * np.sin(nu) - (h / r) * (np.cos(long_asc) * np.sin(arg_peri + nu) + np.sin(long_asc) * np.cos(arg_peri + nu) * np.cos(i))
    v_y = (y * h * e / (r * r_peri)) * np.sin(nu) - (h / r) * (np.sin(long_asc) * np.sin(arg_peri + nu) - np.cos(long_asc) * np.cos(arg_peri + nu) * np.cos(i))
    v_z = (z * h * e / (r * r_peri)) * np.sin(nu) + (h / r) * (np.cos(arg_peri + nu) * np.sin(i))
    return np.array([x, y, z]), np.array([v_x, v_y, v_z])

def cart_2_kep(position, velocity, mu):
#     Function adapted from https://space.stackexchange.com/questions/19322/converting-orbital-elements-to-cartesian-state-vectors/19335#19335
    h_vector = np.cross(position, velocity)
    h = np.linalg.norm(h_vector)
    r = np.linalg.norm(position)
    v = np.linalg.norm(velocity)
    specific_energy = 0.5 * v**2 - mu / r
    a = -mu / (2 * specific_energy)
    e = np.sqrt(1 - h**2 / (a * mu))
    e_vector = eccentricity_vector(position, velocity, h_vector, mu)
    inclination = np.arccos(h_vector[2] / h)
    long_asc = np.arctan2(h_vector[0], -h_vector[1])
    lat = np.arctan2(np.divide(position[2], np.sin(inclination)), (position[0] * np.cos(long_asc) + position[1] * np.sin(long_asc)))
    p = a * (1 - e**2)
    nu = np.arctan2(np.sqrt(p / mu) * np.dot(position, velocity), p - r)
    arg_periapsis = lat - nu
    ecc_anom = 2 * np.arctan(np.sqrt((1 - e) / (1 + e)) * np.tan(nu / 2))
    return a, e, e_vector, magnitude(h_vector), h_vector, inclination, arg_periapsis, long_asc, ecc_anom, specific_energy

class Apsis:
    def __init__(self, apse):
        self.distance = apse["distance"]
        self.velocity = apse["velocity"]
        
    def __repr__(self):
        return f"Apsis at r={round(self.distance * 1e-3, 1)}km, v={round(self.velocity * 1e-3, 3)}km/s"
        
class OrbitalProperties:
    def __init__(self, source, data, mode=None, secondary_mass=m_probe, arg_periapsis=0, long_asc_node=0, inclination=0):
        self.secondary_mass = secondary_mass
        if source == "planet" or source == "circular":
            system, a = data
            self.orbit = "Circular"
            self.about = system
            self.a = a
            self.e = 0.0
            self.T = period(self.a, system.mass)
            self.h = self.a * visviva(system.mass, self.a, self.a)
            self.energy = - G * system.mass / (2 * self.a)
        elif source == "ellipse":
            self.orbit = "Elliptical"
            if mode == "apses":
                primary, r_peri, r_apo = data
                assert r_peri <= r_apo
                self.about = primary
                self.a = (r_peri + r_apo) / 2
                if r_peri == r_apo: self.orbit = "Circular"
                self.e = (r_apo - r_peri) / (2 * self.a)
                self.T = period(self.a, primary.mass)
                self.h = np.sqrt(G * primary.mass * self.secondary_mass**2 * self.a * (1 - self.e**2))
                self.energy = - G * primary.mass * self.secondary_mass / (2 * self.a)
            else:
                raise NotImplementedError
        elif source == "parabolic":
            self.orbit = "Parabolic"
            self.T = np.inf
            self.a = np.inf
            self.e = 1
            self.energy = 0
            if mode == "periapsis":
                primary, r_periapsis = data
                self.about = primary
                self.h = np.sqrt(2 * G * (self.about.mass + self.secondary_mass) * r_periapsis)
            else:
                raise NotImplementedError
        elif source == "hyperbolic":
            self.orbit = "Hyperbolic"
            self.T = np.inf
            if mode == "initial conditions" or mode == "simulate":
                initial_conds, self.about = data
                x_0, y_0, z_0, vx_0, vy_0, vz_0 = initial_conds
                self.position, self.velocity = np.array([x_0, y_0, z_0]), np.array([vx_0, vy_0, vz_0])
                self.a = 1 / ((2 / magnitude(self.position) - magnitude(self.velocity)**2 / (G * self.about.mass)))
                assert self.a < 0
                self.energy = - G * self.about.mass / (2 * self.a)
                self.v_inf = np.sqrt(- G * self.about.mass / self.a)
                guess = 20 * 24 * 60**2 # start by simulating for 20 days to find the periapsis
                r_peri = float
                while True:
                    t = np.linspace(0, guess, 500000)
                    sol = odeint(model_2BP, np.array(initial_conds)*1e-3, t, args=(self.about.mass, ))
                    x, y, z, vx, vy, vz = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3], sol[:, 4], sol[:, 5]
                    r = magnitude(np.array([x, y, z]))
                    if r.argmin() == len(t) - 1:
                        guess += 20 * 24 * 60**2
                        continue
                    else:
                        r_peri = r.min() * 1e3
                        break
                self.e = - r_peri / self.a + 1
                self.h = - self.v_inf * self.a * np.sqrt(self.e**2 - 1)
            elif mode == "direct": # (a, e, primary)
                self.a, self.e, self.about = data
                assert self.a < 0
                self.energy = - G * (self.about.mass + self.secondary_mass) / (2 * self.a)
                self.v_inf = np.sqrt(- G * (self.about.mass + self.secondary_mass) / self.a)
                self.h = - self.v_inf * self.a * np.sqrt(self.e**2 - 1)
        elif source == "state vectors":
            self.about, self.position, self.velocity = data
            self.a, self.e, self.e_vector, self.h, self.h_vector, self.inclination, self.arg_periapsis, self.long_asc_node, self.ecc_anom, self.energy = cart_2_kep(self.position, self.velocity, G * self.about.mass)
            if self.e == 0:
                self.orbit = "Circular"
                self.T = period(self.a, self.about.mass)
            elif self.e < 1:
                self.orbit = "Elliptical"
                self.T = period(self.a, self.about.mass)
            elif self.e == 1:
                self.orbit = "Parabolic"
                self.T = np.inf
                self.a = np.inf
            elif self.e > 1:
                self.orbit = "Hyperbolic"
                self.T = np.inf
                self.v_inf = np.sqrt(- G * self.about.mass / self.a)
            else:
                self.orbit = "Error determining orbit type"
                self.T = "Error"
            self.energy = - G * self.about.mass * secondary_mass / (2 * self.a)
        else:
            raise NotImplementedError
        if source != "state vectors":
            self.arg_periapsis = arg_periapsis
            self.long_asc_node = long_asc_node
            self.inclination = inclination
        self.periapsis = self.__periapsis__()
        self.apoapsis = self.__apoapsis__()

    def __repr__(self):
        return f"{self.orbit} orbit with semi-major axis {round(self.a * 1e-3, 1)}km and eccentricity {round(self.e, 1)}"

    def __periapsis__(self):
        # Create Apsis object for the point in trajectory closest to orbited body
        if self.orbit == "Parabolic":
            r_peri = self.h**2 / (2 * G * (self.about.mass + self.secondary_mass))
            return Apsis({"distance":r_peri, "velocity":visviva(self.about.mass, r_peri, self.a)})
        elif self.orbit == "Hyperbolic":
            r_peri = - self.a * (self.e - 1)
            return Apsis({"distance":r_peri, "velocity":np.sqrt(G * (self.about.mass + self.secondary_mass) * (2 / r_peri - 1 / self.a))})
        else: 
            r_peri = self.a * (1 - self.e)
            return Apsis({"distance":r_peri, "velocity":visviva(self.about.mass, r_peri, self.a)})

    def __apoapsis__(self):
        # Create Apsis object for the point in trajectory furthest from orbited body
        if self.orbit == "Parabolic":
            return Apsis({"distance":self.a, "velocity":0})
        elif self.orbit == "Hyperbolic":
            # Hyperbolic trajectories do not have a finite apoapsis, the velocity then becomes the hyperbolic excess velocity
            return Apsis({"distance":np.inf, "velocity":self.v_inf})
        else:
            r_apo = self.a * (1 + self.e)
            return Apsis({"distance":r_apo, "velocity":visviva(self.about.mass, r_apo, self.a)})

    def velocity(self, *arg):
        if len(arg) == 0:
            return np.sqrt(G * self.about.mass / self.a)
        else:
            if len(arg) > 1: raise ValueError
            return visviva(self.about.mass, *arg, self.a)
        
    def flight_path_angle(self):
        # returns the angle between the RADIAL and velocity vectors
        return np.arccos(np.dot(self.position, self.velocity) / (magnitude(self.position) * magnitude(self.velocity)))
    
    def flight_path_angle_gamma(self):
        # returns the angle between the TANGENT and velocity vectors
        return np.arcsin(np.dot(self.position, self.velocity) / (magnitude(self.position) * magnitude(self.velocity)))

    def true_anomaly(self, *r):
        if self.orbit == "Circular": return "Undefined"
        if hasattr(self, "e_vector"):
            return np.arccos(np.dot(self.e_vector, self.position) / (magnitude(self.e_vector) * magnitude(self.position)))
        else:
            return np.arccos((self.a * (1 - self.e**2) - r[0]) / (self.e * r[0]))

    def inclination(self):
        assert hasattr(self, "h_vector")
        return np.arccos(self.h_vector[2] / magnitude(self.h_vector))
    
    def apply_state_vectors(self, line):
        self.position, self.velocity = np.array(self.position).T[line], np.array(self.velocity).T[line]
        return self
    
    def escape_SOI(self, excess=None):
        if self.orbit != "Circular": raise NotImplementedError
        if self.about.primary.name == "Sun": return self.velocity() * np.sqrt(2)
        return self.velocity() + hohmann(self.a, self.about.SOI, self.about.mass)

    def plot_potential(self, y_window=None, r_max=None):
        plt.close()
        plot_energy(self, y_window=y_window, r_max=r_max)

class Body:
    def __init__(self, name, mass, *radius):
        self.name = name
        self.mass = mass
        self.radius = radius[0] if len(radius) == 1 else 1e3

    def __repr__(self):
        return self.name

    def __str__(self):
        return f"Body - {self.name}: Mass-{round(self.mass, 1)}kg, Radius-{round(self.radius * 1e-3, 1)}km"

    def true_anomaly(self):
        assert hasattr(self, "orbit")
        return self.orbit.true_anomaly()

    def eccentric_anomaly(self):
        assert hasattr(self, "orbit")
        return np.arccos((1 - magnitude(self.orbit.position) / self.orbit.a) / self.orbit.e)
    
    def mean_anomaly(self):
        return self.eccentric_anomaly() - self.orbit.e * np.sin(self.eccentric_anomaly())
    
    def apply_state_vectors(self, *line):
        # forwards the position and velocity vectors from a specified row of simulated data updating the current state of a Body object
        if len(line) == 0:
            self.orbit = self.orbit.apply_state_vectors(-1)
        elif type(line[0]) is int:
            self.orbit = self.orbit.apply_state_vectors(line[0])
        return self
    
class System:
    def __init__(self, primary, orbiting):
        self.primary = primary
        self.satellites = {obj.name:obj for obj in orbiting} if type(orbiting) is list else {orbiting.name:orbiting}
        sat_mass = sum([body.mass for body in self.satellites.values()])
        assert self.primary.mass > sat_mass
        self.mass = self.primary.mass + sat_mass
        self.name = f"{self.primary.name}{''.join([sat for sat in self.satellites])}System"
    
    def __repr__(self):
        return self.name
    
    def __str__(self):
        return f"{self.name} with {len([sat for sat in self.satellites.items()])} satellite(s)"
    
    def define_planet(self, body, a, arg_periapsis=0, long_asc_node=0, inclination=0):
        # defines circular orbits only
        self.satellites[body].orbit = OrbitalProperties("circular", (self, a), secondary_mass=self.satellites[body].mass, arg_periapsis=arg_periapsis, long_asc_node=long_asc_node, inclination=inclination)
        self.satellites[body].SOI = r_soi(self.primary.mass, self.satellites[body].mass, self.satellites[body].orbit.a)
        return self

    def define_planet_vectors(self, body, a, e, inclination, arg_periapsis, long_asc_node, ecc_anom):
        self.satellites[body].orbit = OrbitalProperties("state vectors", (self, *kep_2_cart(a, e, inclination, arg_periapsis, long_asc_node, ecc_anom, G * self.mass)), secondary_mass=self.satellites[body].mass)
        self.satellites[body].SOI = r_soi(self.primary.mass, self.satellites[body].mass, self.satellites[body].orbit.a)
        return self

    def set_eccentric_anomaly(self, body, ecc_anom):
        params = self.satellites[body].orbit
        self.satellites[body].orbit = OrbitalProperties("state vectors", (self, params.position, params.velocity))
        self.satellites[body].SOI = r_soi(self.primary.mass, self.satellites[body].mass, params.a)
        return self
    
    def set_true_anomaly(self, body, true_anom):
        params = self.satellites[body].orbit
        self.satellites[body].orbit = OrbitalProperties("state vectors", (self, *kep_2_cart(params.a, params.e, params.inclination, params.arg_periapsis, params.long_asc_node, true_2_eccentric_anom(true_anom, params.e), G * self.mass)))
        self.satellites[body].SOI = r_soi(self.primary.mass, self.satellites[body].mass, params.a)
        return self

class Simulate:
    def __init__(self, system, from_simulation=False):
        self.system = system.system if from_simulation else system
        if from_simulation:
            prev_sim = system
            if hasattr(prev_sim, "primary"): self.primary = prev_sim.primary.apply_state_vectors()
            if hasattr(prev_sim, "secondary"): self.secondary = prev_sim.secondary.apply_state_vectors()
            if hasattr(prev_sim, "probe"): self.probe = prev_sim.probe.apply_state_vectors()
            
    def __repr__(self):
        return f"Simulation object with {'primary' if hasattr(self, 'primary') else None} {'secondary' if hasattr(self, 'secondary') else None} {'probe' if hasattr(self, 'probe') else None}"
        
    def make_primary(self, name, about, position, velocity):
        self.primary = self.system.satellites[name]
        self.primary.orbit = OrbitalProperties("state vectors", (about, position, velocity))
        return self
        
    def make_secondary(self, name, about, position, velocity):
        self.secondary = self.system.satellites[name]
        self.secondary.orbit = OrbitalProperties("state vectors", (about, position, velocity))
        return self
        
    def make_probe(self, body_obj, about, position, velocity):
        self.probe = body_obj
        self.probe.orbit = OrbitalProperties("state vectors", (about, position, velocity))
        return self
        
    def draw_state_from_system(self, primary, secondary):
        self.primary, self.secondary = self.system.satellites[primary], self.system.satellites[secondary]
        return self

    def simulate(self, to, timestep="s"):
        if timestep.lower() == "days": to = to * 24 * 60**2
        if timestep.lower() == "years": to = to * 365 * 24 * 60**2
        self.t = np.linspace(0, to, 750000)
        if hasattr(self, "probe"):
            probe_state = np.array([*self.probe.orbit.position.T, *self.probe.orbit.velocity.T])
            probe_sol = odeint(model_2BP, probe_state*1e-3, self.t, args=(self.system.mass, )) * 1e3
            self.probe.orbit.position = np.array([probe_sol[:, 0], probe_sol[:, 1], probe_sol[:, 2]])
            self.probe.orbit.velocity = np.array([probe_sol[:, 3], probe_sol[:, 4], probe_sol[:, 5]])
        if hasattr(self, "primary"):
            primary_state = np.array([*self.primary.orbit.position.T, *self.primary.orbit.velocity.T])
            primary_sol = odeint(model_2BP, primary_state*1e-3, self.t, args=(self.system.mass, )) * 1e3
            self.primary.orbit.position = np.array([primary_sol[:, 0], primary_sol[:, 1], primary_sol[:, 2]])
            self.primary.orbit.velocity = np.array([primary_sol[:, 3], primary_sol[:, 4], primary_sol[:, 5]])
            # Scan for an encounter with the primary
            if hasattr(self, "probe"):
                r = magnitude(self.probe.orbit.position - self.primary.orbit.position)
                if r.min() <= self.primary.SOI:
                    minima = argrelmin(np.where(r > self.primary.SOI, r - self.primary.SOI, r + self.primary.SOI))[0]
                    time_of_encounter = self.t[minima[0]] / 24 / 60**2
                    print(f"Found encounter with primary at i[{minima[0]}] ({time_of_encounter} days) at {r[minima[0]]}")
                else:
                    print(f"Closest approach with primary: {r.min() * 1e-6}Mm")
        if hasattr(self, "secondary"):
            secondary_state = np.array([*self.secondary.orbit.position.T, *self.secondary.orbit.velocity.T])
            secondary_sol = odeint(model_2BP, secondary_state*1e-3, self.t, args=(self.system.mass, )) * 1e3
            self.secondary.orbit.position = np.array([secondary_sol[:, 0], secondary_sol[:, 1], secondary_sol[:, 2]])
            self.secondary.orbit.velocity = np.array([secondary_sol[:, 3], secondary_sol[:, 4], secondary_sol[:, 5]])
            # Scan for an encounter with the secondary
            if hasattr(self, "probe"):
                r = magnitude(self.probe.orbit.position - self.secondary.orbit.position)
                if r.min() <= self.secondary.SOI:
                    minima = argrelmin(np.where(r < self.secondary.SOI, r + self.secondary.SOI, r))[0]
                    time_of_encounter = self.t[minima[0]] / 24 / 60**2
                    print(f"Found encounter with secondary at i[{minima[0]}] ({time_of_encounter} days) at {r[minima[0]]}")
                else:
                    print(f"Closest approach with secondary: {r.min() * 1e-6}Mm")
        return self
    
    def relative_approach_parameters(self, body1, body2):
        # When finished, should calculate the relative approach parameters of two bodies encountering each other upon intersecting SOI's
        pass
            
    def plot_simulation(self, scale="AU", timestep="days", x_window=None, y_window=None, z_window=None, encounter=None, markevery=None):
        assert len(self.primary.orbit.position != 3)
        plt.close()
        plot_multi_simulation(self, scale=scale, x_window=x_window, y_window=y_window, z_window=z_window, encounter=encounter, timestep=timestep, markevery=markevery)

def create_simple_system():
    Sun = Body("Sun", sun["mass"], sun["radius"])
    Earth = Body("Earth", earth["mass"], earth["radius"])
    Jupiter = Body("Jupiter", jupiter["mass"], jupiter["radius"])
    Moon = Body("Moon", moon["mass"], moon["radius"])
    EM = System(Earth, Moon).define_planet("Moon", moon["sma"])
    EM = Body(EM.name, EM.mass, 0)
    solsys = System(Sun, [EM, Jupiter])
    solsys = solsys.define_planet(EM.name, earth["sma"], arg_periapsis=earth["arg_peri"], long_asc_node=earth["long_asc"])
    solsys = solsys.define_planet(Jupiter.name, jupiter["sma"], arg_periapsis=jupiter["arg_peri"], long_asc_node=jupiter["long_asc"])
    return solsys

def create_advanced_system():
    Sun = Body("Sun", sun["mass"], sun["radius"])
    Earth = Body("Earth", earth["mass"], earth["radius"])
    Jupiter = Body("Jupiter", jupiter["mass"], jupiter["radius"])
    Moon = Body("Moon", moon["mass"], moon["radius"])
    EM = System(Earth, Moon).define_planet_vectors("Moon", moon["sma"], moon["ecc"], moon["inc"], moon["arg_peri"], moon["long_asc"], 0)
    EM = Body(EM.name, EM.mass, 0)
    solsys = System(Sun, [EM, Jupiter])
    solsys = solsys.define_planet_vectors(EM.name, earth["sma"], earth["ecc"], earth["inc"], earth["arg_peri"], earth["long_asc"], 0)
    solsys = solsys.define_planet_vectors(Jupiter.name, jupiter["sma"], jupiter["ecc"], jupiter["inc"], jupiter["arg_peri"], jupiter["long_asc"], 0)
    return solsys