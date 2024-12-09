# geometry.py
import numpy as np
from constants import lam, half_length, sphere_radius

# geometry.py
import numpy as np
from constants import lam, half_length, sphere_radius

def generate_random_antenna_coords_3D_sphere(num_antennas, lam, sphere_radius):
    """
    Génère des coordonnées pour des antennes dans une sphère.
    1 émetteur, 1 récepteur, le reste en réflecteurs.
    """
    antennes_coords_local = []
    antenna_types = ['emitter', 'receiver'] + ['reflector'] * (num_antennas - 2)

    emitter_coords = (3*lam, 0, -half_length, 3*lam, 0, half_length)
    receiver_coords = (0, 0, -half_length, 0, 0, half_length)

    for antenna_type in antenna_types:
        if antenna_type == 'emitter':
            xw1, yw1, zw1, xw2, yw2, zw2 = emitter_coords
        elif antenna_type == 'receiver':
            xw1, yw1, zw1, xw2, yw2, zw2 = receiver_coords
        else:
            phi = np.random.uniform(0, 2 * np.pi)
            cos_theta = np.random.uniform(-1, 1)
            sin_theta = np.sqrt(1 - cos_theta**2)
            direction = np.array([sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta])

            u = np.random.uniform(0, 1)
            r = u ** (1/3) * sphere_radius
            center = r * direction

            phi_o = np.random.uniform(0, 2 * np.pi)
            cos_theta_o = np.random.uniform(-1, 1)
            sin_theta_o = np.sqrt(1 - cos_theta_o**2)
            orientation = np.array([sin_theta_o * np.cos(phi_o), sin_theta_o * np.sin(phi_o), cos_theta_o])

            xw1, yw1, zw1 = center - half_length * orientation
            xw2, yw2, zw2 = center + half_length * orientation

        antennes_coords_local.append({
            'coords': (xw1, yw1, zw1, xw2, yw2, zw2),
            'type': antenna_type
        })

    return antennes_coords_local


def generate_manual_antenna_coords(antennas_definition):
    """
    antennas_definition: liste de tuples (xw1, yw1, zw1, xw2, yw2, zw2, type)
    """
    antennes_coords_local = []
    for ant in antennas_definition:
        xw1, yw1, zw1, xw2, yw2, zw2, ant_type = ant
        antennes_coords_local.append({
            'coords': (xw1, yw1, zw1, xw2, yw2, zw2),
            'type': ant_type
        })
    return antennes_coords_local
