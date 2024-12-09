# plotting.py 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from constants import lam

def plot_antennas_sphere_3D(antennes_coords, sphere_radius, lam=lam):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    antennes_coords_lambda = []
    for antenna in antennes_coords:
        xw1, yw1, zw1, xw2, yw2, zw2 = antenna['coords']
        antennes_coords_lambda.append({
            'coords': (
                xw1/lam, yw1/lam, zw1/lam,
                xw2/lam, yw2/lam, zw2/lam
            ),
            'type': antenna['type']
        })

    sphere_radius_lambda = sphere_radius / lam

    all_coords = [coord for antenna in antennes_coords_lambda for coord in antenna['coords']]
    all_coords = np.array(all_coords).reshape(-1, 3)
    max_coord = np.max(all_coords, axis=0)
    min_coord = np.min(all_coords, axis=0)
    max_limit = max(np.max(max_coord), np.abs(np.min(min_coord))) * 1.1

    ax.set_xlim([-max_limit, max_limit])
    ax.set_ylim([-max_limit, max_limit])
    ax.set_zlim([-max_limit, max_limit])
    ax.set_box_aspect([1, 1, 1])

    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = sphere_radius_lambda * np.outer(np.cos(u), np.sin(v))
    y = sphere_radius_lambda * np.outer(np.sin(u), np.sin(v))
    z = sphere_radius_lambda * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.3)

    colors = {'emitter': 'red', 'receiver': 'green', 'reflector': 'blue'}
    labels = {'emitter': 'Émetteur', 'receiver': 'Récepteur', 'reflector': 'Réflecteur'}

    plotted_labels = set()
    for antenna in antennes_coords_lambda:
        xw1, yw1, zw1, xw2, yw2, zw2 = antenna['coords']
        antenna_type = antenna['type']
        lbl = labels[antenna_type]
        if lbl not in plotted_labels:
            ax.plot([xw1, xw2], [yw1, yw2], [zw1, zw2],
                    color=colors[antenna_type], marker='o', label=lbl)
            plotted_labels.add(lbl)
        else:
            ax.plot([xw1, xw2], [yw1, yw2], [zw1, zw2],
                    color=colors[antenna_type], marker='o')

    ax.set_xlabel('X (λ)')
    ax.set_ylabel('Y (λ)')
    ax.set_zlabel('Z (λ)')
    plt.title("Antennes générées dans une sphère (recentrées, en λ)")
    ax.legend()
    plt.show()

def visualize_powers(power_data, labels, title, mean_power, dipole_numbers, filename=None):
    x = dipole_numbers
    bar_width = 0.2
    offset = np.arange(len(power_data)) * bar_width

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, data in enumerate(power_data):
        ax.bar(x + offset[i], data, bar_width, label=labels[i])

    ax.axhline(y=mean_power, color='green', linestyle='--', label=f"mean $P_L = {mean_power:.1f} dbm$")
    ax.set_xlabel("Dipole number")
    ax.set_ylabel("Power (dbm)")
    ax.set_title(title)
    ax.set_xticks(x + bar_width * (len(power_data) - 1) / 2)
    ax.set_xticklabels(dipole_numbers)
    ax.legend()
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

def plot_antennas(antennes_coords, lam=lam):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    antennes_coords_lambda = []
    for antenna in antennes_coords:
        xw1, yw1, zw1, xw2, yw2, zw2 = antenna['coords']
        antennes_coords_lambda.append({
            'coords': (
                xw1 / lam, yw1 / lam, zw1 / lam,
                xw2 / lam, yw2 / lam, zw2 / lam
            ),
            'type': antenna['type']
        })

    all_coords = [coord for antenna in antennes_coords_lambda for coord in antenna['coords']]
    all_coords = np.array(all_coords).reshape(-1, 3)
    max_coord = np.max(all_coords, axis=0)
    min_coord = np.min(all_coords, axis=0)
    max_limit = max(np.max(max_coord), np.abs(np.min(min_coord))) * 1.1

    ax.set_xlim([-max_limit, max_limit])
    ax.set_ylim([-max_limit, max_limit])
    ax.set_zlim([-max_limit, max_limit])
    ax.set_box_aspect([1, 1, 1])

    colors = {'emitter': 'red', 'receiver': 'green', 'reflector': 'blue'}
    labels = {'emitter': 'Émetteur', 'receiver': 'Récepteur', 'reflector': 'Réflecteur'}

    plotted_labels = set()
    for antenna in antennes_coords_lambda:
        xw1, yw1, zw1, xw2, yw2, zw2 = antenna['coords']
        antenna_type = antenna['type']
        lbl = labels[antenna_type]
        if lbl not in plotted_labels:
            ax.plot([xw1, xw2], [yw1, yw2], [zw1, zw2],
                    color=colors[antenna_type], marker='o', label=lbl)
            plotted_labels.add(lbl)
        else:
            ax.plot([xw1, xw2], [yw1, yw2], [zw1, zw2],
                    color=colors[antenna_type], marker='o')

    ax.set_xlabel('X (λ)')
    ax.set_ylabel('Y (λ)')
    ax.set_zlabel('Z (λ)')
    plt.title("Antennes générées (recentrées, en λ)")
    ax.legend()
    plt.show()