# optimization.py
import numpy as np
from scipy.optimize import differential_evolution
from simulation import impedance_matrix, tension_Oc_vector, calculate_Voc, calculate_powers, calculate_impedances

def objective_function(params, antennes_coords, impedances_simulation):
    num_antennas = len(antennes_coords)
    impedances = []
    param_index = 0
    for antenna in antennes_coords:
        R = params[param_index]
        X = params[param_index + 1]
        impedances.append(complex(R, X))
        param_index += 2

    impedance_loads = impedance_matrix(num_antennas, antennes_coords, reglage='manuel', impedances_manuel=impedances)
    excitation_voltages = tension_Oc_vector(num_antennas, antennes_coords, reglage='auto')
    results = calculate_Voc(impedances_simulation, num_antennas, excitation_voltages, impedance_loads, antennes_coords)
    powers = calculate_powers(results)

    receiver_index = next((idx for idx, ant in enumerate(antennes_coords) if ant['type'] == 'receiver'), None)
    if receiver_index is None:
        raise ValueError("Aucune antenne de type 'receiver' n'a été trouvée.")

    P_L_receiver = powers['P_L'][receiver_index].real
    return -P_L_receiver

def optimize_impedances(antennes_coords):
    num_antennas = len(antennes_coords)
    impedances_simulation = calculate_impedances(num_antennas, antennes_coords)

    bounds = [(0.0, 1000.0), (-1000.0, 1000.0)] * num_antennas

    result = differential_evolution(
        objective_function,
        bounds,
        args=(antennes_coords, impedances_simulation),
        strategy='best1bin',
        maxiter=1000,
        popsize=15,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True
    )
    return result.x
"""

def optimize_impedances(antennes_coords):
    num_antennas = len(antennes_coords)
    impedances_simulation = calculate_impedances(num_antennas, antennes_coords)

    # Définir les bornes pour les impédances (R et X pour chaque antenne)
    bounds = [(0.0, 1000.0), (-1000.0, 1000.0)] * num_antennas

    # Exécuter l'algorithme d'évolution différentielle
    result = differential_evolution(
        objective_function,
        bounds,
        args=(antennes_coords, impedances_simulation),
        strategy='rand1bin',  # Stratégie pour explorer de nouvelles régions
        maxiter=5000,         # Plus d'itérations pour explorer davantage
        popsize=50,           # Taille de population plus importante pour une meilleure diversité
        tol=1e-6,             # Tolérance plus stricte pour un résultat plus précis
        mutation=(0.6, 1.5),  # Augmenter l'amplitude de la mutation
        recombination=0.9,    # Favoriser le mélange des solutions
        disp=True             # Afficher les résultats intermédiaires pour surveiller la convergence
    )

    # Retourner les impédances optimisées
    return result.x
"""