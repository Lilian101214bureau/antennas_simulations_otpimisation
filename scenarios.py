# scenarios.py
import numpy as np
import matplotlib.pyplot as plt
from constants import freq, c, lam, half_length
from simulation import calculate_impedances, tension_Oc_vector, impedance_matrix, calculate_Voc, calculate_powers

def simulate_and_plot_square_network():
    R = 20.0
    d_values = [(2/3)*lam, 1*lam, (3/2)*lam, 2*lam, (5/2)*lam, 3*lam]
    N_phi = 360
    phi_values = np.linspace(0, 2*np.pi, N_phi)

    def create_antennas(d, phi):
        Cx, Cy = R, 0.0
        base_positions = [
            (Cx + d/2, Cy + d/2),
            (Cx - d/2, Cy + d/2),
            (Cx - d/2, Cy - d/2),
            (Cx + d/2, Cy - d/2),
        ]
        rotated_positions = []
        for (x,y) in base_positions:
            xr = Cx + (x - Cx)*np.cos(phi) - (y - Cy)*np.sin(phi)
            yr = Cy + (x - Cx)*np.sin(phi) + (y - Cy)*np.cos(phi)
            rotated_positions.append((xr, yr))
        emitter_coords = (0, 0, -half_length, 0, 0, half_length)
        antennes_coords_local = [{'coords': emitter_coords, 'type': 'emitter'}]
        antenne_types = ['receiver', 'reflector', 'reflector', 'reflector']
        for i, (x, y) in enumerate(rotated_positions):
            coords_tag = (x, y, -half_length, x, y, half_length)
            antennes_coords_local.append({'coords': coords_tag, 'type': antenne_types[i]})
        return antennes_coords_local

    def get_received_power(antennes_coords_local):
        num_antennas = len(antennes_coords_local)
        impedances_simulation_local = calculate_impedances(num_antennas, antennes_coords_local)
        excitation_voltages = tension_Oc_vector(num_antennas, antennes_coords_local, reglage='auto')
        impedance_loads = impedance_matrix(num_antennas, antennes_coords_local, reglage='auto')
        results = calculate_Voc(impedances_simulation_local, num_antennas, excitation_voltages, impedance_loads, antennes_coords_local)
        powers = calculate_powers(results)
        # L'antenne receiver est l'antenne 2 (index=1)
        P_L_receiver = powers['P_L'][1].real
        return P_L_receiver

    def create_reference_antennas(phi, d):
        Cx, Cy = R, 0.0
        x = Cx + d/2
        y = Cy + d/2
        xr = Cx + (x - Cx)*np.cos(phi) - (y - Cy)*np.sin(phi)
        yr = Cy + (x - Cx)*np.sin(phi) + (y - Cy)*np.cos(phi)
        emitter_coords = (0, 0, -half_length, 0, 0, half_length)
        coords_tag0 = (xr, yr, -half_length, xr, yr, half_length)
        antennes_ref = [{'coords': emitter_coords, 'type': 'emitter'},
                        {'coords': coords_tag0, 'type': 'receiver'}]
        return antennes_ref

    plt.figure(figsize=(10,6))
    for d in d_values:
        P_received_array=[]
        for phi in phi_values:
            antennes_loc = create_antennas(d, phi)
            P_L_receiver = get_received_power(antennes_loc)
            P_received_array.append(P_L_receiver)
        plt.plot(phi_values, P_received_array, label=f"d={d/lam:.2f} λ")

    d_ref=lam
    P_received_ref=[]
    for phi in phi_values:
        antennes_ref = create_reference_antennas(phi,d_ref)
        P_L_ref = get_received_power(antennes_ref)
        P_received_ref.append(P_L_ref)
    plt.plot(phi_values,P_received_ref,'r--',label="référence (Tag0 seul)")

    plt.xlabel("Angle φ (radians)")
    plt.ylabel("Puissance reçue par Tag0 (dBm)")
    plt.title("Puissance reçue par Tag0 en fonction de φ pour différentes valeurs de d")
    plt.grid(True)
    plt.legend()
    plt.show()
