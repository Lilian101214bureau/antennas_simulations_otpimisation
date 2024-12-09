# main.py
from geometry import generate_manual_antenna_coords, generate_random_antenna_coords_3D_sphere
from simulation import (calculate_impedances, tension_Oc_vector, impedance_matrix, 
                        calculate_Voc, afficher_resultats, calculate_powers)
from plotting import plot_antennas, plot_antennas_sphere_3D, visualize_powers
from optimization import optimize_impedances
from constants import lam, half_length, sphere_radius
from scenarios import simulate_and_plot_square_network
import numpy as np

def main():
    print("Début de l'exécution du programme principal...")
    """
    ### Scénario manuel ###
    manual_defs = [
        (2 * lam, 0, -half_length, 2 * lam, 0, half_length, 'emitter'),
        (lam * 100 , 0, -half_length, lam *100, 0, half_length, 'receiver'),
        #(lam, 0, -half_length, lam, 0, half_length, 'reflector')
    ]
    antennes_coords_manual = generate_manual_antenna_coords(manual_defs)

    # Affichage des antennes (scénario manuel)
    plot_antennas(antennes_coords_manual, lam)

    # Simulation initiale (manuel)
    num_antennas_manual = len(antennes_coords_manual)
    impedances_simulation_manual = calculate_impedances(num_antennas_manual, antennes_coords_manual)
    excitation_voltages_manual = tension_Oc_vector(num_antennas_manual, antennes_coords_manual, reglage='auto')
    impedance_loads_manual = impedance_matrix(num_antennas_manual, antennes_coords_manual, reglage='auto')

    results_initial_manual = calculate_Voc(impedances_simulation_manual, num_antennas_manual, 
                                           excitation_voltages_manual, impedance_loads_manual, antennes_coords_manual)
    afficher_resultats(results_initial_manual, format_polaire=False)

    # Optimisation (manuel)
    optimized_manual = optimize_impedances(antennes_coords_manual)
    optimized_impedances_manual = []
    for i in range(num_antennas_manual):
        R = optimized_manual[2*i]
        X = optimized_manual[2*i+1]
        optimized_impedances_manual.append(complex(R,X))

    final_impedance_loads_manual = impedance_matrix(num_antennas_manual, antennes_coords_manual, reglage='manuel', 
                                                    impedances_manuel=optimized_impedances_manual)
    results_optimized_manual = calculate_Voc(impedances_simulation_manual, num_antennas_manual, 
                                             excitation_voltages_manual, final_impedance_loads_manual, antennes_coords_manual)
    afficher_resultats(results_optimized_manual, format_polaire=False)

    # Puissances (manuel)
    powers_initial_manual = calculate_powers(results_initial_manual)
    powers_optimized_manual = calculate_powers(results_optimized_manual)

    # Données avant optimisation (manuel)
    P_V_oc_init_m = powers_initial_manual['P_V_oc'].flatten()
    P_Vin_oc_init_m = powers_initial_manual['P_Vin_oc'].flatten()
    P_in_init_m = powers_initial_manual['P_in'].flatten()
    P_L_init_m = powers_initial_manual['P_L'].flatten()
    dipole_numbers_manual = np.arange(1, num_antennas_manual + 1)
    mean_PL_init_m = np.nanmean(P_L_init_m)
    power_data_init_m = [P_V_oc_init_m, P_Vin_oc_init_m, P_in_init_m, P_L_init_m]
    labels = ["$P_{V_{oc}}$", "$P_{V_{in_{oc}}}$", "$P_{in}$", "$P_{L}$"]
    # Visualisation avant optimisation (manuel)
    visualize_powers(power_data_init_m, labels, "Puissances par dipôle avant optimisation (manuel)", mean_PL_init_m, dipole_numbers_manual)

    # Données après optimisation (manuel)
    P_V_oc_final_m = powers_optimized_manual['P_V_oc'].flatten()
    P_Vin_oc_final_m = powers_optimized_manual['P_Vin_oc'].flatten()
    P_in_final_m = powers_optimized_manual['P_in'].flatten()
    P_L_final_m = powers_optimized_manual['P_L'].flatten()
    mean_PL_final_m = np.nanmean(P_L_final_m)
    power_data_final_m = [P_V_oc_final_m, P_Vin_oc_final_m, P_in_final_m, P_L_final_m]
    # Visualisation après optimisation (manuel)
    visualize_powers(power_data_final_m, labels, "Puissances par dipôle après optimisation (manuel)", mean_PL_final_m, dipole_numbers_manual)

    """
    labels = ["$P_{V_{oc}}$", "$P_{V_{in_{oc}}}$", "$P_{in}$", "$P_{L}$"]
    ### Scénario sphère ###
    num_antennas_sphere = 10 # même nombre d'antennes
    antennes_coords_sphere = generate_random_antenna_coords_3D_sphere(num_antennas_sphere, lam, sphere_radius)


    # Affichage des antennes dans la sphère
    plot_antennas_sphere_3D(antennes_coords_sphere, sphere_radius, lam)

    # Simulation initiale (sphère)
    impedances_simulation_sphere = calculate_impedances(num_antennas_sphere, antennes_coords_sphere)
    excitation_voltages_sphere = tension_Oc_vector(num_antennas_sphere, antennes_coords_sphere, reglage='auto')
    impedance_loads_sphere = impedance_matrix(num_antennas_sphere, antennes_coords_sphere, reglage='auto')

    results_initial_sphere = calculate_Voc(impedances_simulation_sphere, num_antennas_sphere, 
                                           excitation_voltages_sphere, impedance_loads_sphere, antennes_coords_sphere)
    afficher_resultats(results_initial_sphere, format_polaire=False)

    # Optimisation (sphère)
    optimized_sphere = optimize_impedances(antennes_coords_sphere)
    optimized_impedances_sphere = []
    for i in range(num_antennas_sphere):
        R = optimized_sphere[2*i]
        X = optimized_sphere[2*i+1]
        optimized_impedances_sphere.append(complex(R,X))

    final_impedance_loads_sphere = impedance_matrix(num_antennas_sphere, antennes_coords_sphere, reglage='manuel', 
                                                    impedances_manuel=optimized_impedances_sphere)
    results_optimized_sphere = calculate_Voc(impedances_simulation_sphere, num_antennas_sphere, 
                                             excitation_voltages_sphere, final_impedance_loads_sphere, antennes_coords_sphere)
    afficher_resultats(results_optimized_sphere, format_polaire=False)

    # Puissances (sphère)
    powers_initial_sphere = calculate_powers(results_initial_sphere)
    powers_optimized_sphere = calculate_powers(results_optimized_sphere)

    # Données avant optimisation (sphère)
    P_V_oc_init_s = powers_initial_sphere['P_V_oc'].flatten()
    P_Vin_oc_init_s = powers_initial_sphere['P_Vin_oc'].flatten()
    P_in_init_s = powers_initial_sphere['P_in'].flatten()
    P_L_init_s = powers_initial_sphere['P_L'].flatten()
    dipole_numbers_sphere = np.arange(1, num_antennas_sphere + 1)
    mean_PL_init_s = np.nanmean(P_L_init_s)
    power_data_init_s = [P_V_oc_init_s, P_Vin_oc_init_s, P_in_init_s, P_L_init_s]
    # Visualisation avant optimisation (sphère)
    visualize_powers(power_data_init_s, labels, "Puissances par dipôle avant optimisation (sphère)", mean_PL_init_s, dipole_numbers_sphere)

    # Données après optimisation (sphère)
    P_V_oc_final_s = powers_optimized_sphere['P_V_oc'].flatten()
    P_Vin_oc_final_s = powers_optimized_sphere['P_Vin_oc'].flatten()
    P_in_final_s = powers_optimized_sphere['P_in'].flatten()
    P_L_final_s = powers_optimized_sphere['P_L'].flatten()
    mean_PL_final_s = np.nanmean(P_L_final_s)
    power_data_final_s = [P_V_oc_final_s, P_Vin_oc_final_s, P_in_final_s, P_L_final_s]
    # Visualisation après optimisation (sphère)
    visualize_powers(power_data_final_s, labels, "Puissances par dipôle après optimisation (sphère)", mean_PL_final_s, dipole_numbers_sphere)










    simulate_and_plot_square_network()
    print("Programme terminé.")
  
if __name__ == "__main__":
    main()
