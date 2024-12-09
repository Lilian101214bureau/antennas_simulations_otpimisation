# simulation.py
from PyNEC import nec_context
import numpy as np
import math
from constants import freq, lam, segment_count_impair, half_segment, position_half_segment, radius, half_length

def setup_geometry(context, antennes_coords):
    geo = context.get_geometry()
    for idx, antenna in enumerate(antennes_coords):
        xw1, yw1, zw1, xw2, yw2, zw2 = antenna['coords']
        geo.wire(
            tag_id=idx + 1,
            segment_count=segment_count_impair,
            xw1=xw1, yw1=yw1, zw1=zw1,
            xw2=xw2, yw2=yw2, zw2=zw2,
            rad=radius, rdel=1, rrad=1
        )
    context.geometry_complete(0)

def create_context(freq, antennes_coords):
    context = nec_context()
    setup_geometry(context, antennes_coords)
    context.fr_card(0, 1, freq / 1e6, 0)
    return context

def get_currents_per_segment(sc):
    currents_per_segment = []
    if hasattr(sc, 'get_n') and hasattr(sc, 'get_current'):
        n = sc.get_n()
        currents = sc.get_current()
        segment_numbers = sc.get_current_segment_number()
        tags = sc.get_current_segment_tag()
        for i in range(n):
            current = currents[i]
            amplitude = abs(current)
            phase = np.angle(current, deg=True)
            segment_number = segment_numbers[i]
            tag = tags[i]
            currents_per_segment.append({
                'segment_number': segment_number,
                'tag': tag,
                'amplitude': amplitude,
                'phase': phase
            })
    else:
        print("Les méthodes get_n() ou get_current() ne sont pas disponibles.")
    return currents_per_segment

def calculate_self_impedances(num_antennas, antennes_coords):
    self_impedances = np.zeros((num_antennas, num_antennas), dtype=complex)
    for i in range(num_antennas):
        context = create_context(freq, antennes_coords)
        context.ex_card(0, i + 1, half_segment, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for k in range(num_antennas):
            if k != i:
                context.ld_card(4, k + 1, 0, segment_count_impair - 1, 1e50, 1e50, 0.0)
        context.xq_card(0)
        sc = context.get_structure_currents(0)
        ipt = context.get_input_parameters(0)
        voltages = ipt.get_voltage()
        currents = sc.get_current()
        self_impedances[i, i] = voltages[0] / currents[i * segment_count_impair + position_half_segment]
    return self_impedances

def calculate_impedances(num_antennas, antennes_coords):
    self_imped = calculate_self_impedances(num_antennas, antennes_coords)
    mutual_impedances = np.zeros((num_antennas, num_antennas), dtype=complex)
    in_impedances = np.zeros((num_antennas, num_antennas), dtype=complex)
    for i in range(num_antennas):
        for j in range(num_antennas):
            if i != j:
                context = create_context(freq, antennes_coords)
                context.ex_card(0, i + 1, half_segment, 0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                for k in range(num_antennas):
                    if k != j and k != i:
                        context.ld_card(4, k + 1, 0, segment_count_impair - 1, 1e50, 1e50, 0.0)
                context.xq_card(0)
                sc = context.get_structure_currents(0)
                ipt = context.get_input_parameters(0)
                voltages = ipt.get_voltage()
                currents = sc.get_current()
                in_impedances[i, j] = voltages[0] / currents[i * segment_count_impair + position_half_segment]
                mutual_impedances[i, j] = (-(in_impedances[i, j] - self_imped[i, i])*(self_imped[j, j]))**(1/2)  
                
                
    return self_imped + mutual_impedances

def impedance_matrix(num_antennas, antennes_coords, reglage='auto', impedances_manuel=None):
    if reglage == 'manuel':
        if impedances_manuel is None or len(impedances_manuel) != num_antennas:
            raise ValueError("Pour 'manuel', fournissez une liste 'impedances_manuel' de longueur égale à 'num_antennas'.")
        impedances = impedances_manuel
    elif reglage == 'auto':
        impedances = [complex(73, 42.5)] * num_antennas
    else:
        raise ValueError("Le paramètre 'reglage' doit être 'auto' ou 'manuel'.")
    return np.diag(impedances)

def tension_Oc_vector(n_antennes, antennes_coords, reglage='auto', tensions_manuel=None):
    if reglage == 'manuel':
        if tensions_manuel is None or len(tensions_manuel) != n_antennes:
            raise ValueError("Pour 'manuel', fournissez une liste tensions_manuel de longueur égale à n_antennes.")
        tensions = tensions_manuel
    elif reglage == 'auto':
        tensions = []
        for antenna in antennes_coords:
            if antenna['type'] == 'emitter':
                tensions.append(complex(1, 0))
            else:
                tensions.append(complex(0, 0))
    else:
        raise ValueError("Le paramètre 'reglage' doit être 'auto' ou 'manuel'.")

    return np.array(tensions).reshape(-1, 1)

def calculate_Voc(impedance_matrix_data, num_antennas, excitation_voltages, impedance_loads, antennes_coords):
    context = create_context(freq, antennes_coords)
    # Appliquer excitations
    for i in range(num_antennas):
        V = excitation_voltages[i][0]
        if V != 0:
            context.ex_card(0, i+1, half_segment, half_segment, V.real, V.imag, 0,0,0,0)

    # Appliquer loads
    diag_load = np.diag(impedance_loads)
    for i in range(num_antennas):
        ZL = diag_load[i]
        if ZL != 0:
            R = ZL.real
            X = ZL.imag
            L = X/(2*math.pi*freq)
            context.ld_card(0, i+1, half_segment, half_segment, R, L, 0.0)

    context.xq_card(0)
    sc = context.get_structure_currents(0)
    ipt = context.get_input_parameters(0)
    currents = sc.get_current()
    # voltages d'entrée:
    voltages_in = ipt.get_voltage()
    
    currents_at_center = np.zeros((num_antennas,1), dtype=complex)
    
    
    
    for i in range(num_antennas):
        idx_center = i*segment_count_impair+ position_half_segment
        currents_at_center[i,0] = currents[idx_center]
    
    
    
    zin = np.zeros(num_antennas, dtype=complex)

    for i in range(num_antennas):
        # Création de ZL_mod, une matrice diagonale avec ZL sauf à la position (i,i) où c'est 0
        ZL_mod = np.zeros((num_antennas, num_antennas), dtype=complex)
        for j in range(num_antennas):
            if j != i:
                ZL_mod[j, j] = impedance_loads[j, j]
            # Sinon, ZL_mod[i, i] reste à 0

        # Calcul de Zmod
        Zmod = impedance_matrix_data + ZL_mod

        # Calcul de l'inverse de Zmod
        try:
            Zmod_inv = np.linalg.inv(Zmod)
        except np.linalg.LinAlgError:
            
            Zmod_inv = np.zeros((num_antennas, num_antennas), dtype=complex)

        # Calcul de zin[i] en prenant l'inverse du scalaire
        zin_scalar = Zmod_inv[i, i]
        if zin_scalar != 0:
            zin[i] = 1 / zin_scalar
        else:
            zin[i] = np.inf  # Ou une valeur appropriée si le scalaire est nul

    # Calcul de Vin = zin * I
    Vin = zin.reshape(-1, 1) * currents_at_center
    Vl = diag_load[:,None]*currents_at_center
    vin_Oc = excitation_voltages - Vin - Vl
    V = excitation_voltages - Vl
    Vbis = vin_Oc + Vin

    results = {
        'Voc': excitation_voltages,
        'Vl': Vl,
        'Vin': Vin,
        'vin_Oc': vin_Oc,
        'V': V,
        'vbis': Vbis,
        'currents_at_center': currents_at_center,
        'zin': zin,
        'impedance_loads': diag_load,
        'impedance_matrix': impedance_matrix_data
    }
    return results

def calculate_powers(results):
    Voc = results['Voc']
    Vl = results['Vl']
    vin_Oc = results['vin_Oc']
    Vin = results['Vin']
    currents = results['currents_at_center']

    val_P_V_oc = 1000 * (0.5 * np.real(Voc * np.conj(currents)))
    val_P_Vin_oc = 1000 * (0.5 * np.real(vin_Oc * -np.conj(currents)))
    val_P_L = 1000 * (0.5 * np.real(Vl * np.conj(currents)))
    val_P_in = 1000 * (0.5 * np.real(Vin * np.conj(currents)))
    error_logs = []
    def safe_log10(values, name):
    
        problem_indices = np.where(values <= 0)[0]
        if problem_indices.size > 0:
            error_logs.append({
                "name": name,
                "indices": problem_indices.tolist(),
                "values": values[problem_indices].tolist()
            })
        safe_values = values.copy()
        safe_values[safe_values <= 0] = np.nan
        return np.log10(safe_values)

    P_V_oc = safe_log10(val_P_V_oc, "P_V_oc")
    P_Vin_oc = safe_log10(val_P_Vin_oc, "P_Vin_oc")
    P_L = safe_log10(val_P_L, "P_L")
    P_in = safe_log10(val_P_in, "P_in")

    powers = {
        'P_V_oc': P_V_oc,
        'P_Vin_oc': P_Vin_oc,
        'P_L': P_L,
        'P_in': P_in,
    }
    return powers

def afficher_resultats(resultats, format_polaire=False):
    print("\n--- Résultats des calculs ---\n")
    def afficher_element(element, indent=4):
        espace=" "*indent
        if np.iscomplexobj(element):
            if format_polaire:
                module=np.abs(element)
                phase=np.angle(element,deg=True)
                print(f"{espace}Module = {module:.6f}, Phase = {phase:.2f}°")
            else:
                print(f"{espace}{element.real:.6f} + {element.imag:.6f}j")
        elif isinstance(element,(int,float)):
            print(f"{espace}{element:.6f}")
        else:
            print(f"{espace}{element}")

    def afficher_structure(valeur,indent=4):
        espace=" "*indent
        if isinstance(valeur,np.ndarray):
            if valeur.ndim==1:
                for i,element in enumerate(valeur):
                    print(f"{espace}[{i}] :",end="")
                    afficher_element(element,indent=0)
            elif valeur.ndim==2:
                for i,ligne in enumerate(valeur):
                    print(f"{espace}Ligne {i}:")
                    for j,element in enumerate(ligne):
                        print(f"{espace}  [{i},{j}] :",end="")
                        afficher_element(element,indent=0)
            else:
                for i,sous_tableau in enumerate(valeur):
                    print(f"{espace}Sous-tableau {i}:")
                    afficher_structure(sous_tableau,indent+4)
        elif isinstance(valeur,(list,tuple)):
            for i,element in enumerate(valeur):
                print(f"{espace}[{i}] :")
                afficher_structure(element,indent+4)
        elif isinstance(valeur,dict):
            for sous_cle,sous_valeur in valeur.items():
                print(f"{espace}{sous_cle} :")
                afficher_structure(sous_valeur,indent+4)
        else:
            afficher_element(valeur,indent=indent)

    for cle,valeur in resultats.items():
        print(f"{cle} :")
        afficher_structure(valeur,indent=4)
        print()
# simulation.py (extrait, à insérer après les imports et les définitions de constantes)

def calculate_geometric_impedance(L1, L2, r21, theta, theta_prime, lam):
    rA = r21 / lam
    magnitude_factor = (60 * math.pi * L1 * L2) / rA
    orientation_factor = math.sin(theta) * math.sin(theta_prime)
    periodic_factor = math.sin(2 * math.pi * rA) + 1j * math.cos(2 * math.pi * rA)
    Z21 = magnitude_factor * orientation_factor * periodic_factor
    return Z21

def calculate_impedances_geometry_only(num_antennas, antennes_coords, lam):
    """
    Calcule les impédances purement géométriques entre dipôles,
    sans utiliser NEC, selon la formule approchée fournie.
    """
    impedances_geo = np.zeros((num_antennas, num_antennas), dtype=complex)
    length_Li = np.zeros(num_antennas)
    orientations = []

    # Calcul de la longueur et de l'orientation de chaque dipôle
    for i in range(num_antennas):
        xw1, yw1, zw1 = antennes_coords[i]['coords'][0:3]
        xw2, yw2, zw2 = antennes_coords[i]['coords'][3:6]
        orientation_vector = np.array([xw2 - xw1, yw2 - yw1, zw2 - zw1])
        orientations.append(orientation_vector)
        length_Li[i] = np.linalg.norm(orientation_vector)
        print(f"Longueur du dipôle {i+1} : {length_Li[i]:.6f} m")

    # Calcul des impedances géométriques mutuelles
    for i in range(num_antennas):
        for j in range(num_antennas):
            if i == j:
                # Self-impedance géométrique non calculée ici (ou à définir)
                # Vous pouvez laisser à 0 ou une valeur approchée
                # En général, cette partie était supposée être remplacée par le calcul NEC
                # donc laissez impedances_geo[i,j] = 0 si vous n'avez pas de formule.
                pass
            else:
                # Centre du dipôle i
                x1 = (antennes_coords[i]['coords'][0] + antennes_coords[i]['coords'][3]) / 2
                y1 = (antennes_coords[i]['coords'][1] + antennes_coords[i]['coords'][4]) / 2
                z1 = (antennes_coords[i]['coords'][2] + antennes_coords[i]['coords'][5]) / 2
                # Centre du dipôle j
                x2 = (antennes_coords[j]['coords'][0] + antennes_coords[j]['coords'][3]) / 2
                y2 = (antennes_coords[j]['coords'][1] + antennes_coords[j]['coords'][4]) / 2
                z2 = (antennes_coords[j]['coords'][2] + antennes_coords[j]['coords'][5]) / 2

                r_vector = np.array([x2 - x1, y2 - y1, z2 - z1])
                r21 = np.linalg.norm(r_vector)
                L1 = length_Li[i]
                L2 = length_Li[j]

                # Calcul des angles
                u_i = orientations[i] / np.linalg.norm(orientations[i])
                u_j = orientations[j] / np.linalg.norm(orientations[j])
                r_unit = r_vector / r21
                cos_theta = np.dot(u_i, r_unit)
                cos_theta_prime = np.dot(u_j, r_unit)
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                cos_theta_prime = np.clip(cos_theta_prime, -1.0, 1.0)
                theta = math.acos(cos_theta)
                theta_prime = math.acos(cos_theta_prime)

                Z_geo = calculate_geometric_impedance(L1, L2, r21, theta, theta_prime, lam)
                impedances_geo[i, j] = Z_geo
    return impedances_geo

def test_calculate_impedances_geometry_only(num_antennas, antennes_coords, lam):
    expected_impedances = calculate_impedances_geometry_only(num_antennas, antennes_coords, lam)
    print("\nTest des impédances géométriques calculées :")
    for i in range(num_antennas):
        for j in range(num_antennas):
            print(f"Z[{i+1},{j+1}] = {expected_impedances[i, j].real:.2f} + j{expected_impedances[i, j].imag:.2f} Ohm")
