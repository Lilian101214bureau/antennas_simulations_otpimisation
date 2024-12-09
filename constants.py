# constants.py
freq = 868e6  # Fréquence en Hz
c = 299792458  # Vitesse de la lumière en m/s
lam = c / freq  # Longueur d'onde en mètres

segment_count_impair = 5
half_segment = int((segment_count_impair - 1) / 2) + 1
position_half_segment = int((segment_count_impair - 1) / 2)
radius = 1e-6  # Rayon des fils
half_length = 0.25 * lam
sphere_radius = 2*lam
