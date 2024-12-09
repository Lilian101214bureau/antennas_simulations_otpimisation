# Projet de Simulation d’Antennes avec PyNEC

Ce projet permet de simuler des réseaux d’antennes, de calculer leurs impédances, tensions, courants et puissances, puis d’optimiser les impédances de charge afin de maximiser la puissance reçue sur une antenne réceptrice.

## Prérequis

- Python 3.10.11 installé
- Bibliothèques Python suivantes installées :
  - pynec==1.7.3.4
  - numpy
  - matplotlib
  - scipy
  - pygad

Pour installer ces dépendances, vous pouvez utiliser :
```bash
pip install pynec==1.7.3.4 numpy matplotlib scipy pygad

Ref https://theses.hal.science/tel-04550099 majoritairement chap 3 