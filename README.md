# Classification de Formes de Feuilles avec PyTorch

## Description
Ce projet implémente une **classification binaire** pour prédire l'architecture d'une feuille (Classe `0` ou `1`) à partir de ses mesures géométriques.

Il compare deux approches d'apprentissage supervisé avec **PyTorch** :
1.  **Régression Logistique** (Modèle linéaire simple).
2.  **Réseau de Neurones (MLP)** (Modèle profond avec couche cachée).

Le but est de démontrer l'impact du prétraitement des données (notamment le *Shuffling*) et de comparer la performance de modèles de complexité différente.

## Fonctionnalités
* **Chargement & Nettoyage** : Utilisation de `Pandas` pour traiter le dataset `leafshape17.csv`.
* **Shuffling** : Mise en évidence de l'importance du mélange aléatoire des données (passage de 45% à 100% de précision).
* **Entraînement** : Boucles d'apprentissage manuelles avec calcul de la *Binary Cross Entropy (BCELoss)* et optimiseur *Adam*.
* **Visualisation** : Comparaison des courbes de perte (*Loss*) avec `Matplotlib`.
* **Sauvegarde** : Export du meilleur modèle entraîné au format `.pth`.

## Installation
Assurez-vous d'avoir Python installé avec les bibliothèques suivantes :

```bash
pip install torch pandas matplotlib
