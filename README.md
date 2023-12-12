IFT712 - Projet technique d'apprentissage
==============================

Description :
-------------
Le projet consiste à classifier le jeu de données Kaggle  "[classification de feuilles](https://www.kaggle.com/c/leaf-classification)" en utilisant 6 algorithmes de classification différents.

### Algorithmes de classification :
Nous avons utilisé les méthodes de classifications suivantes pour notre projet : 
- Régression Logistique
- Perceptron
- Noyaux K-classe
- SVM
- Multi Layer Perceptron (MLP)
- CNN 


Organisation du projet
------------

    ├── LICENSE
    ├── README.md          <- Description du projet.
    │
    ├── data               <- Le dossier data n'est pas inclus dans le projet.
    │   │                     Il doit-être créé et le dossier images ainsi que
    │   │                     le dataset train.csv doivent-être téléchargés 
    │   │                     depuis le site Kaggle.
    │   │
    │   ├── images         <- Dossier images à télécharger depuis le site Kaggle.
    │   │                     Il contient toutes les images de feuilles à classifier.
    │   │
    │   ├── train.csv      <- Intermediate data that has been transformed.
    │
    │
    ├── models             <- Contient les modèles développé via PyTorch
    │
    ├── notebooks          <- Contient les différents notebooks. Chaque notebook
    │                         contient les expérimentations associéees à un
    │                         algorithme de classification.
    │
    ├── utils              <- Contient des scripts python contenant des fonctions
    │                         que nous réutilisons plusieurs fois comme des
    │                         fonctions de plot ou bien aussi une classe custom
    │                         de chargement de dataset pour pyTorch.    
    │
    │
    └── requirements.txt   <- Le fichier contenant tous les modules nécessaires 
                              pour reproduire l'environnement du projet.


Installation du projet :
----------------------

### 1. Configuration de l'environnement Python

<div style="text-align: justify">
    Le projet nécessite l'utilisation de Python <b>3.8.X</b> ou plus.
    Nous laissons libre choix à la manière dont est créé l'environnement Python.
    Une fois qu'un environnement Python 3.8+ est créé.
    Il faut se mettre à la racine du projet et entrer les commandes :
</div>

```
pip install --upgrade pip
```
```
pip install -r requirements.txt
```

### 2. Récupérer le jeu de données.

Si ce n'est pas déjà fait, il faut télécharger le jeu de données "[classification de feuilles](https://www.kaggle.com/c/leaf-classification)".

Pour ce projet, il faut télécharger les éléments suivants :
- images.zip
- train.csv.zip

À la racine du projet, il faut créer un dossier "<b>data</b>". À l'intérieur de ce dossier, nous ajoutons le dossier <b>images</b> obtenu en décompressant l'archive "<b>images.zip</b>" et le fichier "<b>train.csv</b>" obtenue en décompressant l'archive "<b>train.csv.zip</b>".

Cela devrait donner la structure suivante :

    Projet
    │
    ├── data              
    │   │
    │   ├── images   
    │   │
    │   ├── train.csv 
