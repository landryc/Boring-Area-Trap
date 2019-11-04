# Boring-Area-Trap



Boring Area Trap on MultiArmed Bandit problem.
L'objectif de ce git est d'expérimenter le problème du Boring Area Trap dans les algorithmes d'apprentissage par renforcement, notamment Q-learning. 

L'environnement sur lequel nous faisons nos expérimentations est une version du problème du Bandit à plusieurs bras.
Dans cette version, le bandit n'a que deux bras et la différence de variance entre ces bras est importante. La solution que nous expérimentons est **ASRN (Adaptive Symetric Reward Noising)**.

En outre, nous montrons que l'utilisation de **l'exploration** ou d'un **learning rate faible** peuvent être vus comme des solutions à notre problème.

La seule chose que vous avez à faire pour voir nos résultats est d'exécuter la commande: 
 ``` python3 run.py ``` ou parcourir les dossiers images ci-dessus qui contiennent les résultats de certaines éxécutions avec des épisodes de longueurs différentes.
