# Boring-Area-Trap



Boring Area Trap on MultiArmed Bandit problem.

L'obectif de ce git est de :
<ol>
 <li> Mettre en évidence le problème du Boring Area Trap;</li>
 <li> Présenter la solution ASRN (Adaptive Symetric Reward Noising) de Refael Vivanti et al.;</li>
 <li> Montrer les limites de cette solution, notamment pour des environnements à longue durée de jeu et à zones ennuyeuses multiples</li>
 <li> Montrer enfin que les algorithmes RASRN (Rebooted Adaptive Symetric Reward Noising) que nous proposons permettent de dépasser ces limites. Il s'agit de :</li>
 <ul> 
  <li>Continuous ε decay RASRN;</li>
  <li>Full RASRN;</li>
  <li> Stepwise α RASRN </li>
 </ul>
</ol>

Les environnements dans lesquels nous menont nous expériences sont des versions du Bandit à k bras où k = 2, 3, 4, 5.

La seule chose que vous avez à faire pour voir nos résultats est d'exécuter la commande: 
 ``` python3 run.py ``` ou parcourir les dossiers images ci-dessus qui contiennent les résultats de certaines éxécutions avec des épisodes de longueurs différentes.
