
import numpy as np
import matplotlib.pyplot as plt

class BinsASRN:
    def __init__(self, waiting_period=0, boot_period=10000):
        self.boot_period = boot_period
        self.steps = 0
        self.waiting_period = waiting_period
        self.learning_period = self.boot_period / 10 # La learning period représente ici 1/10 de la boot period. 
        self.learning_experience = []
        self.n_bins = 5
        self.bins = []
        self.bins_noise = []
        self.training_done = False

    # Return a noised reward, noise is adaptive to the actual error. Smaller error causes bigger noise.
    def noise(self, estimated_Q, true_Q, reward):

        err = np.abs(estimated_Q - true_Q) # Calcul de la variance temporelle

        if self.steps < self.waiting_period:    # Période d'attente avant de commencer la phase d'apprentissage.
            noised_reward = reward
        elif self.steps < self.waiting_period + self.learning_period: # Ici, on est dans les premiers instants de la phase d'apprentissage.
            self.learning_experience.append([err, reward]) # On stocke dans un tableau les variances temporelles et les rewards associés aux zones.
            noised_reward = reward
        elif self.steps == self.waiting_period + self.learning_period: # Ici, nous sommes directement après les premiers instants de la phase d'apprentissage.
            learning_experience = np.asarray(self.learning_experience) 
            
            # print(f'learning_experience: {learning_experience}')
            
            exp = learning_experience[learning_experience[:, 0].argsort()] # Trie du tableau learning_experience selon les err croissantes. 
            
            # print(f'exp: {exp}')
            
            step_size = int(np.floor(self.learning_period/self.n_bins))    # Calcul du pas step_size: 10000 / 10 = 1000.
            self.bins = exp[0:int(self.learning_period)-1:step_size, 0]    # Initialisation des bins avec les variances obtenues lors des premiers instant de l'entraînement, et ce tous les step_size pas. Ça fait en tout 10 bins car le step_size = 1000
            
            # print(f'bins: {self.bins}')
            
            bins_stds = np.zeros(self.n_bins)							   # Initialisation à zéro du tableau qui contient les écart-types associés à chaque reward. 
            self.bins_noise = np.zeros(self.n_bins)						   # Initialisation à zero du tableau des bins bruités.
            for i in range(0, int(self.learning_period)-1, step_size):	   # de 0 à 9999 pas de 1000. En tout 10 tours de boucle, un pour chaque bin.
                data = exp[i:i+step_size, 1]							   # On récupère toutes les récompenses sur 1000 steps. Il s'agit des valeurs prise par la variable alétoire r_i.
                bins_stds[int(i / step_size)] = np.std(data)			   # On remplit le tableau des écart-types avec les réels écart-types obtenus grâce aux valeurs de la variable alétoire.
            wanted_err_mean = np.max(bins_stds)							   # On détermine l'écart-type maximal qui sera notre référence pour le calcul de l'amplitude des bruits à appliquer.
            self.bins_noise = np.sqrt(wanted_err_mean*wanted_err_mean - bins_stds*bins_stds) # Calcul des amplitudes du bruit à appliquer par bin (amplitude = écart-type et bin = variable alétoire)
            self.training_done = True                                      # Fin de l'entraînement.
            noised_reward = reward 										   
        else: 															   # Après la phase d'apprentissage le calcul des amplitudes à appliquer.
            assert self.training_done
            selected_bin = np.where(self.bins <= err)[0] 				   # Sélection de l'indice du bin où la variance est inférieure ou égale à la variance estimée par la fonction.
            if len(selected_bin) > 0:
                selected_bin = selected_bin[-1]
            else:
                selected_bin = 0
            noise = self.bins_noise[selected_bin]						   # Sélection de l'amplitude du bruit à appliquer.
            noised_reward = np.random.normal(reward, noise) 			   # Echantillonnage d'une récompense bruitée suivant la loi normale déterminée.  

        self.steps = (self.steps + 1) % self.boot_period
        if self.steps == 0:
            self.learning_experience = list()
            self.bins = list()
            self.bins_noise = list()
            self.training_done = False
        return noised_reward

