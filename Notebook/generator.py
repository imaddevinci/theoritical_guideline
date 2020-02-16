import numpy as np
from sklearn import linear_model
from scipy.stats import norm
from scipy.optimize import fminbound
import matplotlib.pyplot as plt

# this function generate a gaussian distribution 
def generate_gaussian(n, p):
    X = np.random.randn(n, p) #X ~ N(0,1)
    return X

# this function generate true target data
def generate_true_y(X, nb_no_zero, value_no_zero, sigma, noise=False):
    n, p = X.shape
    beta = np.zeros(p)
    beta[0:nb_no_zero] = value_no_zero
    y = np.dot(X, beta)
    if noise is True:
        y = y + np.random.normal(0, sigma, n)
    return y, beta



def True_False(beta_hat,beta):
  # Now we will compute the true positive proportion TPP and
  # the the false discovery proportion FDP

  FDP = []
  TPP = []
  for i in range(beta_hat.shape[1]): #Pour chaque colonne de coefficients prédits (1000)
      # beta_hat_j != 0 and beata_j = 0
      FDP_numerator = len(np.where(np.logical_and(beta_hat[:,i] != 0, beta == 0) == True)[0]) #Nb de coeffs prédits non nuls et qui sont en fait nuls
      #print(FDP_numerator)
      FDP_denominator = max(len(np.where(beta_hat[:,i] != 0)[0]), 1)  #Nb total de coeffs prédits non nuls jusque là
      FDP.append(FDP_numerator/FDP_denominator)

      # beta_hat_j != 0 and beata_j != 0
      TPP_numerator = len(np.where(np.logical_and(beta_hat[:,i] != 0, beta != 0) == True)[0])  #Nb de coeffs prédits non nuls et qui bien sont nuls
      #print(TPP_numerator)
      TPP_denominator = max(len(np.where(beta != 0)[0]), 1) 
      TPP.append(TPP_numerator/TPP_denominator)

      if i > 0 and FDP[i] > 0 and FDP[i-1] == 0:
        print("TPP au premier faux positif : ", TPP[i])

      if TPP[i] >= 0.5 and TPP[i-1] < 0.5:
        print("FDP lorsque TPP = 50% : ", FDP[i])

      if TPP[i] == 1 and TPP[i-1] < 1:
        print("FDP lorsque TPP = 100% : ", FDP[i])

  return TPP,FDP



def simulation_lasso_path(nb_simulations, n=1000, p=1000, nb_no_zero=200, value_no_zero=4, sigma=1, noise=False):
  tpp_simules = np.zeros(nb_simulations)
  fdp_simules = np.zeros(nb_simulations)

  true = np.append(np.ones(nb_no_zero), np.zeros(p - nb_no_zero))

  for i in range(nb_simulations):
    X = generate_gaussian(n, p)
   # y, beta = generate_true_y(X, nb_no_zero, value_no_zero, sigma, noise=True)
    y = np.dot(X, true) + np.random.normal(0, sigma, n)
    _, beta_hat, __ = linear_model.lasso_path(X, y, n_alphas=100) 
    path = (beta_hat != 0) #décrit le choix du Lasso de sélectionner ou non la variable j

    #Compute histograms
    fdp_rate = 0.
    for j in range(nb_simulations - 1):
      choice_j = path[:, j]
      sum_choice_j = sum(choice_j)
      if sum_choice_j != 0:
        fdp_rate = ((sum((choice_j != true) * (choice_j == 1.)) / sum_choice_j))
      else:
        fdp_rate = 0
      tpp_simules[i] = sum((choice_j == true) * (true == 1.)) / nb_no_zero
      #On arrête au moment de la première fausse sélection
      if (fdp_rate != 0.):
            break
    
    tpp_rate = 0.
    for j in range(nb_simulations - 1):
      choice_j = path[:, j]
      sum_choice_j = sum(choice_j)
      tpp_rate = sum((choice_j == true) * (true == 1.)) / nb_no_zero
      
      if sum_choice_j != 0:
        fdp_simules[i] = ((sum((choice_j != true) * (choice_j == 1.)) / sum_choice_j))
      else:
        fdp_simules[i] = 0
      #On arrête lorsque TPP = 100%
      if (tpp_rate == 1):
            break

  return tpp_simules, fdp_simules



  


