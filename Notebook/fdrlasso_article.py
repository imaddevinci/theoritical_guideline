import numpy as np
from sklearn import linear_model
from scipy.stats import norm
from scipy.optimize import fminbound
import matplotlib.pyplot as plt



def fdrlasso(tpp, delta, epsi):
    """
    # https://github.com/wjsu/fdrlasso/blob/master/fdrlasso.m
    % This function calculates the Lasso trade-off curve given tpp (true
    % positive proportion), delta = n/p (shape of the design matrix, or
    % subsampling rate), and epsi = k/p (sparsity ratio).
    % All tpp, delta, and epsi are between 0 and 1; if the
    % pair (delta, epsi) is above the Donoho-Tanner phase transition, tpp
    % should be no larger than u^\star = powermax(delta, epsi)
    %--------------------------------------------------------------------------
    % Copyright @ Weijie Su, Malgorzata Bogdan, and Emmanuel Candes, 2015
    %--------------------------------------------------------------------------
    """

    if tpp > powermax(delta, epsi):
      #print('Invalid input!')
      return None 

    if tpp == 0:
      q = 0;
      return q

    # make stepsize smaller for higher accuracy
    stepsize = 0.1
    tmax = max(10, np.sqrt(delta/epsi/tpp) + 1)
    tmin = tmax - stepsize

    while tmin > 0:
      if lsandwich(tmin, tpp, delta, epsi) < rsandwich(tmin, tpp):
        break
        
      tmax = tmin
      tmin = tmax - stepsize

    if tmin <= 0:
      stepsize = stepsize/100;
      tmax = max(10, np.sqrt(delta/epsi/tpp) + 1)
      tmin = tmax - stepsize
      while tmin > 0:
        if lsandwich(tmin, tpp, delta, epsi) < rsandwich(tmin, tpp):
          break
        
        tmax = tmin
        tmin = tmax - stepsize

    diff = tmax - tmin
    while diff > 1e-6:
      tmid = 0.5*tmax + 0.5*tmin
      if lsandwich(tmid, tpp, delta, epsi) > rsandwich(tmid, tpp):
        tmax = tmid
      else:
        tmin = tmid
        
      diff = tmax - tmin

    t = (tmax + tmin)/2

    q = 2*(1-epsi)*norm.cdf(-t)/(2*(1-epsi)*norm.cdf(-t) + epsi*tpp)

    return q


def lsandwich(t, tpp, delta, epsi):
    Lnume = (1-epsi)*(2*(1+t**2)*norm.cdf(-t) - 2*t*norm.pdf(t)) + epsi*(1+t**2) - delta;
    Ldeno = epsi*((1+t**2)*(1-2*norm.cdf(-t)) + 2*t*norm.pdf(t));
    L = Lnume/Ldeno
    return L



def rsandwich(t, tpp):
    R = (1 - tpp)/(1 - 2*norm.cdf(-t));
    return R



# highest power for delta < 1 and epsilon > epsilon_phase
def powermax(delta, epsilon):
    if delta >= 1:
      power = 1
      return power 
    epsilon_star = epsilonDT(delta)
    
    if epsilon <= epsilon_star:
      power = 1
      return power

    power = (epsilon - epsilon_star)*(delta - epsilon_star)/epsilon/(1 - epsilon_star) + epsilon_star/epsilon;
    return power


def epsilonDT(delta):
    minus_f = lambda x: -(1+2/delta*x*norm.pdf(x) - 2/delta*(1+x**2)*norm.cdf(-x))/(1+x**2-2*(1+x**2)*norm.cdf(-x)+2*x*norm.pdf(x))*delta
    alpha_phase = fminbound(minus_f, 0, 8)
    #epsi = -feval(minus_f, alpha_phase)
    epsi = -minus_f(alpha_phase)
    return epsi