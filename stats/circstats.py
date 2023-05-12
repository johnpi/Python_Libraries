# Python reimplementation of https://rdrr.io/cran/circular/src/R/mle.vonmises.bootstrap.ci.R
# It gives a Maximum likelihood estimation of the Confidence Intervals of the angular mean.
# As well as some other circular statistics functions based on doi: 10.18637/jss.v031.i10
# (c) 2022-2023 Ioannis Pisokas

import numpy as np

# -----------------------------------------------------------------------------
# Functions for calculating the mean vector length of an array of angles.
# Based on doi: 10.18637/jss.v031.i10
# -----------------------------------------------------------------------------
def mean_resultant_vector(u):
    """ Returns the mean resultant vector of an array of angles u.
        Returns a two component array representing the (x, y) 
        components of the mean vector. 
        u : is an array of angles in radians. """
    r = np.array([np.cos(u), 
                  np.sin(u)])
    N = len(u)
    r_mean = np.sum(r, axis=1) / N
    return r_mean 


def resultant_vector_length(u):
    """ Returns the length of the mean resultant vector of an array of angles u.
        u : is an array of angles in radians. 
        It can be used as a measure of whether the unit vectors point in the 
        same direction (returns 1.0) or are uniformly distributed spanning the 
        360deg (returns 0.0). """
    r_mean = mean_resultant_vector(u)
    return np.sqrt(np.sum(r_mean**2))


def r_vector_length(u):
    """ Alternative name for the length of the mean resultant vector of an 
        array of angles u. 
        u : is an array of angles in radians. """
    return resultant_vector_length(u)
# -----------------------------------------------------------------------------


def circ_mean(x):
    """ Returns the mean vector direction
        x   : The data as an array of angles in radians.
    """
    sinr = np.sum(np.sin(x))
    cosr = np.sum(np.cos(x))
    mu = np.arctan2(sinr, cosr)
    return mu


def boot(data, statistic, repetitions = 1000, ensemble_size = 40):
    """ Simple Bootstrap implementation. 
        Gets <repetitions> Subsamples from <data> and calculates <statistic>, 
        the results are returned in an array. 
    """
    data_res = np.empty((repetitions))
    for i in range(repetitions):
        subsample = np.random.choice(data, size=ensemble_size, replace=True)
        data_res[i] = statistic(subsample)
    return data_res


def mle_vonmises_bootstrap_CI(u, mu=None, alpha = 0.05, reps = 1000):
    """ Uses bootstrapping to calculate the Confidence Intervals of the mean angle. 
        Uses Maximul likelihood estimation of the Confidence Intervals of the angular mean.
        The data are given as the value of the u parameter. 
        u      : An array with angles in radians. 
        mu     : (Optional) The angular mean of the data in u.
        alpha  : The confidence interval level, default 95% (0.05).
        reps   : How many times to resample from the data in u.
        Returns: A dict with 
                 result['mu'] array of the means calculated for subsamples of the data.
                 result['mu_ci'] array with the low and high Confidence Interal values.
    """
    
    if mu is None:
        mu = circ_mean(u)
    #bs = boot(data = u, statistic = MleVonmisesMuRad, repetitions = reps, ensemble_size = int(len(u)/2))
    bs = boot(data = u, statistic = circ_mean, repetitions = reps, ensemble_size = int(len(u)/2))
    
    mean = {}
    mean_reps = bs
    mean_reps = np.sort(mean_reps % (2 * np.pi)) # Sort the bootstrap means
    spacings = np.append(np.diff(mean_reps), mean_reps[0] - mean_reps[reps-1] + 2 * np.pi) # Get differences all around
    #max_spacing = range(0, reps)[spacings == np.max(spacings)]
    max_spacing = np.argmax(spacings)
    
    if (max_spacing+1 != reps):
        off_set = 2 * np.pi - mean_reps[max_spacing+1] # wraps around if it exceed the N in this example data case
        mean_reps2 = mean_reps + off_set
    else:
        mean_reps2 = mean_reps
    mean_reps2 = np.sort(mean_reps2 % (2 * np.pi))
    mean_ci = np.quantile(mean_reps2, [alpha/2, 1 - alpha/2])
    if (max_spacing+1 != reps):
        mean_ci = mean_ci - off_set
    
    result = {}
    result['mu'] = mean_reps
    result['mu_ci'] = mean_ci
    
    return result

# Define alternative more recognisable name for mle_vonmises_bootstrap_CI()
circ_CI = mle_vonmises_bootstrap_CI

def circ_summary(u, rads = True):
    """ Returns some summary statistics about angular data.
        The data are given as the value of the u parameter. 
        u      : An array with angles in radians. 
        rads   : (Optional) Specifies if the data in u are in radians.
                 If data in u are in degrees this should be set to False.
        Returns: A tuple with
                 The Angular mean of the data in u.
                 The 95% Confidence Intervals of the angular mean.
                 The circular variance of the data. 
                 The circular standard deviation of the data. 
    """
    # u is an angular variable
    # if the data are in degrees we transform them into radians
    if not rads: u = u * np.pi/180

    n = len(u) # sample size
        
    C = np.sum( np.cos(u) ) / n
    S = np.sum( np.sin(u) ) / n
    est = [C, S]
    
    # mean resultant length
    Rbar = np.sqrt( C**2 + S**2 )
    
    # mu contains the sample mean direction
    mu = np.arctan2(est[1], est[0]) % (2 * np.pi)
    
    # Sample circular variance
    circvar = 1 - Rbar
    # Sample cicrular standard deviation
    circstd = np.sqrt(-2 * np.log(Rbar))
    
    # Rsultant vector length
    R = n * Rbar
    
    # Calculate the CI
    result = mle_vonmises_bootstrap_CI(u, mu = mu, alpha = 0.05, reps = 1000)
    ci = result['mu_ci']
    
    return (mu, ci, circvar, circstd)


# -----------------------------------------------------------------------------
# USAGE: 
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    
    print('Demonstration of caluclating the Confidence Intervals of the angular mean of a sample.')
    print()
    
    # To generate a sample from the von Mises distribution
    from scipy.stats import vonmises
    u = vonmises.rvs(15, loc=0, size=340)
    
    # Get the Confidence Intervals for the angular mean
    result = mle_vonmises_bootstrap_CI(u, mu = None, alpha = 0.05, reps = 1000)

    print('Bootstrap Confidence Intervals for Mean Direction')
    print('Confidence Level:            {}% '.format((1-0.05)*100))
    print('Mean Direction:              Low CI = {:9.6f}   High CI = {:9.6f} '.format(result['mu_ci'][0], result['mu_ci'][1]))
    print('Mean Direction:              Low CI = {:5.2f}   High CI = {:5.2f} '.format(result['mu_ci'][0], result['mu_ci'][1]))
    
    
    # Get some descriptive statistics about the data
    mu, CI, circvar, circstd = circ_summary(u, rads = True)
    
    print('Angular mean:                {} '.format(mu))
    print('Circular variance:           {} '.format(circvar))
    print('Circular standard deviation: {} '.format(circstd))

