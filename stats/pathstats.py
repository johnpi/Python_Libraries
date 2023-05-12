import numpy as np

# -----------------------------------------------------------------------------
# Functions for calculating path statistics.
# -----------------------------------------------------------------------------

def dist_to_length_ratio(trj):
    """ Returns the distance to path length ratio of the given path.
        trj is a pandas DataFrame with two columns x and y holding 
        the coordinate points along the path. 
        Output is in the range 0--1 """
    
    # Get the x and y values
    x = np.array(trj['x'].tolist())
    y = np.array(trj['y'].tolist())
    
    # The step dispacements
    dx = np.diff(x)
    dy = np.diff(y)
    dh = np.hypot(dx, dy)
    
    # The path length and total displacement
    path_length = np.sum(dh)
    displacement_distance = np.hypot(x[-1]-x[0], y[-1]-y[0])
    
    return displacement_distance/path_length


# -----------------------------------------------------------------------------
# USAGE: 
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    
    print('Demonstration of caluclating path characteristics.')
    print()
    
