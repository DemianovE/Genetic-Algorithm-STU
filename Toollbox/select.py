import numpy as np

def _basic_sort( data, score, num_of_points, start_value, add_value ):
    indexes = np.argsort( score )
    value = 0
    rows = []
    for index in num_of_points:
        for _ in range(index):
            rows.append( data[ indexes[ start_value + value ] ] )
            value += add_value
    return rows

def best( data, score, num_of_points ):
    return np.array( _basic_sort( data, score, num_of_points, -1, -1 ) )
    
def worst( data, score, num_of_points ):
    return np.array( _basic_sort( data, score, num_of_points, 0, 1 ) )

def random(data, num_of_points):
    indexes = np.random.choice(np.array(data).shape[0], num_of_points, replace=False)
    return np.array( np.array(data)[indexes] )