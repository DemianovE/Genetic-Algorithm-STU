import numpy as np
import random


def mutation_main( data, chance ):

    full_size = data.shape[0] + data.shape[1]
    
    for _ in range( len( full_size // 10 ) ):
        if random.randit( 0, 100 )  < chance * 100:
            full_size -= 1

    indices_x = np.random.randint( 0, data.shape[0], full_size )
    indices_y = np.random.randint( 0, data.shape[1], full_size )
    
    values = np.array( np.random.rand( *tuple( full_size, 0 ) ) )
    
    data[ indices_x, indices_y ] = values
    
    return data

def mutation_shuffle_perm( data ):
    output = []
        
    for row in data:
        shuffled_indexes = np.random.permutation(len(row))
        output.append([row[i] for i in shuffled_indexes])
    return np.array( output )

