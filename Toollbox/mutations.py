import numpy as np
import random


def mutation_main( data, chance ):

    for index in range(len(data)):
        row = data[index][:]
        
        full_size = len(list(row))
        for index_d in range(len(row)):
            
            if random.randint( 0, 100 )  < chance * 100:
                row[index_d] = random.randint( 0, 1 )
            else:
                pass
        data[index] = row
        
    return np.array(data)

def mutation_shuffle_perm( data ):
    output = []
        
    for row in data:
        shuffled_indexes = np.random.permutation(len(row))
        output.append([row[i] for i in shuffled_indexes])
    return np.array( output )

