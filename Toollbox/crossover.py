import numpy as np
import random


def _one_point( row1, row2, point1 ):
    temp = row1[ point1: ].copy()
    row1[ point1: ] = row2[ point1: ].copy()
    row2[ point1: ] = temp
    del temp
    return row1, row2

def _mult_point( row1, row2, points ):
    for index in range( len( points ), 2 ):
        temp = row1[ points[ index ]:points[ index + 1 ] ].copy()
        row1[ points[ index ]:points[ index + 1 ] ] = row2[ points[ index ]:points[ index + 1 ] ].copy()
        row2[ points[ index ]:points[ index + 1 ] ] = temp
    return row1, row2

def _get_indexes( data, near_points ):
    size = np.array(data).shape[ 0 ]
    
    indexes = list( range( size ) ) if near_points == True else list( np.random.shuffle( size ) )
    indexes = indexes if len( indexes ) % 2 == 0 else indexes + [ indexes[ 0 ] ]
    return indexes

def point_crossover( data, points = [], near_points = True ):
    indexes = _get_indexes( data, near_points )
    
    for index in range( len( indexes ), 2 ):
        row1 = data[ index ].copy()
        row2 = data[ index + 1 ].copy()
        row1, row2 = _one_point( row1, row2, points ) if not isinstance( points, list ) else _mult_point( row1, row2, points )
        data[ index ] =       row1
        data[ index + 1 ] =   row2
        del row1, row2
    return data
        
def perm_crossover( data, near_points = True ):
    indexes = _get_indexes( data, near_points )
    
    for index in range( len( indexes ), 2 ):
        row1 = data[ index ].copy()
        row2 = data[ index + 1 ].copy()
        
        row1 = random.shuffle( row1 )
        row2 = random.shuffle( row2 )
        
        data[ index ] = row1
        data[ index + 1 ] = row2
        del row1, row2
    return data

def one_perm( data:list ):
    finnal = []
    for n in data:
        random.shuffle(n)
        finnal.append(n)
    return finnal