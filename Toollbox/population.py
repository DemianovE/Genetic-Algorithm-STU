import numpy as np

#############################################################################
#                      The Population Class                                 #
#       Can be used to perform Population manipulations for the GA.         #                                             
#       How to use:                                                         #
#           - create( size, limits ) - creation of the new population       #
#                 - size - ( y, x ) of the matrix                           #
#                 - limits - ( low, high ) of the limits                    #
#           - new( best, rest ) - creation of population based on           #
#               best and rest matricex. Uses concatination                  #
#                 - best - best values matrix                               #
#                 - rest - matrix with values after mutation and            #
#                       crossover.                                          #
#           - show() - returnes population                                  #
#############################################################################


class Population:

    def __init__( self ):
        self.pop = None

    def create( self, size:dict, limits:dict = None ):
        self.pop = np.array( np.random.rand( *tuple( size ) ) ) if limits == None else np.array( np.random.uniform( low = limits[0], high = limits[1], size = size ) ) 

    def new( self, best, rest ):
        self.pop = np.concatenate( ( best, rest ), axis=0 )
    
    def show( self ):
        return self.pop