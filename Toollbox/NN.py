import numpy as np
from math import exp

class NN:
    
    def __init__( self, inputs:int, outputs:int, hiden_layers:list ):
        
        self._num_inp = inputs
        self._num_out = outputs
        self._num_hl = hiden_layers
        
        self._inp_matrix = np.zeros(( self._num_inp, 1))
        self._out_matrix = np.zeros(( 1, self._num_hl[-1] ))
        
        self._hl_matrix = []
        self._beas_matrix = []
        
        self._rr_hl_matrix = []
        self._last_hl_matrixes = []
        
        self._create_hd()
        self._creeate_bias()
        self._create_rr()
        
    def _min_max_scaling( self, value, min_original = -1, max_original = 1, min_desired = -40, max_desired = 40):
        return (value - min_original) / (max_original - min_original) * (max_desired - min_desired) + min_desired

        
        
    def _create_hd( self ) -> np.array:
        for index in range(len(self._num_hl)):
            
            if index == 0:
                inp = self._num_inp
            else:
                inp = self._num_hl[index - 1]
            
            out = self._num_hl[index]
            
            self._hl_matrix.append( np.zeros(( out, inp )) )
        
    def _create_rr( self ):
        [self._rr_hl_matrix.append(np.zeros(( size, 1 ))) for size in self._num_hl]
        [self._last_hl_matrixes.append(np.zeros(( size, 1 ))) for size in self._num_hl]
            
        
    def _creeate_bias( self ):
        [self._beas_matrix.append( np.zeros(( size, 1 )) ) for size in self._num_hl]
        
    def _fill( self, matrixes, feed):
        index_global = 0
        for matrix in matrixes:
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    matrix[i][j] = feed[index_global]
                    index_global += 1
                    
    def fill_rr( self, feed ):
        self._fill(self._rr_hl_matrix, feed)
            
    def fill_hl( self, feed ):
        self._fill( self._hl_matrix, feed)
        
    def fill_output( self, feed ):
        index_global = 0
        for i in range(self._out_matrix.shape[0]):
            for j in range(self._out_matrix.shape[1]):
                self._out_matrix[i][j] = feed[index_global]
                index_global += 1
        
    def fill_beas( self, feed ):
        self._fill( self._beas_matrix, feed )
    
    def _sigmoid( self, x ):
        value = 1.0 / (1.0 + exp(-x))
        return self._min_max_scaling( value )
    
    def _tang( self, x ):
        value = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        return self._min_max_scaling( value )
    
    def _chose_af( self, x, af):
        if af == 0:
            return  self._sigmoid(x)
        elif af == 1:
            return self._tang(x)
    
    def _one_calculation( self, w, x, b, af = 1):
        
        pre_AF = w @ x - b
        for index in range(pre_AF.shape[0]):
            pre_AF[index][0] = self._chose_af( pre_AF[index][0], af )
            
        return pre_AF
    
    def _fill_input( self, input ):
        index = 0
        for i in range(self._inp_matrix.shape[0]):
            self._inp_matrix[i][0] = input[index]
            index += 1
            
    def calculate( self, input, af = 1 ):
        
        self._fill_input( input )
        prev_value = self._inp_matrix
        
        for index in range(len(self._hl_matrix)):
            
            w = self._hl_matrix[index]
            b = self._beas_matrix[index]
            
            prev_value = self._one_calculation( w=w, x=prev_value, b=b, af=af )
            
            
        out = self._one_calculation( w=self._out_matrix, x=prev_value, b=np.zeros(( 1, 1 )), af=af) 
        
        return out[0][0]
            
    def calculate_rr( self, input, af = 1 ):
        
        self._fill_input( input )
        prev_value = self._inp_matrix
        
        for index in range(len(self._hl_matrix)):
            
            w = self._hl_matrix[index]
            b = self._beas_matrix[index]
            wu = self._rr_hl_matrix[index]
            u = self._last_hl_matrixes[index]
            
            prev_value = self._one_calculation_rr( wx=w, x=prev_value, b=b, wu=wu, u=u, af=af )
            self._last_hl_matrixes[index] = prev_value
            
        out = self._one_calculation( w=self._out_matrix, x=prev_value, b=np.zeros(( 1, 1 )), af=af) 
        
        return out[0][0]
        
    def _one_calculation_rr( self, wx, x, b, wu, u, af = 1):
        # wx - weight of input
        # x  - input
        # wu - weight of recurrent input
        # u  - input from last result 
        
        wu_last = np.zeros( (int(wu.shape[0]), int(wu.shape[0])) )
        for index in range(wu.shape[0]):
            wu_last[index][index] = wu[index][0]
        pre_AF = (wx @ x + wu_last @ u) - b
        for index in range(pre_AF.shape[0]):
            pre_AF[index][0] = self._chose_af( pre_AF[index][0], af )
            
        return pre_AF
            
        