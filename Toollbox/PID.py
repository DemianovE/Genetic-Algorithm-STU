class PIDController:
    def __init__( self, kp:float, ki:float, kd:float, t_0:int, Ts:int ):
        self._kp: float = kp  # Proportional gain
        self._ki: float = ki  # Integral gain
        self._kd: float = kd  # Derivative gain

        self._e: float = .0
        self._u: float = .0
        
        self._accumulated_error: float = .0
        self._last_error: list = [.0, .0]
        self._last_u:float = .0
        self._last_t: int = t_0
        
        self._Ts = Ts
        self._umax = 5
        self._umin = -5

    def _limits( self, value:float, max:int, min:int ) -> float:
        return max if value > 1 else min if value < 0 else value
    
    def compute( self, state:float, target:float ) -> float:
        
        self._e = target - state
        up = self._kp * self._e
        
        ui = self._Ts * 0.5 * (self._last_error[0] - self._e)
        if self._accumulated_error >= self._umax or self._accumulated_error <= self._umin:
            ui = 0
            
        ud = ((self._e - self._last_error[0]) / self._Ts) * self._kd
         
        self._u = self._accumulated_error + up + ui + ud
        
        self._last_error[1] = self._last_error[0]
        self._last_error[0] = self._e
        
        self._accumulated_error = self._u

        print(f"{up} - {ui} - {ud}")
        
        if self._u < self._umin:
            return self._umin, up, ui, ud
        elif self._u > self._umax:
            return self._umax, up, ui, ud
        return self._u, up, ui, ud