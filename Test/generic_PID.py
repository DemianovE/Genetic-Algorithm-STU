import matplotlib.pyplot as plt
import scipy.signal as signal
import numpy as np

def step_signal():
    step = 0.1
    range_x = [round(x, 1) for x in np.arange(0, 40 + step, step)]
    #range_y = [0 if n < 30 or ( n >= 100 and n < 160) or n >= 220 else 1 for n in np.arange(0, 300 + step, step)]
    #range_y = [0 if n < 20 else 0.5 if n <= 60 else  0.8 if (n > 60 and n < 120) else 0.4 for n in np.arange(0, 200 + step, step)]
    range_y = [0 if n < 5 or ( n >= 10 and n < 15 ) else 10 if (n >= 5 and n < 10) or (n >= 30 and n < 35) else 5 if (n >= 25 and n < 30) else 25 if (n >= 35) else 30 if (n>= 15 and n<20) else 35  for n in np.arange(0, 40 + step, step)]
    print(range_y)
    return range_x, range_y

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
        self._umax = 40
        self._umin = -40

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

if __name__ == "__main__":
    system = signal.TransferFunction([1], [1, 2, 1])
    time, values = step_signal()

    # PID gains
    kp = 1
    ki = 0.8
    kd = 0.05

    # Create a PID controller
    pid = PIDController( kp, ki, kd, 0, 0.1 )

    num = [1]  # Numerator coefficients
    den = [1, 2]  # Denominator coefficients (1 + 2s)
    system = signal.TransferFunction(num, den)
    
    position = 0
    values_PID = []
    pid_output = []
    time_tf = []

    P = []
    I = []
    D = []
    PID_controll_value = []  
    for index in range(len(time)):
        t = time[index]
        value = values[index]
        
        cv, p, i, d = pid.compute( position, value )

        PID_controll_value.append(cv)

        _, responce, _ = signal.lsim(system, PID_controll_value[:index + 1], time[:index+1], X0=0)
        
        if str(responce) == "0.0":
            position = 0
        else:
            position = list(responce)[-1]

        P.append(p)
        I.append(i)
        D.append(d)
        pid_output.append(position)
        time_tf.append(t)
        
        values_PID.append( pid_output )
        
    plt.plot(time, pid_output, label="Process Value", color='orange')
    plt.plot(time, list(values), linestyle='--', label="Setpoint", color='red')
    plt.plot(time, P)
    plt.plot(time, I)
    plt.plot(time, D)
    plt.xlabel("Time Steps")
    plt.legend(["SImulated value", "Target", "P", 'I', "D"])
    plt.show()
