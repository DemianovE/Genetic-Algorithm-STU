import sys
sys.path.append('Toollbox/')

from init import *
import matplotlib.pyplot as plt 
import random, json, datetime
import scipy.signal as signal
import numpy as np

NUM_OF_POP = 20
GENERATIONS = 100
POP_SIZE = 160

POP_MIN = 0
POP_MAX = 1

ENDS_HL = [0, 130]
ENDS_BEAS = [130, 150]
ENDS_OUT = [150, 160]

NOW = datetime.datetime.now()

inp = 3
out = 1
hl = [10, 10]

def step_signal():
    step = 0.1
    range_x = [round(x, 1) for x in np.arange(0, 40 + step, step)]
    range_y = [0 if n < 10 else 0.5 if n <= 20 else  0.8 if (n > 20 and n < 30) else 0.4 for n in np.arange(0, 40 + step, step)]
    return range_x, range_y

def fit_funk( pop, nn ):
    result = []
    full_pid = []
    index_glob = 0
    for one in pop:
        print(f" - [{index_glob}] - START")
        index_glob += 1
        nn.fill_hl( one[ ENDS_HL[0]: ENDS_HL[1] ] )
        nn.fill_beas( one[ ENDS_BEAS[0]: ENDS_BEAS[1] ] )
        nn.fill_output( one[ ENDS_OUT[0]: ENDS_OUT[1] ] )
        
        system = signal.TransferFunction([1], [1, 2, 1])
        time, values = step_signal()
        
        position = 0
        last_error = 0
        target = 0
        pid_output = []
        whole_error = 0

        PID_controll_value = []  
        for index in range(len(time)):
            value = values[index]
            
            target = value
            last_error = value - position
            whole_error += abs(last_error)
            
            cv = nn.calculate( [ target, last_error, position])
            
            PID_controll_value.append(cv)
            _, responce, _ = signal.lsim(system, PID_controll_value[:index + 1], time[:index+1], X0=0)
            
            if str(responce) == "0.0":
                position = 0
            else:
                position = list(responce)[-1]
            pid_output.append(position)
        result.append(whole_error)
        full_pid.append(pid_output)
    return result, full_pid

def save( data, data2 ):
    with open(f'pop_{NOW}.json', 'w') as f:
        json.dump( np.array( data ).tolist(), f)
        
    with open(f'fit_{NOW}.json', 'w') as f:
        json.dump( np.array( data2 ).tolist(), f)

def plot_plt( values, best ):
    
    time, signal = step_signal()
    
    
    
    for pl in values:
        plt.plot(time, pl, linestyle='--', color='orange')
        
    plt.plot(time, best, linestyle='--', color='red')
    plt.plot(time, signal, label="Setpoint", color='black')
    plt.show()

if __name__ == "__main__":
    nn = NN( inp, out, hl )

    pop = [[random.uniform(POP_MIN, POP_MAX) for _ in range( POP_SIZE )] for _ in range(NUM_OF_POP)]
    best_overall = []

    for _ in range( GENERATIONS ):
        
        print(f"[{_}] - DONE")
        
        fit, values_plt = fit_funk( pop, nn )
        best_overall.append(min(fit))
        
        index_min = np.argmin(fit)
        
        best = select.worst( pop, score = fit, num_of_points = [1, 1, 1] )
        
        rest_1 = select.worst( pop, score = fit, num_of_points = [3, 3, 2, 2, 2 ] )
        rest_2 = select.random( pop, 5 )
        
        rest_1 = crossover.point_crossover( rest_1, [ 10, 80, 140 ] )
        rest_2 = crossover.point_crossover( rest_2, [ 30, 100, 130 ] )
        
        rest_1 = mutations.mutation_main( rest_1, 0.05 )
        rest_2 = mutations.mutation_main( rest_2, 0.3 )
        best_1 = select.worst( pop, fit, [1])
        
        print(min(fit))

        pop = list(np.concatenate(( np.array( best ), np.array( rest_1 ), np.array( rest_2 )), axis=0))
        save(pop, fit)

    plot_plt( values_plt, values_plt[index_min] )