import sys
sys.path.append('Toollbox/')

from init import *
import matplotlib.pyplot as plt 
import random, json, datetime
import scipy.signal as signal
import numpy as np
import time
from multiprocessing.pool import ThreadPool

NUM_OF_POP = 40
GENERATIONS = 100
POP_SIZE = 160

POP_MIN = 0
POP_MAX = 1

ENDS_HL = [0, 130]
ENDS_BEAS = [130, 150]
ENDS_OUT = [150, 160]

DEV = False
OFF_FILE = False
FILE_PATH = "OUTPUT/"

NOW = datetime.datetime.now()

inp = 3
out = 1
hl = [10, 10]



def step_signal():
    step = 0.1
    range_x = [round(x, 2) for x in np.arange(0, 40 + step, step)]
    range_y = [0 if n < 5 or ( n >= 10 and n < 15 ) else 10 if (n >= 5 and n < 10) or (n >= 30 and n < 35) else 5 if (n >= 25 and n < 30) else 25 if (n >= 35) else 30 if (n>= 15 and n<20) else 35  for n in np.arange(0, 40 + step, step)]
    #range_y = [0 if n < 20 else 0.5 if n <= 60 else  0.8 if (n > 60 and n < 120) else 0.4 for n in np.arange(0, 200 + step, step)]
    return range_x, range_y


def fit_one( pop, nn ):
    
    result = []
    full_pid = []
    index_glob = 0
    one = pop
    start_time = time.time()
    index_glob += 1
    nn.fill_hl( one[ ENDS_HL[0]: ENDS_HL[1] ] )
    nn.fill_beas( one[ ENDS_BEAS[0]: ENDS_BEAS[1] ] )
    nn.fill_output( one[ ENDS_OUT[0]: ENDS_OUT[1] ] )
    
    system = signal.TransferFunction([1], [1, 2, 1])
    time_signal, values = step_signal()
    
    position = 0
    last_error = 0
    target = 0
    pid_output = []
    whole_error = 0

    PID_controll_value = []  
    for index in range(len(time_signal)):
        value = values[index]
        
        target = value
        last_error = value - position
        whole_error += abs(last_error)
        
        cv = nn.calculate( [ target, last_error, position])
        
        PID_controll_value.append(cv)
        _, responce, _ = signal.lsim(system, PID_controll_value[:index + 1], time_signal[:index+1], X0=0)
        
        if str(responce) == "0.0":
            position = 0
        else:
            position = list(responce)[-1]
        pid_output.append(position)
    print(f" - [{index_glob}] - START ({time.time() - start_time})")
    return whole_error, pid_output

def wrap(args):
    return fit_one(*args)

def threads_fit( pop, nn ):
    pool = ThreadPool(20)
    results = pool.map(wrap, zip(pop, [nn for _ in range(range(len(pop)))]))
    pool.close()
    pool.join()
    
    return []
    

def fit_funk( pop, nn ):
    
    result = []
    full_pid = []
    index_glob = 0
    for one in pop:
        start_time = time.time()
        index_glob += 1
        nn.fill_hl( one[ ENDS_HL[0]: ENDS_HL[1] ] )
        nn.fill_beas( one[ ENDS_BEAS[0]: ENDS_BEAS[1] ] )
        nn.fill_output( one[ ENDS_OUT[0]: ENDS_OUT[1] ] )
        
        system = signal.TransferFunction([1], [1, 2, 1])
        time_signal, values = step_signal()
        
        position = 0
        last_error = 0
        target = 0
        pid_output = []
        whole_error = 0

        PID_controll_value = []  
        for index in range(len(time_signal)):
            value = values[index]
            
            target = value
            last_error = value - position
            whole_error += abs(last_error)
            
            cv = nn.calculate( [ target, last_error, position])
            
            PID_controll_value.append(cv)
            _, responce, _ = signal.lsim(system, PID_controll_value[:index + 1], time_signal[:index+1], X0=0)
            
            if str(responce) == "0.0":
                position = 0
            else:
                position = list(responce)[-1]
            pid_output.append(position)
        result.append(whole_error)
        full_pid.append(pid_output)
        print(f" - [{index_glob}] - START ({time.time() - start_time})")
    return result, full_pid

def save( data, data2, data3 ):
    with open(f'{FILE_PATH}pop_{NOW}.json', 'w') as f:
        json.dump( np.array( data ).tolist(), f)
        
    with open(f'{FILE_PATH}pop_last.json', 'w') as f:
        json.dump( np.array( data ).tolist(), f)
        
    with open(f'{FILE_PATH}fit_{NOW}.json', 'w') as f:
        json.dump( np.array( data2 ).tolist(), f)
    with open(f'{FILE_PATH}fit_last.json', 'w') as f:
        json.dump( np.array( data2 ).tolist(), f)
    with open(f'{FILE_PATH}success_{NOW}.json', 'w') as f:
        json.dump( np.array( data3 ).tolist(), f)
    
def open_json():
    with open(f'{FILE_PATH}pop_last.json', 'r') as json_file:
        data = json.load(json_file)
        
    with open(f'{FILE_PATH}fit_last.json', 'r') as json_file:
        fit = json.load(json_file)
    index = np.argmin(fit)
    return [data[index].copy() for _ in range(NUM_OF_POP)]

def plot_plt( values, best, overall, axs ):
    
    time_signal, signal_value = step_signal()
    
    axs[0].clear()
    axs[1].clear()
    for pl in values:
        axs[0].plot(time_signal, pl, linestyle='--', color='orange')
        
    axs[0].plot(time_signal, best, color='red')
    axs[0].plot(time_signal, signal_value, label="Setpoint", color='black')
    
    axs[1].plot(range(len(overall)), overall)
    plt.pause(0.1)
    

if __name__ == "__main__":
    
    if DEV == False:
        NUM_OF_POP = 1
        GENERATIONS = 1
    
    nn = NN( inp, out, hl )

    if OFF_FILE == False:
        pop = [[random.uniform(POP_MIN, POP_MAX) for _ in range( POP_SIZE )] for _ in range(NUM_OF_POP)]
    else:
        pop = open_json()
    best_overall = []

    plt.show(block=False)
    time_signal, signal_value = step_signal()
    ig, axs = plt.subplots(2, 1)
    
    for _ in range( GENERATIONS ):
        
        print(f"[{_}] - DONE")
        
        fit, values_plt = fit_funk( pop, nn )
        best_overall.append(min(fit))
        print(f'Averange - {sum(fit)/ len(fit)}')
        index_min = np.argmin(fit)
        
        if DEV == True:
            best = select.worst( pop, score = fit, num_of_points = [4, 4] )
            
            rest_1 = select.worst( pop, score = fit, num_of_points = [14, 10] )
            rest_2 = select.random( pop, 8 )
            
            rest_1 = crossover.point_crossover( rest_1, [ 10, 80, 140 ] )
            rest_2 = crossover.point_crossover( rest_2, [ 30, 100, 130 ] )
            
            rest_1 = mutations.mutation_main( rest_1, 0.05 )
            rest_2 = mutations.mutation_main( rest_2, 0.4 )
            best_1 = select.worst( pop, fit, [1])
            
            print(min(fit))

            pop = list(np.concatenate(( np.array( best ), np.array( rest_1 ), np.array( rest_2 )), axis=0))
            save(pop, fit, best_overall)
            
        plot_plt( values_plt, values_plt[index_min], best_overall, axs )
    plt.show()