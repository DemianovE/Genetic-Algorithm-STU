import sys
sys.path.append('Toollbox/')


from Toollbox.init import *
import matplotlib.pyplot as plt 
NUM_OF_POP = 100
GENERATIONS = 10000

start_pop = list(range(1, 19))


pop = [start_pop.copy() for _ in range(NUM_OF_POP)]

best_overall = []

for _ in range( GENERATIONS ):
    
    fit, min_val = custom_fit.point_coord_fit( pop )
    best_overall.append(min_val)
    
    best = select.best( pop, score = fit, num_of_points = [5, 3, 12] )
    
    rest_1 = select.best( pop, score = fit, num_of_points = [10, 10, 10, 10] )
    rest_2 = select.random( pop, 40 )
    
    rest_1 = crossover.one_perm( rest_1 )
    rest_2 = mutations.mutation_shuffle_perm( rest_2 )
    best_1 = select.best( pop, fit, [1])

    pop = list(np.concatenate(( np.array( best ), np.array( rest_1 ), np.array( rest_2 )), axis=0))
    print(f"[{_}] - DONE - [{best_overall[-1]}]")

plt.plot(range(len(best_overall)), best_overall)
plt.show()

print(f"best - [{max(fit)}]")
# Extract x and y coordinates
data_points = [[0, 0], [77, 68], [12, 75], [32, 17], [51, 64],
               [20, 19], [72, 87], [80, 37], [35, 82], [2, 15],
               [18, 90], [33, 50], [85, 52], [97, 27], [37, 67],
               [20, 82], [49, 0], [62, 14], [7, 60], [100, 100]]

# List of indexes of points to mark
# Extract the marked points based on the indexes
x = []
y = []
best_1 = best[0]
for index in best_1:
  index = int(index)
  x.append(data_points[index][0])
  y.append(data_points[index][1])
x = [0] + x + [100]
y = [0] + y + [100]
plt.scatter(x, y, marker='*', color='red', s=100, label='Marked Points')
plt.plot(x, y)

# Add labels and legend
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Graph with Marked Points')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()







