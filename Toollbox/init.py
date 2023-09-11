# This file is used to include all toolbox function of GA and NE algorithms
# All classes are imported seperately, the functions should be accessable only through type.

### GA Population ###
from Toollbox.population import Population


### Crossover functions ###
import Toollbox.crossover as crossover

### Mutation functions ###
import Toollbox.mutations as mutations

### Selection functions ###
import Toollbox.select as select

### Custom fit functions ###
import Toollbox.custom_fit as custom_fit

### Libs ###
import numpy as np
import matplotlib as plt
import random