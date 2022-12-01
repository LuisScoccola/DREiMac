import numpy as np
import matplotlib.pyplot as plt
import random
def random_circle_walk(num_steps, start_position = 0):
    y = start_position
    # Generate the time points [1, 2, 3, ... , n]
    timepoints = np.arange(num_steps)
    positions = [y]
    directions = ["UP", "DOWN"]
    for i in range(1, num_steps):
        # Randomly select either UP or DOWN
        step = random.choice(directions)
        
        # Move the object up or down
        if step == "UP":
            y += 0.1
        elif step == "DOWN":
            y -= 0.1
        # Keep track of the positions
        positions.append(y)
    for i in range(len(positions)):
        positions[i] = positions[i] % 1
    return timepoints, positions

def skipping_circle_walk(num_walks, walk_size):
    path = []
    for walk in range(num_walks):
        start = np.random.uniform(0,1)
        timepoints, positions = random_circle_walk(walk_size, start)
        path.extend(positions)
    return path