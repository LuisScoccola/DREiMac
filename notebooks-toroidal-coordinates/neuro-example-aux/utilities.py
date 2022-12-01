import numpy as np

# Generates random neurons on a circle thought of as R/Z
# so each neuron is represented as a point in [0,1) and 1 = 0
def generate_random_neurons_on_circle(num_neurons):
    neuron_positions = np.random.uniform(0,1, num_neurons)
    return neuron_positions

# Generates neurons uniformly distributed on circle
def generate_uniform_neurons_on_circle(num_neurons):
    neuron_positions = np.linspace(0,1,num_neurons)
    return neuron_positions

# Generates a 1 or 0; chance of a 1 is 1/probability
def random_binary(probability):
    prob = np.floor(probability)
    guess = np.random.randint(0, prob)
    if guess == 0:
        return 1
    else:
        return 0

    

# Generates a num_rows by num_cols matrix with uniformly random floats between min and max
def generate_random_uniform_conn_matrix(num_rows, num_cols, min, max):
    matrix = np.random.uniform(min, max, (num_rows, num_cols))
    return matrix

# Generates a random bias vector of length size with each entry a uniformly random float between min and max
def generate_random_uniform_bias_vector(min, max, size=1):
    bias_vector = np.random.uniform(min, max, size)
    return bias_vector

# Generates evenly spaced circular path around a circle 
def generate_circular_path(winding_number, num_steps, start_position):
    raw_points = np.linspace(0 + start_position, winding_number + start_position, num_steps)
    points_on_circle = raw_points % 1
    return points_on_circle




# Calculates distance between two points on a circle thought of as R/Z
def calculate_circle_distance(phi, theta):
    if 0<=phi and phi < 1 and 0<= theta and theta < 1:
        difference = np.abs(phi-theta)
        d_1 = difference
        d_2 = 1 - difference
        distance = min(d_1, d_2)
        return distance
    elif phi == 1 and 0<= theta <= 1:
        distance = min(1-theta, theta)
        return distance
    elif 0 <= phi <=1 and theta == 1:
        distance = min(phi, 1 - phi)
        return distance
    else:
        raise Exception("Invalid angles plugged into `calculate_circle_distance'")



def calculate_neu_reponse_matrix_reg_2(reg_1_response_matrix, connection_matrix, vector_of_biases):
    neu_resp_matrix_region_2 = np.matmul(connection_matrix, reg_1_response_matrix)
    for row_index in range(len(neu_resp_matrix_region_2[:, 0])):
        for column_index in range(len(neu_resp_matrix_region_2[0, :])):
            neu_resp_matrix_region_2[row_index, column_index] = neu_resp_matrix_region_2[row_index, column_index] - vector_of_biases[row_index]
    neu_resp_matrix_region_2 = relu(neu_resp_matrix_region_2)
    return neu_resp_matrix_region_2



# Defines relu function
def relu(x):
    return np.maximum(0,x)



# Defines heavyside function
def heavyside_response_function(heavyside_parameter):
    def response_function(distance):
        if distance <= heavyside_parameter:
            return 1
        else:
            return 0
    return response_function


# Defines relu(linear response function)
# Note that "slope" should be negative
def linear_relu_response_function(max_rate, slope):
    if slope >= 0:
        raise Exception("Non-negative slope input to linear relu response function.")
    else:
        def response_function(distance):
            naive_firing_rate = max_rate + slope*distance
            corrected_firing_rate = relu(naive_firing_rate)
            return corrected_firing_rate
        return response_function





# Input: list of neurons as positions on a circle, list of points on circle representing a path, 
# and reponse_function: distance |-> firing rate
# Output: matrix whose (i, j) entry is firing rate of neuron i at step j
def calculate_neural_response_matrix(neurons_on_circle, path_on_circle, response_function):
    num_neurons = len(neurons_on_circle)
    num_steps = len(path_on_circle)
    response_matrix = np.zeros((num_neurons, num_steps))
    for neuron_index in range(num_neurons):
        for path_index in range(num_steps):
            distance = calculate_circle_distance(neurons_on_circle[neuron_index], path_on_circle[path_index])
            response = response_function(distance)
            response_matrix[neuron_index, path_index] = response
    return response_matrix

