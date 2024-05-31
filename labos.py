import numpy as np
import argparse
import sys


# Generate weights based on normal distribution
def gauss_normal():
   mean = 0
   st_dev = 0.1
   num_samples = 1
   return np.random.normal(mean, st_dev, num_samples)


# Logistic sigmoid function - chosen transition function
def logistic_sigmoid(x):
   return 1 / (1 + np.exp(-x))


# Calculate mean squared error
def calculate_error(expected_values, computed_values):
   n = len(computed_values[0])
   err = 0
   for index in range(n):
      err += (expected_values[index] - computed_values[0][index])**2
   err = err / n
   return err

# Finds best specimen among current population
def evaluate_population(population, expected_values, input_columns, num_layers):
   evaluation = {}
   for specimen in population:
      computed_values = specimen.calculate_nn_outputs(input_columns, num_layers)
      error = calculate_error(expected_values, computed_values)
      evaluation[specimen] = error
   return evaluation


def mutate_weights(weights, mutation_probability, gauss_noise_std):
   new_weights = []
   for weight in weights:
      if np.random.rand() < mutation_probability:
         weight += np.random.normal(0, gauss_noise_std)
      new_weights.append(weight)
   return new_weights


def mutate_bias(bias, mutation_probability, gauss_noise_std):
   if np.random.rand() < mutation_probability:
      bias += np.random.normal(0, gauss_noise_std)
   return bias


# Function for creating 
def cross_and_mutate(two_random_specimen, num_layers, num_neurons_per_layer, num_input, mutation_probability, gauss_noise_std):
   first = two_random_specimen[0]
   second = two_random_specimen[1]
   
   new_specimen = NeuralNetwork(num_layers, num_neurons_per_layer, num_input)
   
   # Cross weights from both specimen using mean and mutate them
   for i in range(num_layers):
      for j in range(num_neurons_per_layer):
         # Cross
         new_weights = []
         for k in range(len(first.layers[i].neurons[j].weights)):
            new_weight = (first.layers[i].neurons[j].weights[k] + second.layers[i].neurons[j].weights[k]) / 2
            new_weights.append(new_weight)
         # Mutate and set new weights
         new_weights = mutate_weights(new_weights, mutation_probability, gauss_noise_std)
         new_specimen.layers[i].neurons[j].weights = new_weights
            
         # Cross, mutate and set biases
         new_bias = (first.layers[i].neurons[j].bias + second.layers[i].neurons[j].bias) / 2
         new_bias = mutate_bias(new_bias, mutation_probability, gauss_noise_std)
         new_specimen.layers[i].neurons[j].bias = new_bias
    
   return new_specimen


# Fundamental unit of neural layer, takes in n inputs and computes one final value
class Neuron:
   def __init__(self, num_inputs):
      self.num_inputs = num_inputs
      self.bias = gauss_normal()
      self.weights = [gauss_normal() for i in range(num_inputs)]
      self.sum = 0
      self.output = 0
   
   def sum_all(self, input_values):
      self.sum = 0
      for i in range(len(input_values)):
         self.sum += input_values[i] * self.weights[i]
      self.sum += self.bias
    
   def transition_function(self):
      self.output = logistic_sigmoid(self.sum)
   

# Used as middle-man for computing outputs of a single neuron layer
class NeuralLayer:
   def __init__(self, num_neurons, num_input):
      self.num_neurons = num_neurons
      self.neurons = [Neuron(num_input) for i in range(num_neurons)]
   
   def calculate_layer_output(self, input_values):
      output = []
      for neuron in self.neurons:
         neuron.sum_all([input_values])
         neuron.transition_function()
         output.append(neuron.output)
      return output


# N inputs -> one final output
class NeuralNetwork:
   def __init__(self, num_layers, num_neurons_per_layer, num_input):
      self.layers = [NeuralLayer(num_neurons_per_layer, num_input) for i in range(num_layers)]
      self.bias = gauss_normal()

   def calculate_nn_outputs(self, input_columns, num_layers):
      nn_outputs = []
      for column in input_columns:
         for value in column:
            layer1 = self.layers[0]
            output_layer1 = layer1.calculate_layer_output(value)
            nn_output = sum(output_layer1)
            if num_layers == 2:
               layer2 = self.layers[1]
               output_layer2 = layer2.calculate_layer_output(output_layer1)
               nn_output = sum(output_layer2)
            nn_outputs.append(nn_output + self.bias)
      return nn_outputs


def main():
   # Parse arguments 
   parser = argparse.ArgumentParser()
   parser.add_argument("--train")
   parser.add_argument("--test")
   parser.add_argument("--nn")
   parser.add_argument("--popsize")
   parser.add_argument("--elitism")
   parser.add_argument("--p")
   parser.add_argument("--K")
   parser.add_argument("--iter")
   args = parser.parse_args()

   # Storing provided values in variables
   train_data = args.train
   test_data = args.test
   nn_architecture = args.nn
   popsize = int(args.popsize)
   elitism = int(args.elitism)
   mutation_probability = float(args.p)
   gauss_noise_std = float(args.K)
   iterations_number = int(args.iter)

   # Read training dataset
   with open(train_data, "r") as file:
      data = file.readlines()
      data = [line.strip("\n") for line in data]

   # Construct lists of variable columns
   variables = data[0].split(",")
   columns = [[] for nun in range(len(variables))]
   for row in data[1:]:
      values = row.split(",")
      for i, value in enumerate(values):
         columns[i].append(float(value))
   goal_variable_column = columns[-1]
   input_columns = columns[:-1]

   # Initialize n different neural networks; where n = popsize
   population = []
   for nun in range(popsize):
      if nn_architecture == "5s":
         num_layers = 1
         num_neurons = 5
         nn = NeuralNetwork(num_layers, num_neurons, len(input_columns))

      elif nn_architecture == "20s":
         num_layers = 1
         num_neurons = 20
         nn = NeuralNetwork(num_layers, num_neurons, len(input_columns))

      elif nn_architecture == "5s5s":
         num_layers = 2
         num_neurons = 5
         nn = NeuralNetwork(num_layers, num_neurons, len(input_columns))

      else:
         sys.exit("Given non supported nn architecture.")
      population.append(nn)

   # Evaluate population based on mean error squared
   # Generate sorted list based on MES, list[0]->best performing, smallest MES
   evaluation = evaluate_population(population, goal_variable_column, input_columns, num_layers)
   sorted_evaluation = {key: value for key, value in sorted(evaluation.items(), key=lambda item: item[1])}
   list_of_sorted_evaluation = [[key, value] for key, value in sorted_evaluation.items()]
   current_best_error = list_of_sorted_evaluation[0][1]

   # Run genetic algorithm for training population of neural networks
   for iteration in range(iterations_number):
      if iteration % 2000 == 0:
         print(f"[Train error @{iteration}]: {current_best_error}")
      
      # Creating new population that includes elites from last population
      new_population = []
      elite_specimens = [sublist[0] for sublist in list_of_sorted_evaluation[:elitism]]
      new_population.extend(elite_specimens)

      while len(new_population) < popsize:
         two_random_specimens = np.random.choice(population, size=2, replace=False)
         new_specimen = cross_and_mutate(two_random_specimens, num_layers, num_neurons, len(input_columns), mutation_probability, gauss_noise_std)
         new_population.append(new_specimen)

      # Evaluate new population
      population = new_population
      evaluate_population(population, goal_variable_column, input_columns, num_layers)
      sorted_evaluation = {key: value for key, value in sorted(evaluation.items(), key=lambda item: item[1])}
      list_of_sorted_evaluation = [[key, value] for key, value in sorted_evaluation.items()]
      current_best_error = list_of_sorted_evaluation[0][1]

   # Best specimen is trained
   # Read testing dataset
   with open(train_data, "r") as file:
      data = file.readlines()
      data = [line.strip("\n") for line in data]

   # Construct lists of variable columns
   variables = data[0].split(",")
   columns = [[] for nun in range(len(variables))]
   for row in data[1:]:
      values = row.split(",")
      for i, value in enumerate(values):
         columns[i].append(float(value))
   goal_variable_column = columns[-1]
   input_columns = columns[:-1]

   # Run best model with test data
   winner = list_of_sorted_evaluation[0][0]
   computed_values = winner.calculate_nn_outputs(input_columns, num_layers)
   test_error = calculate_error(goal_variable_column, computed_values)
   print(f"[Test error]: {test_error}")
   

if __name__ == "__main__":
   main()

# py labos.py --train sine_train.txt --test sine_test.txt --nn 5s --popsize 10 --elitism 1 --p 0.1 --K 0.1 --iter 2000
# py labos.py --train rastrigin_train.txt --test rastrigin_test.txt --nn 5s --popsize 10 --elitism 1 --p 0.3 --K 0.5 --iter 2000


