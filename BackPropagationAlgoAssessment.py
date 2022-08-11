import math as m
import matplotlib.pyplot as plt

# initialize network as list of layers, and each layer as array of neurons with given weights 
network = list()
# last weight for each neuron is the bias 
hidden_layer = [{'weights':[0.74, 0.8, 0.35, 0.9]}, {'weights': [0.13, 0.4, 0.97, 0.45]}, 
{'weights': [0.68, 0.1, 0.96, 0.36]}]
network.append(hidden_layer)
output_layer = [{'weights': [0.35, 0.5, 0.9, 0.98]}, {'weights': [0.8, 0.13, 0.8, 0.92]}]
network.append(output_layer)

# activation = sum(weight_i * input_i) + bias

# calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1] # bias is last weight in weights array 
    for i in range(len(weights)-1): # in range of weights not including bias 
        activation += weights[i] * inputs[i]
    return activation

# calculate output using sigmoid function
# output = 1 / 1 + exp^(-activation)

# transfer neuron activation
def transfer(activation):
    return 1.0/(1.0 + m.exp(-activation))

def log(out):
    return -(m.log((1/out)-1))


# forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row # initial input is row of training data
    for layer in network:
        new_inputs = [] # empty array for new inputs 
        for neuron in layer: 
            activation = activate(neuron['weights'], inputs) # calculates each neuron's activation
            neuron['output'] = transfer(activation) # calculates each neurons output
            new_inputs.append(neuron['output']) # appends outputs to array 
        inputs = new_inputs # outputs from previous layer become inputs to next layer 
    #out1 = log(inputs[0])
    #out2 = log(inputs[1])
    #out = [out1, out2]
    return inputs

# test forward propagation
row = [0.3, 0.5, 0.75]
output = forward_propagate(network, row)
print(output)
print('forward')
print(network)

# derivative = output * (1.0-output)
# calculate the derivative of a neuron output 
def transfer_derivative(output):
    return output * (1-output)

# backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))): # starts from the output layer anmd works backwards 
        layer = network[i] # represents each layer in the network
        errors = list()
        if i != len(network)-1: # if not output layer 
            for j in range(len(layer)): # iterates through each neuron in layer 
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j] * neuron['delta']) # calculates the hidden layer error 
                errors.append(error) # adds the errors to a list 
                
        else:
            for j in range(len(layer)): 
                neuron = layer[j] # represents each neuron in the layer 
                errors.append(neuron['output'] - expected[j]) # adds the result of the output error to erros list 
        for j in range(len(layer)):
            #neuron = layer[j]
            layer[j]['delta'] = errors[j] * transfer_derivative(neuron['output']) # calculates the delta error for each neuron 
    #return errors 

def update_weights(network, row, learning_rate): 
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] -= learning_rate * neuron['delta'] * inputs[j] # subtracts the errors from each weight 
            neuron['weights'][-1] -= learning_rate * neuron['delta'] 

def steps(epochs): # adds epoch values to an array 
    steps = []
    for i in range(epochs):
        steps.append(i+1)
    return steps 

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, epochs):
    sum_array = []
    for epoch in range(epochs): # iterate through specified number of epochs 
        error = 0
        for row in train: # for each setof training data 
            outputs = forward_propagate(network, row) # propagtes outputs training data
            #print(outputs)
            expected = row[-2:] # target outputs are last two values in row of data
            backward_propagate_error(network, expected) # calculate errors of each neuron
            update_weights(network, row, l_rate) # update weights in network 
            error += sum([(expected[i]-outputs[i]) for i in range(len(expected))]) # total mean squared sum error
            #error = error
            error = (error**2)*0.5 # squares the summed errors and divides by the number of outputs 
            
        sum_array.append(error) # adds errors to an array after each epoch
        print(network)
    return sum_array   
      
def softmax(out1, out2, out = True): # calculates probability distribution of outputs 
    if out == True:
        return m.exp(out1)/(m.exp(out1) + m.exp(out2)) # calcualtes for outout y1
    else:
        return m.exp(out2)/(m.exp(out2) + m.exp(out1)) # calculates for output y2

dataset = [[0.5, 1.0, 0.75, 1, 0],
[1.0, 0.5, 0.75, 1, 0],
[1.0, 1.0, 1.0, 1, 0],
[-0.01, 0.5, 0.25, 0, 1],
[0.5, -0.25, 0.13, 0, 1],
[0.01, 0.02, 0.05, 0, 1]]

sums = train_network(network, dataset, 0.1, 100)
step = steps(100)

plt.plot(step, sums)
plt.xlabel('Epoch')
plt.ylabel('Squared Error')
plt.title('Learning Curve')
plt.show()

row = [0.3, 0.7, 0.9]
outputs = forward_propagate(network, row)
print(outputs)
out1 = outputs[0]
out2 = outputs[1]
prob1 = softmax(out1, out2, True)
prob2 = softmax(out1, out2, False)
print(prob1, prob2)

# make a prediction with a network
def predict(network, row): # takes network and and input parameters 
    outputs = forward_propagate(network, row) # calculates output 
    for i in range(len(outputs)): # iterates through two output values 
        if outputs[i] >= 0.5: # rounds up the output to 1 
            outputs[i] = 1
        else:
            outputs[i] = 0 # if output is less than 0.5 it rounds down 
    return outputs
    #print(network)
    
for row in dataset: # for each row in dataset 
    prediction = predict(network, row) 
    print("Expected = %s, Got = %s" % (row[-2:], prediction)) # print expected outputs and actual outputs comparison




