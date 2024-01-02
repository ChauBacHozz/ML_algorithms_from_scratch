import numpy as np
import pickle
def ReLU(input):
    return np.maximum(input, 0)
def leaky_ReLU(input):
    return np.maximum(0.03 * input, input)
def Linear(input):
    return input
def Sigmoid(input):
    return 1 / (1 + np.exp(-input))
def Swish(input):
    return input * Sigmoid(input)

def ReLU_de(input):
    return (input > 0) * 1
def leaky_ReLU_de(input):
    return np.where(input > 0, 1, 0.03) 
def Linear_de(input):
    return 1
def Sigmoid_de(input):
    return np.exp(input)/ np.square(np.exp(input) + 1)
def Swish_de(input):
    return Swish(input) + Sigmoid(input)*(1 - Swish(input))
def MSE(predict, target):
    loss = np.square(target - predict)
    m = target.shape[1]
    cost = 1 / m * np.sum(loss)
    return np.squeeze(cost)
def MSEGrad(target, predict):
    delta = -2 * (target - predict)
    return delta
def deriative_func(func):
    if func.__name__ == "ReLU":
        return (ReLU_de)
    if func.__name__ == "leaky_ReLU":
        return (leaky_ReLU_de)
    if func.__name__ == "Linear":
        return (Linear_de)
    if func.__name__ == "Sigmoid":
        return (Sigmoid_de)
    if func.__name__ == "Swish":
        return (Swish_de)

class Layer:
    def __init__(self, row, col, act_func, isoutputlayer = False):
        self.isoutputlayer = isoutputlayer
        self.left_layer_output = None
        self.weight = 0.01 * np.random.randn(col, row)
        # self.weight = np.random.randn(col,row) * np.sqrt(2/row)
        self.bias = np.zeros((col, 1))
        self.dW = None
        self.db = None
        self.activationForward = act_func
        self.activationBackward = deriative_func(act_func)
        self.input = None
        self.d_input = None
        self.output = None
        
    def forward(self, input_data):
        self.left_layer_output = input_data
        self.input =  np.dot(self.weight, input_data) + self.bias
        self.output = self.activationForward(self.input)
        return self.output
    
    def backward(self, d_input):
        m = self.left_layer_output.shape[1]
        self.dW = 1 / m * np.dot(d_input, self.left_layer_output.T)
        self.db = 1 / m * np.sum(d_input, axis=1, keepdims=True)
        delta = np.dot(self.weight.T, d_input)
        return delta

    def update(self, learning_rate):
        self.weight -= learning_rate * self.dW  
        self.bias -= learning_rate * self.db 

class Neural_Network:
    def __init__(self, n_nodes_list, hiddenlayers_actfunc, saved_weight = None, saved_bias = None):
        self.pred_his = []
        self.layers = []
        for i in range (len(n_nodes_list) - 1):
            self.layers.append(Layer(row = n_nodes_list[i], col=n_nodes_list[i + 1], act_func=hiddenlayers_actfunc))
        if (saved_weight != None) and (saved_bias != None):
            for i in range (len(n_nodes_list) - 1):
                self.layers[i].weight = saved_weight[i]
                self.layers[i].bias = saved_bias[i]
        self.layers[-1].activationForward = Linear
        self.layers[-1].activationBackward = Linear_de
        self.cost_his = []
        self.test_cost_his = []
    def weight2_sum(self):
        sum = 0
        for layer in self.layers:
            sum += np.sum(np.square(layer.bias))
        return sum
    def forward(self, input):
        x = np.copy(input)
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def backward(self, target, predict, alpha):
        delta = MSEGrad(target, predict) + alpha * self.weight2_sum();
        for layer in reversed(self.layers):
            d_input = delta * layer.activationBackward(layer.input)
            delta = layer.backward(d_input)
    def update(self, learning_rate = 0.01):
        for layer in self.layers:
            layer.update(learning_rate)
    def fit(self, input_matrix, target, X_test, y_test, learning_rate = 0.001, alpha = 0.00001, epochs = 1000 ,lr_down = True, lr_decay = 50):
        fit_range = [1,2,3,4,5,6,7,8,9,10]
        for i in range (epochs):
            predict = self.forward(input_matrix)
            self.backward(target, predict, alpha)
            self.update(learning_rate= learning_rate)
            if lr_down:
                if i % 1000 == 0:
                    learning_rate -= learning_rate / lr_decay
            if (i + 1) % 100 == 0:
                self.pred_his.append(predict)
                self.cost_his.append(MSE(predict, target)) 
                self.test_cost_his.append(MSE(self.forward(X_test), y_test))
            if (i + 1) * 10 / (epochs) in fit_range:
                print(f"Loading {(i + 1) * 100 / (epochs)}%")
        print("Learning process completed!!!")
    def save_model(self):
        w_list = []
        b_list = []
        for layer in self.layers:
            w_list.append(layer.weight)
            b_list.append(layer.bias)
        pickle.dump(w_list , open('weights.pkl' , 'wb' ) )
        pickle.dump(b_list , open('bias.pkl' , 'wb' ) )