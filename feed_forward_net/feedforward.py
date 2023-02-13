import numpy as np

class FFNet:
    """Fast Feed Forward Network
    """    
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        """Initialize  Feed Forward network

        Args:
            input_neurons (int): number of input neurons
            hidden_neurons (int): number of hidden neurons
            output_neurons (int): number of output neurons
        """        
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        self.ranges = np.zeros((self.input_neurons, 2))
        # Activation Functions
        self.activation_hidden = lambda x: 2/(1+np.exp(-2*x))-1
        self.activation_output = lambda x: x
        # Initialize weights
        self.IW = np.random.rand(self.hidden_neurons, self.input_neurons)
        self.LW = np.random.rand(self.output_neurons, self.hidden_neurons)
        # Initialize biases
        self.bias = np.random.rand(self.hidden_neurons, 1)
        self.bias2 = np.random.rand(self.output_neurons, 1)
    
    def evaluate(self, x_data, output):
        """Evaluate the network

        Args:
            x_data (np.array): input data
            output (np.array): actual results of the network
        Returns:
            np.array: output of the network
        """    
        # Initialize Inbetween layers
        self.V1 = np.zeros((self.hidden_neurons,x_data.shape[1]))
        self.Y1 = np.zeros((self.hidden_neurons,x_data.shape[1]))
        self.Y2 = np.zeros((self.output_neurons,x_data.shape[1]))
        self.V2 = np.zeros((self.output_neurons,x_data.shape[1]))

        # Generating input for the hidden layer
        n   = x_data.shape[1]
        self.V1  = np.dot(np.append(self.IW,self.bias,axis=1), np.append(x_data,np.ones((1,n)),axis=0))

        # Generating output of the hidden layer TANGENT SIGMOIDAL
        self.Y1  = self.activation_hidden(self.V1)
        # Generating input for the output layer
        self.V2  = np.dot(np.append(self.LW,self.bias2,axis=1), np.append(self.Y1,np.ones((1,n)),axis=0))

        # Generating output of the hidden layer LINEAR
        self.Y2  = self.activation_output(self.V2)

        MSE = np.sum((self.Y2-output)**2)/n

        return self.Y2, MSE
    
    def trainBackProp(self, x_data, output, epochs, learning_rate, goal = 1e-7, adaptive = False):
        """Train the network using backpropagation

        Args:
            x_data (array): input data
            output (array): result data
            epochs (int): itrartions for training
            learning_rate (float): learing rate
        """        
        self.activation_hidden_derivative = lambda x: 1-(2/(1+np.exp(-2*x))-1)**2 
        
        Y, previous_mse = self.evaluate(x_data, output)
        error = output - Y
        counter = 0

        for i in range(epochs):
            #Store old weights
            old_IW = self.IW
            old_LW = self.LW
            old_bias = self.bias
            old_bias2 = self.bias2
            #Calc derivatives
            dEdyk = -1
            dykdvk = error
            dvkdwjk = self.Y1

            dEdwjk = np.sum(dEdyk*dykdvk*dvkdwjk)

            dvkdyj = self.LW
            dyjdvj = self.activation_hidden_derivative(self.V1)
            dvjdwij = x_data

            dEdwij = np.sum(dEdyk*np.dot(dykdvk.T,np.dot(dvkdyj,np.dot(dyjdvj,dvjdwij.T))))

            dvkdbj = 1

            dEdbj = np.sum(dEdyk*dykdvk*dvkdbj)

            dvjdbi = 1

            dEdbi = np.sum(dEdyk*np.dot(dykdvk.T,np.dot(dvkdyj,np.dot(dyjdvj,dvjdbi))))

            #Update weights

            self.LW -= learning_rate*dEdwjk
            self.IW -= learning_rate*dEdwij
            self.bias -= learning_rate*dEdbi
            self.bias2 -= learning_rate*dEdbj

            Y, mse = self.evaluate(x_data, output)
            error = output - Y
            
            if mse < goal:
                print("Goal reached")
                break
        
            if not adaptive:
                
                if mse > previous_mse:
                    counter += 1
                    print("MSE increased")
                    if counter > 10:
                        print("MSE increased 10 times in a row, stopping training")
                        break
                    continue

                counter = 0
                previous_mse = mse

            else:

                if mse > previous_mse:
                    print("MSE increased")
                    #Revert to old weights
                    self.IW = old_IW
                    self.LW = old_LW
                    self.bias = old_bias
                    self.bias2 = old_bias2
                    #Increase learning rate
                    learning_rate *= 1.01
                    continue
                else:
                    learning_rate /= 1.001
                
                previous_mse = mse
            
    def trainLM(self, x_data, output, epochs, learning_rate, goal = 1e-7, adaptive = False):
        """Train the network using Levenberg-Marquardt

        Args:
            x_data (array): input data
            output (array): result data
            epochs (int): itrartions for training
            learning_rate (float): learing rate
        """        
        self.activation_hidden_derivative = lambda x: 1-(2/(1+np.exp(-2*x))-1)**2 
        
        Y, previous_mse = self.evaluate(x_data, output)
        error = output - Y
        counter = 0

        for i in range(epochs):
            #Store old weights
            old_IW = self.IW
            old_LW = self.LW
            old_bias = self.bias
            old_bias2 = self.bias2
            
            #Calc derivatives
            






