import numpy as np
from sklearn.cluster import KMeans

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
        self.IW = np.random.rand(self.hidden_neurons, self.input_neurons)- 0.5
        self.LW = np.random.rand(self.output_neurons, self.hidden_neurons) - 0.5
        # Initialize biases
        self.bias = np.random.rand(self.hidden_neurons, 1)
        self.bias2 = np.zeros((self.output_neurons, 1))
    
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

        # Generating output of the output layer LINEAR
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
        self.bias2 = KMeans(n_clusters=self.output_neurons).fit(output.T.reshape(-1, 1)).cluster_centers_.T
        Y, previous_mse = self.evaluate(x_data, output)
        error = output - Y
        counter = 0
        mse_list = np.zeros(epochs)

        for i in range(epochs):
            #Store old weights
            old_IW = self.IW
            old_LW = self.LW
            old_bias = self.bias
            old_bias2 = self.bias2
            #Calc derivatives

            dEdyk = (2*error/x_data.shape[1])
            dEdwjk = np.dot(dEdyk,self.Y1.T)

            delta = np.dot(dEdyk.T, self.LW) * (1 - self.Y1**2).T
            dEdwij = np.dot(delta.T, x_data.T)

            dEdbj = np.sum(dEdyk, axis=1).reshape(self.output_neurons,1)
            dEdbi = np.sum(delta, axis=0).reshape(self.hidden_neurons,1)

            #Update weights

            self.LW -= learning_rate*dEdwjk
            self.IW -= learning_rate*dEdwij
            self.bias -= learning_rate*dEdbi
            self.bias2 -= learning_rate*dEdbj

            Y, mse = self.evaluate(x_data, output)
            error = Y- output
            mse_list[i] = mse
            if mse < goal:
                print("Goal reached")
                break
            if learning_rate > 1e10:
                print("Learning rate too high")
                break
            if not adaptive:
                
                if mse > previous_mse:
                    counter += 1
                    if counter > 10:
                        print("MSE increased 10 times in a row, stopping training")
                        break
                    continue

                counter = 0
                previous_mse = mse

            else:

                if mse > previous_mse:

                    previous_mse = mse
                    counter += 1
                    learning_rate *= 10
                    continue
                else:
                    learning_rate /= 1.1
                
                previous_mse = mse
        return mse_list
            
    def trainLM(self, x_data, output, epochs, learning_rate, goal = 1e-7, adaptive = False, min_grad = 1e-10):
        """Train the network using Levenberg-Marquardt

        Args:
            x_data (array): input data
            output (array): result data
            epochs (int): itrartions for training
            learning_rate (float): learing rate
        """        
        self.bias2 = KMeans(n_clusters=self.output_neurons).fit(output.T.reshape(-1, 1)).cluster_centers_.T
        variables = np.concatenate((self.IW, self.LW,self.bias,self.bias2), axis=None)
        Y, previous_mse = self.evaluate(x_data, output)
        error = output - Y
        counter = 0
        mse_list = np.zeros(epochs)

        for i in range(epochs):
            #Store old weights
            old_variables = variables

            #Calculate Jacobian
            J = self._calcJacobian(x_data)


            #Calculate Gradient
            grad = np.dot(J.T,error.T)

            #Check if gradient is small enough
            if np.linalg.norm(grad) < min_grad:
                print("Gradient is small enough")
                break


            #Calculate Hessian
            H = np.dot(J.T,J)

            #Calculate update variable
            variables -= np.dot(np.linalg.pinv(H+learning_rate*np.identity(H.shape[0])),grad).flatten()

            #Update weights
            self.IW = variables[:self.input_neurons*self.hidden_neurons].reshape(self.hidden_neurons,self.input_neurons)
            self.LW = variables[self.input_neurons*self.hidden_neurons:(self.input_neurons+self.output_neurons)*self.hidden_neurons].reshape(self.output_neurons,self.hidden_neurons)
            self.bias = variables[(self.input_neurons+self.output_neurons)*self.hidden_neurons:(self.input_neurons+self.output_neurons+1)*self.hidden_neurons].reshape(self.hidden_neurons,1)
            self.bias2 = variables[(self.input_neurons+self.output_neurons+1)*self.hidden_neurons:].reshape(self.output_neurons,1)

            Y, mse = self.evaluate(x_data, output)
            error = output - Y
            mse_list[i] = mse
            if learning_rate >1e20:
                print("Mu too high")
                break
            if mse < goal:
                    print("Goal reached")
                    break
            if not adaptive:

                if mse > previous_mse:

                        previous_mse = mse
                        break
                previous_mse = mse
            else:
                if mse > previous_mse:
                    #Revert to old weights
                    variables = old_variables
                    #Increase learning rate
                    learning_rate *= 3
                    continue
                else:
                    learning_rate /= 1.5
                
                previous_mse = mse
        return mse_list


    def _calcJacobian(self, input_data):
                    
        n_samples = input_data.shape[1]
        n_inputs = input_data.shape[0]
        n_neurons = self.IW.shape[0]
        #Calc derivatives
        dedyk = -1
        dykdvk = 1
        dvkdwjk = self.Y1

        dedwjk = dedyk*dykdvk*dvkdwjk

        dvkdyj = self.LW
        dyjdvj = 1 - self.Y1**2

        dedwij = np.zeros((n_samples, n_neurons, n_inputs))

        for i in range(input_data.shape[0]):
            dvjdwij = np.ones(self.IW[:,[i]].shape)*input_data[[i],:]

            dedwij[:,:,i] =dedyk * dykdvk *( dvkdyj * (dyjdvj.T * dvjdwij.T))


        dvkdbj = np.ones((n_samples, self.output_neurons))

        dedbj = dedyk*dykdvk*dvkdbj

        dvjdbi = np.ones((n_samples, n_neurons))

        dedbi = dedyk*dykdvk*dvkdyj*dyjdvj.T*dvjdbi

        J = np.concatenate([np.reshape(dedwij,(n_samples,n_neurons*n_inputs)),dedwjk.T,dedbi,dedbj], axis=1)
        return J







