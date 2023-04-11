import numpy as np
from rbf_neural_net.rbf import RbfNet
from sklearn.cluster import KMeans
    
class RbfLMnet(RbfNet):
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        super().__init__(input_neurons, hidden_neurons, output_neurons)

    def trainLM(self, input, output, epochs, goal, mu,  min_grad=1e-20, adaptive = False, center_init = 'random'):
        """Train network using the Leven-Maruardt method

        Args:
            input (np.array): input data
            output (np.array): actual output data
            epochs (int): number of epochs
            goal (float): desired performance, training stops is goal reached
            min_grad (float): minimal gradient, training stops if abs gradient value drops below the value
            mu (float):  learning rate
            adaptive (bool): if true, mu is adapted during training
            center_init (str): method for initializing centers, 'random' or 'kmeans'
        """       
        # Initialize centers
        if center_init == 'random':
            self.centers = np.random.rand(self.hidden_neurons, self.input_neurons)
        elif center_init == 'kmeans':
            for i in range(self.input_neurons):
                self.centers[:,[i]] = KMeans(n_clusters=self.hidden_neurons, random_state=0, n_init = 'auto').fit(input[i,:].reshape(-1,1)).cluster_centers_
        else:
            raise ValueError("Invalid center initialization method")
        
        variables = np.concatenate((self.IW, self.LW, self.centers), axis=None)
        Y, mes_previous = self.evaluate(input, output)
        error = (Y-output).T
        mu_list = np.zeros(epochs)
        mse_list = np.zeros(epochs)

        if not adaptive:
            for i in range(epochs):
                #print("Epoch: ", i)
                #Store Old Weights

                mu_list[i] = mu

                J = self._calcjacobian(input)

                # Calculate gradient
                grad = np.dot(J.T, error)

                #Check if gradient is small enough
                if np.linalg.norm(grad) < min_grad:
                    print("Gradient is small enough")
                    break

                # Calculate Hessian
                H = np.dot(J.T, J)

                # Calculate weight update
                variables -= np.dot(np.linalg.pinv(H + mu*np.eye(H.shape[0])), grad).flatten()

                # Update weights
                self.IW = variables[:self.hidden_neurons*self.input_neurons].reshape(self.hidden_neurons, self.input_neurons)
                self.LW = variables[self.hidden_neurons*self.input_neurons:self.hidden_neurons*self.input_neurons+self.hidden_neurons*self.output_neurons].reshape(self.output_neurons, self.hidden_neurons)
                self.centers = variables[self.hidden_neurons*self.input_neurons+self.hidden_neurons*self.output_neurons:].reshape(self.hidden_neurons, self.input_neurons)

                Y, mse= self.evaluate(input, output)
                error = (Y-output).T

                if mse < goal:
                    print("Goal reached")
                    break
                if mse > mes_previous:
                    mes_previous = mse
                    break

                mes_previous = mse
        else:
            for i in range(epochs):
                #print("Epoch: ", i)
                mu_list[i] = mu

                #Store Old Weights 
                old_variables = variables

                J = self._calcjacobian(input)


                # Calculate gradient
                grad = np.dot(J.T, error)

                #Check if gradient is small enough
                if np.linalg.norm(grad) < min_grad:
                    print("Gradient is small enough")
                    break

                # Calculate Hessian
                H = np.dot(J.T, J)

                # Calculate weight update
                variables += np.dot(np.linalg.pinv(H + mu*np.eye(H.shape[0])), grad).flatten()

                # Update weights
                self.IW = variables[:self.hidden_neurons*self.input_neurons].reshape(self.hidden_neurons, self.input_neurons)
                self.LW = variables[self.hidden_neurons*self.input_neurons:self.hidden_neurons*self.input_neurons+self.hidden_neurons*self.output_neurons].reshape(self.output_neurons, self.hidden_neurons)
                self.centers = variables[self.hidden_neurons*self.input_neurons+self.hidden_neurons*self.output_neurons:].reshape(self.hidden_neurons, self.input_neurons)

                Y, mse = self.evaluate(input, output)
                error = (Y-output).T
                mse_list[i] = mse
                if mu >1e20:
                    print("Mu too high")
                    break
                if mse < goal:
                    print("Goal reached")
                    break
                if mse > mes_previous:
                    # print("MSE increased")
                    mes_previous = mse
                    mu *= 10
                    # Revert to old weights
                    self.IW = old_variables[:self.hidden_neurons*self.input_neurons].reshape(self.hidden_neurons, self.input_neurons)
                    self.LW = old_variables[self.hidden_neurons*self.input_neurons:self.hidden_neurons*self.input_neurons+self.hidden_neurons*self.output_neurons].reshape(self.output_neurons, self.hidden_neurons)
                    self.centers = old_variables[self.hidden_neurons*self.input_neurons+self.hidden_neurons*self.output_neurons:].reshape(self.hidden_neurons, self.input_neurons)
                    continue
                mes_previous = mse
                mu /= 1.01

        return mu_list, mse_list

    def _calcjacobian(self, input):
        """Calculates the Jacobian matrix for the RBF network

        Args:
            input (array): input data
            output (array): output of first hidden layer

        Returns:
            matirx: Jacobian matrix
        """        
        n_samples = input.shape[1]
        n_inputs = input.shape[0]
        n_neurons = self.IW.shape[0]
        
        dedyk = -1
        dykdvk = 1
        dvkdwjk = self.Y1
        dedwjk = dedyk * dykdvk * dvkdwjk # output weights update
        dvkdyj = self.LW 
        dyjdvj = -self.Y1 # -exp(vj)
        dedwij = np.zeros((n_samples, n_neurons, n_inputs)) # input weights update
        dedcij = np.zeros((n_samples, n_neurons, n_inputs))

        for i in range(n_inputs):
            dvjdwij = 2 *self.IW[:,i].T*((input[[i],:]-self.centers[:,[i]]).T)**2
            dvjdcij = -2*self.IW[:,[i]].T**2*(input[[i],:]-self.centers[:,[i]]).T
            
            dedwij[:,:,i] = dedyk * dykdvk * dvkdyj * dyjdvj.T * dvjdwij
            dedcij[:,:,i] = dedyk * dykdvk * dvkdyj * dyjdvj.T * dvjdcij
        

        
        # Shape jacobian
        J = np.concatenate([np.reshape(dedwij,(n_samples,n_neurons*n_inputs)), dedwjk.T, np.reshape(dedcij, (n_samples, n_neurons*n_inputs))], axis=1)
        return J
    



