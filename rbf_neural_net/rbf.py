import numpy as np
from sklearn.cluster import KMeans

class RbfNet():
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons

        # Initialize weights
        self.IW = np.random.randn(self.hidden_neurons, self.input_neurons)
        self.LW = np.random.randn(self.output_neurons, self.hidden_neurons)

        # Initialize other parameters
        self.ranges = np.zeros((self.input_neurons, 2))
        self.centers = np.zeros((self.hidden_neurons, self.input_neurons))

    


    def trainLR(self, x_data, output):
        #Train the network using linear regression
        
        
        # Initialize centers
        for i in range(self.input_neurons):
            self.centers[:,[i]] = KMeans(n_clusters=self.hidden_neurons, random_state=0, n_init = 'auto').fit(x_data[i,:].reshape(-1,1)).cluster_centers_


        # Generate x_data for hidden layer
        Nin = x_data.shape[0]
        L_end = x_data.shape[1]
        V1 = np.zeros((self.hidden_neurons,L_end))
        V1 = np.zeros((self.hidden_neurons,L_end))
        for i in range(self.input_neurons):
            V1 += (self.IW[:,[i]]*x_data[[i],:]-self.IW[:,[i]]*self.centers[:,[i]]*np.ones((1,L_end)))**2
        
        # Generate output of hidden layer
        A = np.exp(-V1)
        
        # Generate output for output layer
        self.LW = np.dot(np.linalg.pinv(A).T,output.T)

        return self.LW

    def evaluate(self, x_data, output):
        # Calculate the output of the network
        # x_data: (input_size, 1)
        # output: (output_size, 1)
        
        # Generate x_data for hidden layer
        Nin = x_data.shape[0]
        L_end = x_data.shape[1]
        Nhidden = self.centers.shape[0]
        # Initialize Inbetween layers
        self.V1 = np.zeros((Nhidden,L_end))
        self.Y1 = np.zeros((Nhidden,L_end))
        self.Y2 = np.zeros((self.output_neurons,L_end))

        for i in range(Nin):
            self.V1 += (self.IW[:,[i]]*x_data[[i],:]-self.IW[:,[i]]*self.centers[:,[i]]*np.ones((1,L_end)))**2
        
        # Generate output of hidden layer
        self.Y1 = np.exp(-self.V1)
        
        # Generate output for output layer
        self.Y2 = np.dot(self.LW,self.Y1)

        # Calculate error
        error = output - self.Y2
        mse = np.sum(error**2)/L_end
        
        return self.Y2, mse