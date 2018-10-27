class TwoLayerMultilayerPerceptron:
    ## Initializing the class of the Multilayer Perceptron
    def __init__(self, inputs, neurons_hidden_layer1, neurons_hidden_layer2, learning_rate, outputs, epochs, init_method = 'random', activation_method = 'sigmoid', error_metric = 'least_squares'):
        self.neurons_input_layer = inputs
        self.neurons_hidden_layer1 = neurons_hidden_layer1
        self.neurons_hidden_layer2 = neurons_hidden_layer2
        self.neurons_output_layer = outputs
        self.learning_rate = learning_rate
        self.init_method = init_method
        self.activation_method = activation_method
        self.weights_hidden_layer1 = self.weights_initialization(self.neurons_input_layer, self.neurons_hidden_layer1)
        self.weights_hidden_layer2 = self.weights_initialization(self.neurons_hidden_layer1, self.neurons_hidden_layer2)
        self.weights_output_layer = self.weights_initialization(self.neurons_hidden_layer2, self.neurons_output_layer)
        self.bias_hidden_layer1 = self.weights_initialization(1, self.neurons_hidden_layer1)
        self.bias_hidden_layer2 = self.weights_initialization(1, self.neurons_hidden_layer2)
        self.bias_output_layer = self.weights_initialization(1, self.neurons_output_layer)
        self.epochs = epochs
        self.error_metric = error_metric
        self.training_accuracy = []
        
    # function that computes for the initialization of weights
    def weights_initialization(self, inputs, neurons):
        # randomly initializes weights to small values
        if self.init_method == 'random':
            weights = 0.1 * (np.random.rand(inputs, neurons))
        return weights
        
    # function that computes for the summation of the products of the weights and the input
    def net_internal_activity(self, vector1, vector2):
        return np.dot(vector1, vector2)
    
    # function that computes for the activation function 
    def activation_function(self, vector):
        if self.activation_method == 'sigmoid':
            return 1.0/(1.0 + np.exp(-1.0 * vector))
        
    def derivative_activation_function(self, vector):
        if self.activation_method == 'sigmoid':
            return ((vector) * (1 - vector))
        
    # function that computes the cost function for the predicted and true labels
    def cost_function(self, predicted, true):
        cost = 0
        if self.error_metric == 'least_squares':
            difference = predicted - true
            cost = ((difference * difference).sum())/2.0
        elif self.error_metric == 'cross_entropy': 
            cost = -1 * (np.sum(((true * np.log(predicted)) + ((1 - true) * np.log(1- predicted)))))
        return cost
        
    def shuffle_dataset(self, x_train, y_train):
        assert len(x_train) == len(y_train)
        p = np.random.permutation(len(x_train))
        return x_train[p], y_train[p]

    # function that trains the algorithm using backpropagation
    def train(self, x_train, y_train):
        cost_per_epoch = np.zeros(self.epochs)
        for epoch in range(0,self.epochs):
            correct_classification = 0.0
            cost_per_x = np.zeros(len(x_train))
            x_train, y_train = self.shuffle_dataset(x_train, y_train)
            for i in range(0, len(x_train)):
                #### forward pass for ith input
                # compute the net internal activities for the input layer and hidden layer 1
                net_internal_activity_hidden_layer1 = self.net_internal_activity(x_train[i], self.weights_hidden_layer1) + self.bias_hidden_layer1
                # activating the net internal activities of hidden layer 1
                activated_hidden_layer1 = self.activation_function(net_internal_activity_hidden_layer1)
                
                # compute the net internal activities for the hidden layer 1 and hidden layer 2
                net_internal_activity_hidden_layer2 = self.net_internal_activity(activated_hidden_layer1, self.weights_hidden_layer2) + self.bias_hidden_layer2
                # activating the net internal activities of hidden layer 2
                activated_hidden_layer2 = self.activation_function(net_internal_activity_hidden_layer2)
                
                # compute the net internal activities for the hidden layer 2 and output layer
                net_internal_activity_output_layer = self.net_internal_activity(activated_hidden_layer2, self.weights_output_layer)  + self.bias_output_layer
                # activating the net internal activities of the output layer
                activated_output_layer = self.activation_function(net_internal_activity_output_layer)
                
                if np.argmax(activated_output_layer) == np.argmax(y_train[i]):
                    correct_classification = correct_classification + 1
                
                # get the error for the ith prediction
                error = activated_output_layer - y_train[i]
                
                # compute the cost for the ith prediction
                cost = self.cost_function(activated_output_layer, y_train[i])
                
                #### backward computations
                # computing the delta for the output layer
                delta_output_layer = error * self.derivative_activation_function(activated_output_layer)
                    
                # computing the delta for hidden layer 2
                delta_hidden_layer2 = np.dot(delta_output_layer, self.weights_output_layer.T) * self.derivative_activation_function(activated_hidden_layer2)
                # computing the delta for hidden layer 1
                delta_hidden_layer1 = np.dot(delta_hidden_layer2, self.weights_hidden_layer2.T) * self.derivative_activation_function(activated_hidden_layer1)
                
                #### weight updates
                # update weights in output layer
                self.weights_output_layer = self.weights_output_layer + ((-1.0 * self.learning_rate) * (np.dot(delta_output_layer.T, activated_hidden_layer2))).T
                # update weights in hidden layer 2
                self.weights_hidden_layer2 = self.weights_hidden_layer2 + ((-1.0 * self.learning_rate) * (np.dot(delta_hidden_layer2.T, activated_hidden_layer1))).T
                # update weights in hidden layer 1
                self.weights_hidden_layer1 = self.weights_hidden_layer1 + ((-1.0 * self.learning_rate) * (np.dot(delta_hidden_layer1.T, x_train[i].reshape(1,len(x_train[i]))))).T
                
                ### bias updates
                # update bias in the output layer
                self.bias_output_layer = self.bias_output_layer + (self.learning_rate * delta_output_layer)
                # update bias in hidden layer 2
                self.bias_hidden_layer2 = self.bias_hidden_layer2 + (self.learning_rate * delta_hidden_layer2)
                # update bias in hidden layer 1
                self.bias_hidden_layer1 = self.bias_hidden_layer1 + (self.learning_rate * delta_hidden_layer1)
                
                cost_per_x[i] = cost
            cost_per_epoch[epoch] = cost_per_x.sum()
            accuracy = (correct_classification / len(x_train))
            self.training_accuracy.append(accuracy)
            print("Epoch #" + str(epoch + 1) + " | Accuracy: " + str(accuracy) + " | Cost: " + str(cost_per_epoch[epoch]))
            
    # function that trains the algorithm using backpropagation
    # and predicts the test set per epoch to observe the performance of the model
    def train_predict(self, x_train, y_train, x_test, y_test):
        test_accuracies = []
        cost_per_epoch = np.zeros(self.epochs)
        for epoch in range(0,self.epochs):
            correct_classification = 0.0
            cost_per_x = np.zeros(len(x_train))
            x_train, y_train = self.shuffle_dataset(x_train, y_train)
            for i in range(0, len(x_train)):
                #### forward pass for ith input
                # compute the net internal activities for the input layer and hidden layer 1
                net_internal_activity_hidden_layer1 = self.net_internal_activity(x_train[i], self.weights_hidden_layer1) + self.bias_hidden_layer1
                # activating the net internal activities of hidden layer 1
                activated_hidden_layer1 = self.activation_function(net_internal_activity_hidden_layer1)
                
                # compute the net internal activities for the hidden layer 1 and hidden layer 2
                net_internal_activity_hidden_layer2 = self.net_internal_activity(activated_hidden_layer1, self.weights_hidden_layer2) + self.bias_hidden_layer2
                # activating the net internal activities of hidden layer 2
                activated_hidden_layer2 = self.activation_function(net_internal_activity_hidden_layer2)
                
                # compute the net internal activities for the hidden layer 2 and output layer
                net_internal_activity_output_layer = self.net_internal_activity(activated_hidden_layer2, self.weights_output_layer)  + self.bias_output_layer
                # activating the net internal activities of the output layer
                activated_output_layer = self.activation_function(net_internal_activity_output_layer)
                
                if np.argmax(activated_output_layer) == np.argmax(y_train[i]):
                    correct_classification = correct_classification + 1
                
                # get the error for the ith prediction
                error = activated_output_layer - y_train[i]
                
                # compute the cost for the ith prediction
                cost = self.cost_function(activated_output_layer, y_train[i])
                
                #### backward computations
                # computing the delta for the output layer
                delta_output_layer = error * self.derivative_activation_function(activated_output_layer)
                    
                # computing the delta for hidden layer 2
                delta_hidden_layer2 = np.dot(delta_output_layer, self.weights_output_layer.T) * self.derivative_activation_function(activated_hidden_layer2)
                # computing the delta for hidden layer 1
                delta_hidden_layer1 = np.dot(delta_hidden_layer2, self.weights_hidden_layer2.T) * self.derivative_activation_function(activated_hidden_layer1)
                
                #### weight updates
                # update weights in output layer
                self.weights_output_layer = self.weights_output_layer + ((-1.0 * self.learning_rate) * (np.dot(delta_output_layer.T, activated_hidden_layer2))).T
                # update weights in hidden layer 2
                self.weights_hidden_layer2 = self.weights_hidden_layer2 + ((-1.0 * self.learning_rate) * (np.dot(delta_hidden_layer2.T, activated_hidden_layer1))).T
                # update weights in hidden layer 1
                self.weights_hidden_layer1 = self.weights_hidden_layer1 + ((-1.0 * self.learning_rate) * (np.dot(delta_hidden_layer1.T, x_train[i].reshape(1,len(x_train[i]))))).T
                
                ### bias updates
                # update bias in the output layer
                self.bias_output_layer = self.bias_output_layer + (self.learning_rate * delta_output_layer)
                # update bias in hidden layer 2
                self.bias_hidden_layer2 = self.bias_hidden_layer2 + (self.learning_rate * delta_hidden_layer2)
                # update bias in hidden layer 1
                self.bias_hidden_layer1 = self.bias_hidden_layer1 + (self.learning_rate * delta_hidden_layer1)
                
                cost_per_x[i] = cost
            cost_per_epoch[epoch] = cost_per_x.sum()
            # testing the algorithm on the validation set
            test_predictions = []
            for i in data.values[2440:]:
                test_predictions.append(self.predict(i))    
            test_accuracies.append(self.compute_accuracy(test_predictions, y_test))
            
            accuracy = (correct_classification / len(x_train))
            self.training_accuracy.append(accuracy)
            print("Epoch #" + str(epoch + 1) + " | Accuracy: " + str(accuracy) + " | Cost: " + str(cost_per_epoch[epoch]))
            
        return test_accuracies
    
    def predict(self, x):
        #### forward pass for ith input
        # compute the net internal activities for the input layer and hidden layer 1
        net_internal_activity_hidden_layer1 = self.net_internal_activity(x, self.weights_hidden_layer1) + self.bias_hidden_layer1
        # activating the net internal activities of hidden layer 1
        activated_hidden_layer1 = self.activation_function(net_internal_activity_hidden_layer1)

        # compute the net internal activities for the hidden layer 1 and hidden layer 2
        net_internal_activity_hidden_layer2 = self.net_internal_activity(activated_hidden_layer1, self.weights_hidden_layer2) + self.bias_hidden_layer2
        # activating the net internal activities of hidden layer 2
        activated_hidden_layer2 = self.activation_function(net_internal_activity_hidden_layer2)

        # compute the net internal activities for the hidden layer 2 and output layer
        net_internal_activity_output_layer = self.net_internal_activity(activated_hidden_layer2, self.weights_output_layer)  + self.bias_output_layer
        # activating the net internal activities of the output layer
        activated_output_layer = self.activation_function(net_internal_activity_output_layer)
        
        return np.argmax(activated_output_layer) + 1
    
    def compute_accuracy(self, predicted, true):
        correct = 0.0
        for i in range(0, len(predicted)):
            if predicted[i] == true[i]:
                correct = correct + 1
            
        return correct/len(predicted)