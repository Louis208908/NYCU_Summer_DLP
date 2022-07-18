import secrets
import numpy as np
import secrets
import matplotlib.pyplot as plt
# np.random.seed(secrets.randbelow(1_000_000_000))

# class report_helper:
#     def __init__(self):
#         self.__ = "Hi, 這裡是個彩蛋";


def print_data_linear(x, y, type):
    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    if type == "groud":
        plt.title("Ground truth", fontsize=18)
    else:
        plt.title("Predictions", fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

def print_loss_curve_linear(epochs, losses):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curve", fontsize=18)
    for i in range(len(epochs)):
        if i % 50 == 0:
            plt.plot(epochs[i], losses[i], "ro")

def print_accuracy_linear(epochs, accuracies):
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy curve", fontsize=18)
    for i in range(len(epochs)):
        if i % 50 == 0:
            plt.plot(epochs[i], accuracies[i], "ro")

def print_data_xor(x, y, type):
    plt.xlabel("x[0]")
    plt.ylabel("x[1]")
    if type == "groud":
        plt.title("Ground truth", fontsize=18)
    else:
        plt.title("Predictions", fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")

def print_loss_curve_xor(epochs, losses):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning curve", fontsize=18)
    for i in range(len(epochs)):
        if i % 50 == 0:
            plt.plot(epochs[i], losses[i], "ro")

def print_accuracy_xor(epochs, accuracies):
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy curve", fontsize=18)
    for i in range(len(epochs)):
        if i % 50 == 0:
            plt.plot(epochs[i], accuracies[i], "ro")

def show_result(input_x, input_y, prediction, epoch, loss, accuracy, type):
    plt.figure(figsize=(15, 12), label=type.upper())
    plt.suptitle(type.upper(), fontsize=24, y=1)
    plt.subplot(2, 2, 1)
    print_data_linear(input_x, input_y, "groud")
    plt.subplot(2, 2, 2)
    print_data_linear(input_x, prediction[-1], "prediction")
    plt.subplot(2, 2, 3)
    print_loss_curve_linear(epoch, loss)
    plt.subplot(2, 2, 4)
    print_accuracy_linear(epoch, accuracy)

class data_loader:
    def __init__(self, type):
        self.type = type
        if type == "linear":
            self.generate_fcn = self.generate_linear
        elif type == "xor":
            self.generate_fcn = self.generate_XOR_easy

    def generate_linear(self, n=100):

        pts = np.random.uniform(0, 1, (n, 2))
        inputs = []
        labels = []
        for pt in pts:
            inputs.append([pt[0], pt[1]])
            distance = (pt[0] - pt[1]) / 1.414
            if pt[0] > pt[1]:
                labels.append(0)
            else:
                labels.append(1)
        return np.array(inputs), np.array(labels).reshape(n, 1)

    def generate_XOR_easy(self):

        inputs = []
        labels = []
        for i in range(11):
            inputs.append([0.1 * i, 0.1 * i])
            labels.append(0)
            if 0.1 * i == 0.5:
                continue
            inputs.append([0.1 * i, 1 - 0.1 * i])
            labels.append(1)

        return np.array(inputs), np.array(labels).reshape(21, 1)

class layers:
    def __init__(self, input_dim, output_dim, layer_name, act_fcn_type, data_type):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_name = layer_name
        self.activation_fcn = activations(act_fcn_type)

        self.weight_matrix = np.random.randn(self.input_dim, self.output_dim)
        self.self_bias = np.random.randn(1, self.output_dim)
        #z = weight*input + bias
        '''     show layer parameters( weight & bias )
        # print("Constructing layer: " + layer_name);
        # self.show_weight();
        # self.show_bias();
        '''

        self.gradient_matrix = np.zeros((self.input_dim, self.output_dim))
        self.gradient_bias = np.zeros((1, self.output_dim))
        self.previous_gradient_matrix = np.zeros(
            (self.input_dim, self.output_dim))
        self.previous_gradient_bias = np.zeros((1, self.output_dim))
        if (act_fcn_type == "relu" or act_fcn_type == "leaky relu") and data_type == "xor":
            self.weight_matrix *= 0.1
            self.self_bias *= 0.1
        if ( act_fcn_type == "none"):
            self.weight_matrix *= 0.01
            self.self_bias *= 0.01

    def forward_output(self, input):
        self.stored_input = input
        # store the input for calculation of gradient
        return input.dot(self.weight_matrix) + self.self_bias
        #output = x。W + b

    def backward_propagate(self, loss):
        ''' print dimensions
        print(self.layer_name);
        print("Show dimensions of loss:   " + str(loss.shape));
        print("Show dimensions of input:  " + str(self.stored_input.shape));
        print("Show dimensions of weight: " + str(self.weight_matrix.shape));
        '''
        # print("Show dimensions of loss:   " + str(loss.shape));
        self.gradient_matrix = (self.stored_input.T.dot(loss))
        self.gradient_bias = loss.sum(axis=0, keepdims=True)
        return loss.dot(self.weight_matrix.T)

    def update_weight(self, learning_rate, momentum=0):
        if momentum != 0:
            momentum_gradient_matrix = momentum * self.previous_gradient_matrix + (1 - momentum) * self.gradient_matrix
            momentum_gradient_bias = momentum * self.previous_gradient_bias + (1 - momentum) * self.gradient_bias
            self.weight_matrix -= learning_rate * momentum_gradient_matrix
            self.self_bias -= learning_rate * momentum_gradient_bias
            self.previous_gradient_matrix = momentum_gradient_matrix
            self.previous_gradient_bias = momentum_gradient_bias
        else:
            self.weight_matrix -= learning_rate * self.gradient_matrix
            self.self_bias -= learning_rate * self.gradient_bias

    def show_bias(self):    # fcm to show bias mat
        print("Show bias matrix")
        print(self.self_bias)

    def show_weight(self):  # fcn to show weight mat
        print("Show weight matrix")
        print(self.weight_matrix)

class convolutional:
    def __init__(self,kernel_size, activation_type):
        self.kernel_size = kernel_size;
        self.activation_fcn = activations(activation_type);

        self.kernel_matrix = np.random.randn(self.kernel_size[0],self.kernel_size[1]);
        self.kernel_gradient_mat = np.zeros((self.kernel_size[0],self.kernel_size[1]));
        self.previous_gradient_matrix = np.zeros((self.kernel_size[0], self.kernel_size[1]));
        # print("Initialize gradient mat of conv, dim: " + str(self.kernel_gradient_mat.shape))

    def forward_output(self,input_x):
        # print(input_x)

        input_padded = np.pad(input_x,((0,0),(1,1)), 'constant');
        # print(input_padded);
        row,col = input_padded.shape;
        self.stored_for_backward = input_padded;
        
        output = np.zeros((row,col - self.kernel_size[1] + 1))

        for i in range(row):
            for j in range(col - self.kernel_size[1] + 1):
                output[i][j] = input_padded[i][j:j+self.kernel_size[1]].dot(self.kernel_matrix.T)
        return output;

    def backward_propagate(self, loss):
        input_padded = np.pad(loss, ((0, 0), (1, 1)), 'constant')
        row, col = input_padded.shape
        flipped_kernel = np.flip(self.kernel_matrix)

        computing_gradient = np.zeros((loss.shape[0],self.kernel_size[1]));
        for i in range(loss.shape[1]):
            for j in range(loss.shape[1]):
                computing_gradient[i][j] = loss[i].dot(self.stored_for_backward[i,j:j+loss.shape[1]].T)
        self.kernel_gradient_mat = np.sum(computing_gradient, axis = 0);

        output_grad_for_next_layer = np.zeros((row, col - self.kernel_size[1] + 1))

        for i in range(row):
            for j in range(col - self.kernel_size[1] + 1):
                output_grad_for_next_layer[i][j] = input_padded[i][j:j + self.kernel_size[1]].dot(self.kernel_matrix.T)
        return output_grad_for_next_layer;


    def update_weight(self, learning_rate, momentum = 0):
        if momentum != 0:
            momentum_gradient_matrix = momentum * self.previous_gradient_matrix + (1 - momentum) * self.kernel_gradient_mat
            self.kernel_matrix -= learning_rate * momentum_gradient_matrix
            self.previous_gradient_matrix = momentum_gradient_matrix
            
        else:
            self.kernel_matrix -= learning_rate * self.kernel_gradient_mat
            

class activations:
    def __init__(self, activationType):
        self.type = activationType

    def forward_activation(self, input_z):
        if self.type == "sigmoid":
            self.stored_for_backward = 1.0 / (1.0 + np.exp(-input_z))
            return self.stored_for_backward
        elif self.type == "relu":
            self.stored_for_backward = input_z
            return np.maximum(input_z, 0)
        elif self.type == "tanh":
            self.stored_for_backward = np.divide(
                np.exp(input_z) - np.exp(-input_z), (np.exp(input_z) + np.exp(-input_z)))
            return self.stored_for_backward
        elif self.type == "leaky relu":
            self.stored_for_backward = input_z
            return np.maximum(input_z, 0) + np.minimum(0.001 * input_z, 0)
        else:
            self.stored_for_backward = input_z
            return input_z

    def backward_activation(self, loss):
        if self.type == "sigmoid":
            current_gradient = np.multiply(
                self.stored_for_backward, (1.0 - self.stored_for_backward))
            return current_gradient * loss
        elif self.type == "relu":
            output = loss.copy()
            output[self.stored_for_backward < 0] = 0
            return output
        elif self.type == "tanh":
            return (1 - self.stored_for_backward ** 2) * loss
        elif self.type == "leaky relu":
            output = loss.copy()
            output[self.stored_for_backward < 0] *= 0.001
            # print(loss)
            # print(output)
            return output
        else:
            return loss

class NN:
    def __init__(self, input_dim, hidden_layer_dim, output_dim, activation_type, input_type, learning_rate, optimizer="sgd", momentum=0):

        self.conv_layer = convolutional([1,3],"sigmoid");
        self.input_layer = layers( input_dim, hidden_layer_dim, "input", activation_type, input_type);
        self.hidden_layer1 = layers( hidden_layer_dim, hidden_layer_dim, "hidden1", activation_type, input_type);

        if activation_type == "none":
            self.hidden_layer2 = layers(hidden_layer_dim, output_dim, "hidden2", "sigmoid", input_type);
        else:
            self.hidden_layer2 = layers(hidden_layer_dim, output_dim, "hidden2", activation_type, input_type);
        # self.inner_activate_fcn = activations(activation_type);
        # self.output_activation = activations(activation_type);
        self.data_loader = data_loader(input_type);
        self.learning_rate = learning_rate;
        self.optimizer = optimizer;
        if optimizer == "momentum":
            self.momentum = momentum

    def forward_propagate(self, input):  # x -> z -> a ->z -> a -> z -> output
        '''     old model design    (I've moved the activation fcn into the class of layer, which was sepearated before)
        self.output1 = self.input_layer.forward_output(input);

        self.activated_output1 = self.inner_activate_fcn.forward_activation(self.output1);

        self.output2 = self.hidden_layer1.forward_output(self.activated_output1);

        self.activated_output2 = self.inner_activate_fcn.forward_activation(self.output2);

        self.output3 = self.hidden_layer2.forward_output(self.activated_output2);

        self.activated_output3 = self.output_activation.forward_activation(self.output3);

        self.final_result = self.activated_output3;

        return self.final_result;
        '''
        
        self.output1 = self.conv_layer.forward_output(input);

        self.activated_output1 = self.conv_layer.activation_fcn.forward_activation(self.output1);
        
        self.output2 = self.input_layer.forward_output(self.activated_output1)

        self.activated_output2 = self.input_layer.activation_fcn.forward_activation(self.output2)

        self.output3 = self.hidden_layer1.forward_output(self.activated_output2)

        self.activated_output3 = self.hidden_layer1.activation_fcn.forward_activation(self.output3)

        self.output4 = self.hidden_layer2.forward_output(self.activated_output3)

        self.activated_output4 = self.hidden_layer2.activation_fcn.forward_activation(self.output4)

        self.final_result = self.activated_output4

        return self.final_result

    def backward_propagate(self, loss):
        '''     old model design  (I've moved the activation fcn into the class of layer, which was sepearated before)
        gradient1 = self.output_activation.backward_activation(loss);        # output-> z3
        gradient2 = self.hidden_layer2.backward_propagate(gradient1);   # z3 -> a2
        gradient3 = self.inner_activate_fcn.backward_activation(gradient2);   # a2 -> z2
        gradient4 = self.hidden_layer1.backward_propagate(gradient3);   # z2 -> a1
        gradient5 = self.inner_activate_fcn.backward_activation(gradient4);   # a1 -> z1
        _ = self.input_layer.backward_propagate(gradient5);             # z1 -> input
        '''

        gradient1 = self.hidden_layer2.activation_fcn.backward_activation(loss)        # output-> z3
        gradient2 = self.hidden_layer2.backward_propagate(gradient1)   # z3 -> a2
        gradient3 = self.hidden_layer1.activation_fcn.backward_activation(gradient2)   # a2 -> z2
        gradient4 = self.hidden_layer1.backward_propagate(gradient3)   # z2 -> a1
        gradient5 = self.input_layer.activation_fcn.backward_activation(gradient4)   # a1 -> z1
        gradient6 = self.input_layer.backward_propagate(gradient5)                  # z1 -> input
        gradient7 = self.conv_layer.activation_fcn.backward_activation(gradient6)
        gradient8 = self.conv_layer.backward_propagate(gradient7)
        #gradient 8 is not used since conv_layer is the first layer

        '''     show dimensions of gradients
        print("Show dimensions of gradients: ");
        print("Gradient1 dim: " + str(gradient1.shape));
        print(gradient1);
        print("Gradient2 dim: " + str(gradient2.shape));
        print(gradient2);
        print("Gradient3 dim: " + str(gradient3.shape));
        print(gradient3);
        print("Gradient4 dim: " + str(gradient4.shape));
        print(gradient4);
        print("Gradient5 dim: " + str(gradient5.shape));
        print(gradient5);
        print("gradient7 dim: " + str(gradient7.shape))
        '''

    def update_layers(self, learning_rate):
        if self.optimizer == "sgd":
            # print("here")
            self.input_layer.update_weight(learning_rate)
            self.hidden_layer1.update_weight(learning_rate)
            self.hidden_layer2.update_weight(learning_rate)
            self.conv_layer.update_weight(learning_rate)
        elif self.optimizer == "momentum":
            self.input_layer.update_weight(learning_rate, self.momentum)
            self.hidden_layer1.update_weight(learning_rate, self.momentum)
            self.hidden_layer2.update_weight(learning_rate, self.momentum)
            self.conv_layer.update_weight(learning_rate, self.momentum)

    def train(self, epoch, inputX, inputY):
        epochs = list()
        predictions = list()
        losses = list()
        accuracies = list()
        predicts = list()
        print("Now training " + self.data_loader.type + " model ...")
        for i in range(epoch):
            '''     update weight without batch( train and update weight once per data)
            # for j in range(inputX.shape[0]):
            #     predictY = self.forward_propagate(np.reshape(inputX[j],(1,2)))
            #     loss = np.mean((predictY - inputY[j]) ** 2)
            #     self.backward_propagate(2 * (predictY - inputY[j]) / inputY[j].shape[0])
            #     self.update_layers(self.learning_rate)
            '''
            predictY = self.forward_propagate(inputX)
            loss = np.mean((predictY - inputY) ** 2)
            self.backward_propagate(2 * (predictY - inputY) / inputY.shape[0])
            self.update_layers(self.learning_rate)
            predictY = self.forward_propagate(inputX)
            # loss = np.mean((predictY - inputY) ** 2)

            # self.backward_propagate(2 * (predictY - inputY) / inputY.shape[0])
            # self.update_layers(self.learning_rate)

            predictYs = np.zeros(predictY.shape)
            predictYs[np.absolute(predictY >= 0.5)] = 1
            predictYs[np.absolute(predictY < 0.5)] = 0
            predicts.append(predictYs)
            accuracy = (predictYs == inputY).sum() / inputY.shape[0]
            accuracies.append(accuracy)
            losses.append(loss)
            predictions.append(predictY)
            epochs.append(i + 1)
            if (i + 500) % 500 == 0:
                print("epoch {:5d}, loss: {:.4f}, accuracy: {:.4f}".format(
                    i + 500, loss, accuracy))
        return epochs, losses, accuracies, predicts

if __name__ == "__main__":
    np.random.seed(1)

    ''' model using optimizer: momentum
    model_xor = NN(2,100,1,"sigmoid", "xor",0.5,0.95);
    '''
    model_xor = NN(2,100,1,"sigmoid", "xor",0.1);
    # model_xor = NN(2,100,1,"leaky relu", "xor",0.01,"sgd",0);
    # model_xor = NN(2,150,1,"sigmoid", "xor",0.5,"momentum",0.95);

    # using linear model
    # model_linear = NN(2, 4, 1, "none", "linear", 0.05, "sgd", 0)
    # input_dataX, input_dataY = model_linear.data_loader.generate_fcn(1000)
    # epochs, losses, accuracies, prediction = model_linear.train(
    #     15000, input_dataX, input_dataY)
    # show_result(input_dataX, input_dataY, prediction,
    #             epochs, losses, accuracies, "Linear data")

    # using xor model
    input_dataX2, input_dataY2 = model_xor.data_loader.generate_fcn();
    epochs2,losses2,accuracies2,prediction2 = model_xor.train(30000,input_dataX2, input_dataY2);
    show_result(input_dataX2,input_dataY2,prediction2,epochs2,losses2,accuracies2,"Xor data");

    plt.show()
    # conv = convolutional([1,3],"sigmoid");
    # convert_data = np.reshape(input_dataX2[0],(1,2))
    # print(convert_data.shape)
    # conv.forward_output(input_dataX2[0]);

    '''
    code below is used to visualize the validation prediction v.s ground truth
    '''

    # validate_linear_X, validate_linear_Y = model_linear.data_loader.generate_fcn(1000);
    # predict_linear_Y = model_linear.forward_propagate(validate_linear_X);
    #validating linear model

    # validate_xor_X, validate_xor_Y = model_xor.data_loader.generate_fcn();
    # predict_xor_Y = model_xor.forward_propagate(validate_xor_X);
    # validating xor model

    # predict_linear_Ys = np.zeros(predict_linear_Y.shape);
    # for i in range(validate_linear_Y.shape[0]):
    #     if validate_linear_Y.shape[0] > 100:
    #         if i % 100 == 0:
    #             print("linear prediction: {:.4f}, label: {:1d}, actual: {:1d}".format(float(predict_linear_Y[i]), int(predict_linear_Y[i] >= 0.5) * 1 , int(validate_linear_Y[i])));

    # predict_xor_Ys = np.zeros(predict_xor_Y.shape);
    # for i in range(validate_xor_Y.shape[0]):
    #     print("xor prediction: {:.4f}, label: {:1d}, actual: {:1d}".format(float(predict_xor_Y[i]), int(predict_xor_Y[i] >= 0.5) * 1 , int(validate_xor_Y[i])));

    '''
    code below is used to get validation accuracy
    '''
    # predict_linear_Ys[np.absolute(predict_linear_Y >= 0.5)] = 1;
    # predict_linear_Ys[np.absolute(predict_linear_Y < 0.5)] = 0;
    # accuracy = (predict_linear_Ys == validate_linear_Y).sum() / validate_linear_Y.shape[0]
    # print("Linear Validate accuracies: " + str(accuracy));

    # predict_xor_Ys[np.absolute(predict_xor_Y >= 0.5)] = 1;
    # predict_xor_Ys[np.absolute(predict_xor_Y < 0.5)] = 0;
    # accuracy = (predict_xor_Ys == validate_xor_Y).sum() / validate_xor_Y.shape[0]
    # print("XOR Validate accuracies: " + str(accuracy));
