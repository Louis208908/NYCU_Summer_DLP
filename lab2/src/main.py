from inspect import Parameter
import torch
import numpy as np
import matplotlib.pyplot as plt
import secrets
import dataloader
# np.random.seed(secrets.randbelow(1_000_000_000)) 
from torch.utils.data import TensorDataset, DataLoader
from EEGNet import EEGNet
from DeepConvNet import DeepConv


def plot(epoch,train,test,net,act_fcn):
    plt.suptitle("Comparison of accuracy between models." + "(" +  net + ")");
    plt.xlabel("epoch");
    plt.ylabel("Accuracy(%)");
    if(act_fcn == "relu"):
        plt.plot(epoch,train,"r");
        plt.plot(epoch,test,"g");
    elif act_fcn == "leaky_relu":
        plt.plot(epoch,train,"b");
        plt.plot(epoch,test,"y");
    elif act_fcn == "elu":
        plt.plot(epoch,train,"c");
        plt.plot(epoch,test,"m");

    



class trainer():
    def __init__(self,epochs, network_type, dropout_rate,device):
        self.epochs = epochs;
        self.network_type = network_type;
        self.droupout_rate = dropout_rate;
        self.device = device

        self.modelList = list();
        if(network_type == "EEGNET"):
            model_relu = EEGNet("relu", self.droupout_rate).to(device);
            model_lrelu = EEGNet("leaky_relu", self.droupout_rate).to(device);
            model_elu = EEGNet("elu", self.droupout_rate).to(device);
        elif(network_type == "DeepConv"):
            model_relu = DeepConv("relu", self.droupout_rate).to(device);
            model_lrelu = DeepConv("leaky_relu", self.droupout_rate).to(device);
            model_elu = DeepConv("elu", self.droupout_rate).to(device);

        self.modelList.append(model_relu);
        self.modelList.append(model_lrelu);
        self.modelList.append(model_elu);
    def train_model(self, training_loader):
        print("now training model: " + self.network_type);
        for id,model in enumerate(self.modelList):
            print("now training activation fcn type: " + model.activation_type)
            model.train(self.device, training_loader,self.epochs);
    def evaluate_model(self, testing_loader):
        print("now training network: " + str(self.network_type));
        for id,model in enumerate(self.modelList):
            print("now evaluating activation fcn type: " + model.activation_type)
            model.evaluate(self.device,testing_loader);

    def train_evaluate(self, training, testing):
        # print("now training network: " + str(self.network_type))
        # fw = open("./record.csv","a");
        for id, model in enumerate(self.modelList):
            print("now using activation fcn type: " + model.activation_type)
            best_epoch , best_acc,epochs ,train_acc, test_acc = model.train_and_eval(self.device, training, testing, self.epochs)
            # fw.write("{:11s}, {:10s}, {:13.1E}, {:7.4f},{:7.4f} ,{:10d}, {:9.4f}\n".format(
            #     self.network_type, model.activation_type, self.lr, self.momentum, self.droupout_rate, best_epoch, best_acc))
            if best_acc > 0.87:
                print("Best epoch: " + str(best_epoch));
                print("Best_acc: " + str(best_acc));
                
            plot(epochs,train_acc,test_acc,self.network_type,model.activation_type);
        plt.legend(["ReLU train", "ReLU test", "Leaky_ReLU train", "Leaky_ReLU test", "ELU train", "ELU test"])    
        plt.show()    


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type != "cuda":
        print("Not using GPU for training!")

    # torch.manual_seed(12)
    # torch.cuda.manual_seed(12)
    # torch.cuda.manual_seed_all(12)


    train_data, train_label, test_data, test_label = dataloader.read_bci_data();
    train_loader, test_loader = dataloader.load_data(train_data,train_label, test_data, test_label,64,64);

    plt.figure(figsize=(15, 12))
    # for _ in range(10):
    # model_trainer = trainer(300,"EEGNET",0.25,device);
    model_trainer = trainer(300,"DeepConv",0.55,device);
    model_trainer.train_evaluate(train_loader, test_loader)



