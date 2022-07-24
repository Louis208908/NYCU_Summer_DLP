from torch import _adaptive_avg_pool2d, batch_norm
import torch.nn as nn
import torch
import copy

class EEGNet(nn.Module):
    def __init__(self, activation_type,lr ,dropout_rate = 0.25):   
        #   droupout_rate Pr{模型中的某些node不被activation fcn 激活，希望可以以此減少overfit出現的機率}
        super(EEGNet, self).__init__();

        self.activation_type = activation_type;
        self.learning_rate = lr

        if(activation_type == "relu"):
            self.activation_fcn = nn.ReLU();
        elif(activation_type == "elu"):
            self.activation_fcn = nn.ELU(alpha = 0.8);
        elif(activation_type == "leaky_relu"):
            self.activation_fcn = nn.LeakyReLU();

        self.dropout_rate = dropout_rate;

        self.firstConv_layer = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1,51), stride=(1,1), padding=(0,25), bias=False),
            nn.BatchNorm2d(16)
        );

        self.depthWiseConv_layer = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(2,1), stride=(1,1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            self.activation_fcn,
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4), padding=0),
            nn.Dropout(p = self.dropout_rate)
        );

        self.separableConv_layer = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1,15), stride=(1,1), padding=(0,7), bias=False),
            nn.BatchNorm2d(32),
            self.activation_fcn,
            nn.AvgPool2d(kernel_size=(1,8), stride=(1,8)),
            nn.Dropout(p = self.dropout_rate)
        )

        self.classiy_layer = nn.Sequential(
            nn.Linear(in_features = 736,out_features = 2, bias = True)
        )
    
    def forward(self, input_data):
        a1 = self.firstConv_layer(input_data);
        a2 = self.depthWiseConv_layer(a1);
        a3 = self.separableConv_layer(a2);
        a4 = a3.view(a3.shape[0],-1);
        # 將a3這個tensor flatten 成一維陣列

        final_result = self.classiy_layer(a4);

        return final_result;
        print("a1 dim: " + str(a1.size()));
        print("a2 dim: " + str(a2.size()));
        print("a3 dim: " + str(a3.size()));
        print("a4 dim: " + str(a4.size()));


    def evaluate(self,device,testing_dataloader):
        correct = 0;
        total_samples = 0;
        for batch_idx, (data, target) in enumerate(testing_dataloader):
            data = data.to(device);
            target = target.to(device);
            prediction = self(data);
            # print({"prediction type: ":prediction.type()});
            preds = torch.argmax(prediction, dim = 1);
            # print({"preds type: ":preds.type()});
            correct += torch.sum(preds == target).item()
            total_samples += preds.shape[0]
            
        accuracy = correct / total_samples;
        # print({"Testing accuracy: ":accuracy});
        return accuracy

    def train_and_eval(self, device, training, testing, epoch):
        loss_fcn = nn.CrossEntropyLoss()
        # optim = torch.optim.Adam(self.parameters(), lr=lr, weight_decay = 0.01)
        optim = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        best_epoch = 0
        best_acc = 0
        epochs, train_accs, test_accs = list(), list(), list()
        for i in range(1, epoch + 1):
            self.train()
            correct, total_samples, total_loss = 0, 0, 0
            for batch_idx, (data, target) in enumerate(training):
                optim.zero_grad()
                # 因為gradient 會是累加的，所以需要用這個zero_grad()將前一次的grad清除
                data = data.to(device)
                target = target.to(device)
                prediction = self(data)
                preds = torch.argmax(prediction, dim=1)
                # 把forward 出來的一個數值轉成我們需要的label (0或1)
                loss = loss_fcn(prediction, target)
                correct += torch.sum(preds == target).item()
                total_samples += preds.shape[0]
                total_loss += loss.item()
                # .item() 用來取得純量
                # torch.sum() 用來計算prediction當中有多少個跟target是相同的
                loss.backward()
                #算出所有loss的來源的gradient
                optim.step()
                # 根據loss.backward算出來的gradient修正weight
            train_acc = correct / total_samples

            #evaluation process
            self.eval();
            test_acc = self.evaluate(device,testing)
            epochs.append(i)
            train_accs.append(train_acc * 100.0)
            test_accs.append(test_acc * 100.0)
            if(test_acc > best_acc):
                best_acc = test_acc
                best_epoch = i
                if test_acc > 0.87:
                    fileName = "../model_depository/" + "EEGNET_" + self.activation_type + "_" + str(test_acc * 100.0) + ".pt"
                    # state_dict = copy.deepcopy(self.state_dict())
                    torch.save(self,fileName)
                    # self.evaluate(device,testing)
            if i % 50 == 0:
                print({"now epoch: ": i, "loss: ": loss.item(), "training accuracy: ": train_acc, "testing accuracy: ": test_acc})
        return best_epoch, best_acc, epochs, train_accs, test_accs
        # return best_epoch, best_acc
