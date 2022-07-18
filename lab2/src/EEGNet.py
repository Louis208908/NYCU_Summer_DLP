from re import L
from torch import _adaptive_avg_pool2d, batch_norm
import torch.nn as nn
import torch

class EEGNet(nn.Module):
    def __init__(self, activation_type, momentum, dropout_rate = 0.25):   
        #   droupout_rate Pr{模型中的某些node不被activation fcn 激活，希望可以以此減少overfit出現的機率}
        super(EEGNet, self).__init__();

        self.activation_type = activation_type;

        if(activation_type == "relu"):
            self.activation_fcn = nn.ReLU();
        elif(activation_type == "elu"):
            self.activation_fcn = nn.ELU();
        elif(activation_type == "leaky_relu"):
            self.activation_fcn = nn.LeakyReLU();

        self.dropout_rate = dropout_rate;

        self.firstConv_layer = nn.Sequential(
            nn.Conv2d(1,16, kernel_size=(1,51), padding=(0,25), bias=False),
            nn.BatchNorm2d(16)
        );

        self.depthWiseConv_layer = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=(2,1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            self.activation_fcn,
            nn.AvgPool2d(kernel_size=(1,4), stride = (1,4)),
            nn.Dropout(p = self.dropout_rate)
        );

        self.separableConv_layer = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15),padding=(0,7), bias=False),
            nn.BatchNorm2d(32),
            self.activation_fcn,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(p = self.dropout_rate)
        )

        self.classiy_layer = nn.Sequential(
            nn.Linear(in_features = 736,out_features = 2)
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

    def train(self,device,training_dataloader ,epochs, lr):
        loss_fcn = nn.CrossEntropyLoss();
        # optim = torch.optim.Adam(self.parameters(), lr = lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, weight_decay = 0.01 )
        optim = torch.optim.Adam(self.parameters())
        for i in range(1,epochs + 1):
            for batch_idx, (data, target) in enumerate(training_dataloader):
                optim.zero_grad();
                # 因為gradient 會是累加的，所以需要用這個zero_grad()將前一次的grad清除
                data = data.to(device);
                target = target.to(device);
                prediction = self(data);
                # print(prediction)
                # print("prediction dim: " + str(prediction.size()))
                preds = torch.argmax(prediction,dim = 1)
                # 把forward 出來的一個數值轉成我們需要的label (0或1)
                # print(preds)
                loss = loss_fcn(prediction,target);

                # accuracy = torch.sum(preds == target).item() / len(target)
                # .item() 用來取得純量
                # torch.sum() 用來計算prediction當中有多少個跟target是相同的
                loss.backward();
                #算出所有loss的來源的gradient
                '''
                y = ax + bz + cd
                y.backward();
                這樣就會計算出x,z,d,a,b,c的gradient
                '''

                optim.step();
                # 根據loss.backward算出來的gradient修正weight

            if i % 10 == 0:
                print({"now epoch: ":i, "loss: ":loss.item(), "accuracy: ": accuracy});

    def evaluate(self,device,testing_dataloader):
        for batch_idx, (data, target) in enumerate(testing_dataloader):
            data = data.to(device);
            target = target.to(device);
            prediction = self(data);
            # print({"prediction type: ":prediction.type()});
            preds = torch.argmax(prediction, dim = 1);
            # print({"preds type: ":preds.type()});
            
            accuracy = torch.sum(preds == target).item()/ len(target);
            print({"Testing accuracy: ":accuracy});

    def train_and_eval(self, device, training, testing, epoch, lr):
        loss_fcn = nn.CrossEntropyLoss()
        optim = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, weight_decay = 0.01)
        # optim = torch.optim.Adam(self.parameters())
        # optim = torch.optim.Adam(self.parameters(), lr=lr)
        best_epoch = 0
        best_acc = 0
        epochs, train_accs, test_accs = list(), list(), list()
        for i in range(1, epoch + 1):
            correct, total_samples, total_loss = 0, 0, 0
            for batch_idx, (data, target) in enumerate(training):
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
                ''' Ex:
                y = ax + bz + cd
                y.backward();
                這樣就會計算出x,z,d,a,b,c的gradient
                '''
                optim.step()
                # 根據loss.backward算出來的gradient修正weight
                optim.zero_grad()
                # 因為gradient 會是累加的，所以需要用這個zero_grad()將前一次的grad清除
            train_acc = correct / total_samples

            #evaluation process
            for batch_idx, (data, target) in enumerate(testing):
                data = data.to(device)
                target = target.to(device)
                prediction = self(data)
                preds = torch.argmax(prediction, dim=1)
                test_acc = torch.sum(preds == target).item() / len(target)
            epochs.append(i)
            train_accs.append(train_acc * 100.0)
            test_accs.append(test_acc * 100.0)
            if(test_acc > best_acc):
                best_acc = test_acc
                best_epoch = i
            # if i % 50 == 0:
            #     print({"now epoch: ": i, "loss: ": loss.item(), "training accuracy: ": accuracy, "testing accuracy: ": test_acc})
        return best_epoch, best_acc, epochs, train_accs, test_accs
        # return best_epoch, best_acc
