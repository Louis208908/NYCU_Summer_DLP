from re import L
from torch import _adaptive_avg_pool2d, batch_norm
import torch.nn as nn
import torch


class DeepConv(nn.Module):
    def __init__(self, activation_type,momentum ,dropout_rate=0.25):
        #   droupout_rate Pr{模型中的某些node不被activation fcn 激活，希望可以以此減少overfit出現的機率}
        super(DeepConv, self).__init__()
        self.activation_type = activation_type


        if(activation_type == "relu"):
            self.activation_fcn = nn.ReLU()
        elif(activation_type == "elu"):
            self.activation_fcn = nn.ELU()
        elif(activation_type == "leaky_relu"):
            self.activation_fcn = nn.LeakyReLU()

        self.dropout_rate = dropout_rate
        self.momentum = momentum;

        self.conv0 = nn.Conv2d(1, 25, kernel_size=(1, 5))

        self.conv1 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=(2, 1)),
            # nn.BatchNorm2d(25, eps=1e-05, momentum = momentum),
            nn.BatchNorm2d(25),
            self.activation_fcn,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(self.dropout_rate)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1, 5)),
            nn.BatchNorm2d(50),
            self.activation_fcn,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(self.dropout_rate)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1, 5)),
            nn.BatchNorm2d(100),
            self.activation_fcn,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(self.dropout_rate)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1, 5)),
            nn.BatchNorm2d(200),
            self.activation_fcn,
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(self.dropout_rate)
        )
        self.classify = nn.Sequential(
            nn.Linear(8600, 2),
        )

    def forward(self, input_data):
        a1 = self.conv0(input_data);
        a2 = self.conv1(a1);
        a3 = self.conv2(a2);
        a4 = self.conv3(a3);
        a5 = self.conv4(a4);


        flatten_a5 = a5.view(a5.shape[0],-1);

        final_result = self.classify(flatten_a5);

        return final_result
        print("input dim: " + str(input_data.size()));
        print("a1 dim: " + str(a1.size()));
        print("a2 dim: " + str(a2.size()));
        print("a3 dim: " + str(a3.size()));
        print("a4 dim: " + str(a4.size()));
        print("a5 dim: " + str(a5.size()));
        print("flatten a5 dim: " + str(flatten_a5.size()));
        print("final dim: " + str(final_result.size()));


    def train(self, device, training_dataloader, epochs, lr):
        loss_fcn = nn.CrossEntropyLoss()
        # optim = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False, weight_decay = 0.01)
        optim = torch.optim.Adam(self.parameters())
        for i in range(1, epochs + 1):
            for batch_idx, (data, target) in enumerate(training_dataloader):
                data = data.to(device)
                target = target.to(device)
                prediction = self(data)
                # print(prediction)
                # print("prediction dim: " + str(prediction.size()))
                preds = torch.argmax(prediction, dim=1)
                # 把forward 出來的一個數值轉成我們需要的label (0或1)
                # print(preds)
                loss = loss_fcn(prediction, target)

                accuracy = torch.sum(preds == target).item() / len(target)
                # .item() 用來取得純量
                # torch.sum() 用來計算prediction當中有多少個跟target是相同的
                loss.backward()
                #算出所有loss的來源的gradient
                '''
                y = ax + bz + cd
                y.backward();
                這樣就會計算出x,z,d,a,b,c的gradient
                '''

                optim.step()
                # 根據loss.backward算出來的gradient修正weight
                optim.zero_grad()
                # 因為gradient 會是累加的，所以需要用這個zero_grad()將前一次的grad清除

            if i % 50 == 0:
                print({"now epoch: ": i, "loss: ": loss.item(),"accuracy: ": accuracy})

    def evaluate(self, device, testing_dataloader):
        for batch_idx, (data, target) in enumerate(testing_dataloader):
            data = data.to(device)
            target = target.to(device)
            prediction = self(data)
            # print({"prediction type: ":prediction.type()});
            preds = torch.argmax(prediction, dim=1)
            # print({"preds type: ":preds.type()});

            accuracy = torch.sum(preds == target).item() / len(target)
            print({"Testing accuracy: ": accuracy})

    def train_and_eval(self,device,training,testing,epoch,lr):
        loss_fcn = nn.CrossEntropyLoss()
        # optim = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, amsgrad=False)
        # optim = torch.optim.Adam(self.parameters(), lr=lr)
        optim = torch.optim.Adam(self.parameters())
        best_epoch = 0;
        best_acc = 0;
        epochs, train_accs, test_accs = [], [], []
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
                '''
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
            train_accs.append(train_acc)
            test_accs.append(test_acc)
            if(test_acc > best_acc):
                best_acc = test_acc;
                best_epoch = i;
            if i % 10 == 0:
                print({"now epoch: ": i, "loss: ": loss.item(),"training accuracy: ":train_acc, "testing accuracy: ":test_acc})
        # return best_epoch, best_acc,epochs,train_accs,test_accs        
        return best_epoch, best_acc
