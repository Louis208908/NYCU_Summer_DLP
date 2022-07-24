import torch
from EEGNet import EEGNet
import dataloader

def model_evaluation(model,test,device):
    for batch_idx, (data, target) in enumerate(test):
        data = data.to(device)
        target = target.to(device)
        model.eval();
        prediction = model(data)
        # print({"prediction type: ":prediction.type()});
        preds = torch.argmax(prediction, dim=1)
        # print({"preds type: ":preds.type()});

        accuracy = torch.sum(preds == target).item() / len(target)
        print({"Testing accuracy: ": accuracy})

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_data, train_label, test_data, test_label = dataloader.read_bci_data();
    train_loader, test_loader = dataloader.load_data(train_data,train_label, test_data, test_label,64,1080);
    model = torch.load("../model_depository/EEGNET_leaky_relu_89.07407407407408.pt").to(device)

    # model = model.to(device)
    print(model.evaluate(device, test_loader))