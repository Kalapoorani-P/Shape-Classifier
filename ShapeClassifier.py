import torch 
import torchvision
from torchvision import transforms
from torch import nn,optim
from torch.utils.data import DataLoader,Dataset,Sampler
import glob 
import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F


torch.manual_seed(1)
writer = SummaryWriter('classifier/runs')

class ShapeDataset(Dataset):
    def __init__(self,path ,dataset_type):
        
        """
        self.path contain the path of dataset 
        mode = 1 or 0 ( if mode is zero consider as train set else consider as val set )

        """
        self.path =  path
        self.Image_path_list = []
        self.classes = []

        self.dataset_type = dataset_type
        self.classlabel = {}
        label_no = 0
        for label in os.listdir(path):
            if os.path.isdir(path+label):
                self.classlabel[label] = label_no
                label_no+=1 
        for data_path in glob.glob(path+"**/*.jpg"):
            self.Image_path_list.append(data_path)
        
        np.random.shuffle(self.Image_path_list)

        if self.dataset_type==1:
            self.train = self.Image_path_list[:int(len(self.Image_path_list)*0.8)]
            
        elif self.dataset_type == 0:
            self.test = self.Image_path_list[int(len(self.Image_path_list)*0.8):]

        else:
            self.allImage = self.Image_path_list

        


    def getImageAndLabel(self,data,index):
        img_path = data[index]
        label = torch.tensor(self.classlabel[img_path.split('/')[-2]])

        image = cv2.resize(cv2.imread(img_path,0),(28,28),interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(image)/255.0
        # image = torch.cat(image,dim=0)
        # print(image.shape)
        return image,label

    def __getitem__(self,index):
       
        if self.dataset_type==1 :
            return self.getImageAndLabel(self.train,index)

        elif self.dataset_type == 0:
            return self.getImageAndLabel(self.test,index)
        else:
            return self.getImageAndLabel(self.allImage,index)

    def __len__(self):
        if self.dataset_type==1:
            return len(self.train)

        elif self.dataset_type == 0:
            return len(self.test)

        else:
            return len(self.allImage)
            




class LinBNRelu(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()
        self.ln = nn.Linear(input_size,output_size)
        self.bn = nn.BatchNorm1d(num_features=output_size)
        self.act = nn.ReLU()
    def forward(self,x):
        x = self.ln(x)
        x = self.bn(x)
        x = self.act(x)
        return x



class ShapeClassifierNetwork(nn.Module):
   def __init__(self,input_size,output_size):
      super().__init__()
      self.input_layer = LinBNRelu(input_size,128)
      self.hidden1 = LinBNRelu(128,128)
      self.hidden2 = LinBNRelu(128,128)
      self.hidden3 = LinBNRelu(128,64)
      self.output_layer = nn.Linear(64,output_size)

   def forward(self,x):
      x = self.input_layer(x)
      x = self.hidden1(x)
      x = self.hidden2(x)
      x = self.hidden3(x)
      x = self.output_layer(x)

      return x 

def one_hot_encoded(y):
    num_classes=5
    y_onehot = torch.zeros(y.size(0),num_classes)
    y_onehot.scatter_(1, y.view(-1, 1).long(), 1).float()
    return y_onehot


def cross_entropy(input, target):
    logsoftmax = nn.LogSoftmax()
    return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))



def train(model,train_loader,test_loader,optimizer,epochs=100):
    loss_list = []
    accuracyList = []
    test_loss_list = []
    for epoch in range(epochs):
        total_train_loss,toal_test_loss,correct=0,0,0
        for x_train,y_train in train_loader:
            # print(x_train)
            model.train()
            z = model(x_train.view(-1,28*28))
            optimizer.zero_grad()
            y_train = one_hot_encoded(y_train)
            loss = cross_entropy(z,y_train)
            
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()



        for x_test,y_test in test_loader:
            # print(x_test)
            model.eval()
            z_test = model(x_test.view(-1,28*28))
           
            y_enc = one_hot_encoded(y_test)
            test_loss = cross_entropy(z_test,y_enc)
            
            _,z_yhat = torch.max(z_test,1)
            correct += (y_test==z_yhat).sum().item()
            toal_test_loss += test_loss
        val_accuracy = (correct / len(test_dataset)) * 100
        
        if epoch%10==0:
            print(f'Epoch {epoch+0:03}: | Train Loss: {total_train_loss:.3f} | Test Loss:{toal_test_loss:.3f} | val_accuracy: {val_accuracy:.5f}')
        loss_list.append(total_train_loss)
        accuracyList.append(val_accuracy)

        writer.add_scalar(f"Train/Train-loss",total_train_loss,epoch)
        writer.add_scalar(f"Acc/Accuracy",val_accuracy,epoch)
        writer.add_scalar(f"Train/Test-loss",toal_test_loss,epoch)
    plt.plot(range(epochs),loss_list,label="loss")
    plt.plot(range(epochs),accuracyList,label="accuracy")
    plt.legend()
    writer.close()
    return model
def get_model():
    model = torch.load("model.pth")
    model.eval()
    return model.state_dict()

def test(path,model):
    testDataset = ShapeDataset(path,2)
    for img,label in testDataset:
        pass



if __name__=="__main__":
    path="ImageData/"
    train_dataset = ShapeDataset(path,1)
    test_dataset = ShapeDataset(path,0)
    train_loader = DataLoader(dataset=train_dataset,batch_size=64,shuffle=True,num_workers=4)
    test_loader = DataLoader(dataset=test_dataset,batch_size=64,shuffle=True,num_workers=4)
    input_dim = 28*28 
    output_dim = 5
    model = ShapeClassifierNetwork(input_dim,output_dim)
    loss_func = nn.CrossEntropyLoss()
    soft_max = torch.nn.Softmax()
    optimizer = optim.Adam(model.parameters(),lr = 0.1)
    # Tensorboard - Image and Graph Draw 
    images, labels = next(iter(train_loader))
    # print(images.shape,images.dtype)
    img_grid = torchvision.utils.make_grid(images.unsqueeze(1))
    writer.add_image('shape_images', img_grid)
    writer.add_graph(model,images.view(-1,28*28))
    trained_model = train(model,train_loader,test_loader,optimizer,epochs=400)
    torch.save(model.state_dict(), 'ShapeModels/model_weights.pth')
    torch.save(trained_model,"ShapeModels/model.pth")
    # print(get_model())