import torch
import cv2 
from ShapeClassifier import ShapeClassifierNetwork

def test(img_path):
    labels = {0:'Circle', 1:'Rectangle', 2:'Square', 3:'Triangle', 4:'Star'}
    model_file = "ShapeModels/model_weights.pth"
    model = ShapeClassifierNetwork(28*28,5)
    model.load_state_dict(torch.load(model_file))
    model.eval()
    # img = cv2.imread(f"{img_path}",0)
    img = cv2.resize(cv2.imread(img_path,0),(28,28))
    img= torch.from_numpy(img)/255.0
    z = model(img.view(-1,28*28))
    # z = model(torch.rand(1, 784))
    print(z.shape)
    print(z)
    _,yhat = torch.max(z,1)
    print("Label : ", labels[int(yhat)])
test("TestImage/st1.png")