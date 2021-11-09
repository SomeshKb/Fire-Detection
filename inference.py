import os
import cv2
from torchvision import datasets
import torch
from torchvision.transforms import transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
Image.LOAD_TRUNCATED_IMAGES = True
plt.ion()

class_names = class_names = ['Fire', 'Neutral', 'Smoke']

def loadModel():
    print("[INFO] loading model...")
    model = torch.load('./trained-models/model_final.pt')
    model.eval()
    return model

def predict(image,model):
    prediction_transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    image = prediction_transform(image)[:3,:,:].unsqueeze(0)
    image = image.cuda()

    pred = model(image)
    idx = torch.argmax(pred)
    prob = pred[0][idx].item()*100
    
    return class_names[idx], prob

def predictImage(path,model):
    img = Image.open(path)
    plt.imshow(img)
    plt.show()
    prediction, prob = predict(img,model)
    print(prediction, prob)

def predictVideo(videoFile,model):
    cap = cv2.VideoCapture(videoFile)

    while True:

        ret, image = cap.read()
        draw = image.copy()
        draw = cv2.resize(draw,(640,480))

        draw = transforms.ToPILImage()(draw)
        prediction, prob = predict(draw,model)

        print(prediction)

        if prediction == 'Neutral':
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        cv2.putText(image, (prediction+' '+str(prob)), (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


        cv2.imshow('framename', image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


model = loadModel()

predictVideo('crop-buring.mp4',model)





