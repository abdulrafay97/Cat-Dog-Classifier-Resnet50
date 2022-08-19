#Import Packages
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision import models, transforms
import torch.nn as nn
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

icon = Image.open('icon.jpg')
st.set_page_config(page_title='Classifier', page_icon = icon)
st.header('Cat And Dog Classifier')

#Load Model
def Resnet50():
    model = models.resnet50(pretrained=False).to(device)

    model.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 2)).to(device)

    model.load_state_dict(torch.load('weights.h5' , map_location=torch.device('cpu')) )
    model.eval()
    return model

#Calculating Prediction
def Predict(img):
    Mod = Resnet50()
    pred_logits_tensor = Mod(img)
    
    pred_probs = F.sigmoid(pred_logits_tensor).cpu().data.numpy()
    return pred_probs



#Get Image
file_up = st.file_uploader('Upload an Image', type = "jpg")

#Normalizing
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

#Transforming the Image
data_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize
            ])

if file_up is not None:
    # display image that user uploaded
    image = Image.open(file_up)
    st.image(image, caption = 'Uploaded Image.', use_column_width = True)
    st.write("")
    st.write("Just a second ...")
    img = data_transform(image)
    img = torch.reshape(img , (1, 3, 224, 224))
    prob = Predict(img)

    if round(prob[0][0]) == 0:
        st.write("Its a Dog")
    if round(prob[0][0]) == 1:
        st.write("Its a Cat")
