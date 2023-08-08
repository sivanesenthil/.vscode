import gradio as gr
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

#  ResNet50 model
model = models.resnet50(pretrained=True)
model.eval()

# Preprocess 
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Labels
LABELS_PATH = "imagenet.txt"
with open(LABELS_PATH) as f:
    labels = [line.strip() for line in f.readlines()]

def classify_image(input_image):
    input_image = Image.fromarray(input_image.astype('uint8'), 'RGB')
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    
    # predictions
    with torch.no_grad():
        output = model(input_batch)
    predicted_idx = torch.argmax(output).item()
    
    predicted_label = labels[predicted_idx]
    
    return predicted_label

#  Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs="image",
    outputs="text",
    title="ImageNet model",
    description="Upload an image ",
     allow_flagging='never',
)

# Launch 
interface.launch(share=True)
