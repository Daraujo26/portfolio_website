from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import base64
import numpy as np
from PIL import Image
import io
import torch
from .torch_models import get_model
from torchvision import transforms

device = torch.device('cpu')
model_instance = get_model(device)


model_path = '/var/portfolio_website/www/django_portfolio/portfolio_website/proj_model.pt'
state_dict = torch.load(model_path, map_location=torch.device('cpu'))

model_instance.load_state_dict(state_dict)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

model_instance.eval()

def show_resume(request):
    return render(request, 'frontend/resume.html')

@csrf_exempt
def show_digit(request):
    if request.method == 'POST':
        # handle the prediction
        data = json.loads(request.body)
        image_data = base64.b64decode(data['image'].split(',')[1])

        image = Image.open(io.BytesIO(image_data))
        image = image.convert('L')
        
        # convert the image to a tensor and apply normalization
        image_tensor = transform(image).unsqueeze(0).to('cpu')  # Use the 'transform' instead of just 'ToTensor'

        # use the model for inference
        with torch.no_grad():
            predictions = model_instance(image_tensor)
            
        # get the predicted number
        predicted_number = torch.argmax(predictions, dim=1).item()  # Convert tensor to python scalar using .item()

        return JsonResponse({'number': predicted_number})

    else:
        # render the digit drawing and recognition page
        return render(request, 'frontend/digit_recognition.html')
