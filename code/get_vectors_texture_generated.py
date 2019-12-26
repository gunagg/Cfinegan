import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import os
import pickle


model = models.resnet18(pretrained=True)
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

model.eval()

scaler = transforms.Scale((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_vector(image_name):
    # 1. Load the image with Pillow library
    img = Image.open(image_name)
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    The 'avgpool' layer has an output size of 512
    my_embedding = torch.zeros(1, 512, 1, 1)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        my_embedding.copy_(o.data)
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return np.reshape(my_embedding.data.numpy(), [-1])

base_dir = "../output/images_cartoons"

dict_val = {}
img_names = []
embeddings = []

for idx, img_name in enumerate(os.listdir(base_dir)) :
	img_name = os.path.join(base_dir, img_name)

	# img_names.append(img_names)


	embedding = get_vector(img_name)

	dict_val[img_name] = embedding
	
	print("%d steps reached "%idx)
	# print(embedding.shape)

	# if idx >= 5 :
	# 	break
np.save("embeddings_cartoons.npy", dict_val)




