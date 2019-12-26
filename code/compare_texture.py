import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import os
import pickle
from shutil import copy
import scipy.misc

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

def save_images(img_name1, img_name2, new_name) :
    cropped_image = np.array(Image.open(img_name1))
    x,y,z = cropped_image.shape
    max_dim = max(x,y)
    padded_img = np.zeros([max_dim, max_dim, z])
    padded_img[0:x,0:y,:] = cropped_image
    img1 = scipy.misc.imresize(padded_img, (128, 128))
    img2 = np.array(Image.open(img_name2))
    img3 = np.concatenate((img1, img2), axis = 1)
    img = Image.fromarray(img3.astype(np.uint8))
    img.save(new_name)


dict_val = {}
img_names = []
embeddings = []

# dict_val = np.load("embeddings.npy", allow_pickle = True).item()
dict_val = np.load("embeddings_cartoons.npy", allow_pickle = True).item()


embeddings = []
img_names = []
for key in dict_val :
    img_names.append(key)
    embeddings.append(dict_val[key])

cropped_val = np.load("embeddings_cartoons_cropped.npy", allow_pickle = True).item()
cropped_embeddings = []
cropped_img_names = []
for key in cropped_val :
    cropped_img_names.append(key)
    cropped_embeddings.append(cropped_val[key])



embeddings = np.array(embeddings)
cropped_embeddings = np.array(cropped_embeddings)

new_dir = "cartoons_compare_images"
if not os.path.exists(new_dir) :
    os.makedirs(new_dir)

for i in range(len(cropped_embeddings)) :
    cr_emb = np.reshape(cropped_embeddings[i], [-1, 512])
    distances = np.sum(np.square(embeddings - cr_emb), axis = -1)
    idx = np.argmin(distances)
    new_name = os.path.join(new_dir, cropped_img_names[i].split('/')[-1].split(".")[0] + "+" + img_names[idx].split('/')[-1])
    save_images(cropped_img_names[i], img_names[idx], new_name)
    if i % 100 == 0 :
        print("%d steps reached "%i)

# print(embeddings.shape)
# print(cropped_embeddings.shape)




