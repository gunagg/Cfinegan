import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import os
import pickle
from torch.utils import model_zoo
from collections import OrderedDict
import pandas as pd

def load_model(model_name):

	model_urls = {
			'resnet50_trained_on_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
			'resnet50_trained_on_SIN_and_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
			'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
	}

	model = models.resnet50(pretrained=False)
	print(model._modules.keys())
	layer = model._modules.get('avgpool')
	model = torch.nn.DataParallel(model).cuda()
	checkpoint = model_zoo.load_url(model_urls[model_name])
	model.load_state_dict(checkpoint["state_dict"])
	return model, layer


# model = load_model("resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN")
model, layer = load_model("resnet50_trained_on_SIN")
# model = models.resnet18(pretrained=True)
# Use the model object to select the desired layer
# for k, v in model.module.state_dict().items():
# 	print(k, v.size())


	# print(layer)



model.eval()

transforms = transforms.Compose([transforms.Resize(256),
									  transforms.CenterCrop(224),
									  transforms.ToTensor(),
									  transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])
								 ])  


def get_vector(image_name):
	# 1. Load the image with Pillow library
	img = Image.open(image_name)
	# 2. Create a PyTorch Variable with the transformed image
	t_img = Variable(transforms(img).unsqueeze(0))
	# 3. Create a vector of zeros that will hold our feature vector
	#    The 'avgpool' layer has an output size of 512
	my_embedding = torch.zeros(1, 2048, 1, 1)
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


dict_val = {}
img_names = []
embeddings = []

filepath = os.path.join("../data/cartoons/", 'images.txt')
df_filenames = pd.read_csv(filepath, delim_whitespace=False, header=None)
filenames = df_filenames[1].tolist()

for idx, img_name in enumerate(filenames) :
    try :
        img_name = os.path.join("../data/cartoons/images/", img_name)
        embedding = get_vector(img_name)

        dict_val[img_name] = embedding
        
        print("%d steps reached "%idx)
    except :
        print("error occured")
        continue

np.save("embeddings_shape_cartoons_cropped.npy", dict_val)




