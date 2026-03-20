import torch
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from base.utils import instantiate_from_config, get_device 
import random
from tqdm import tqdm
import clip
import torch
from PIL import Image

import open_clip

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = torch.device('cuda')
print(device)

model_list = ['RN50','RN101','ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-H-14']

brain_data = 'EEG'
# brain_data = 'MEG'

for model_name in model_list:
    model_name
    device = "cuda" if torch.cuda.is_available() else "cpu"


    if model_name in ['RN50', 'RN101','ViT-B/32', 'ViT-B/16', 'ViT-L/14']:
        model, preprocess = clip.load(model_name, device=device)

    else:
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained='laion2b_s32b_b79k',
            device=device
        )

    model = model.to(device)

    class ImageTextDataset(Dataset):
        def __init__(self, image_files, preprocess, blur_transform):
            self.image_files = image_files
            self.preprocess = preprocess
            self.blur_transform = blur_transform

        def __len__(self):
            return len(self.image_files)

        def __getitem__(self, idx):
            image = Image.open(self.image_files[idx]).convert("RGB")
            image = image.resize((224, 224)) 
            image = self.blur_transform(image)
            image = self.preprocess(image)

            return image

    config_path = 'configs/ubp.yaml'
    config = OmegaConf.load(config_path)
    config['c'] = 6
    blur_param = config['data']['blur_type']
    for blur_kernel_size in [51]:
        for system_g in [3]:
            config['blur_kernel_size'] = blur_kernel_size 
            config['system_g'] = system_g
            blur_transform = instantiate_from_config(blur_param)

            original_model_name = model_name  

            for mode in ['train', 'test']:
                if mode == 'train':
                    if brain_data == 'EEG':
                        data_path = "/your_data_path/training_images/data/training_images/"
                    else:
                        data_path = "/your_data_path/Image_set/training_images/"
                else: 
                    if brain_data == 'EEG':
                        data_path = "/your_data_path/training_images/data/test_images/"
                    else:
                        data_path = "/your_data_path/Image_set/test_images/"


                image_files = glob.glob(data_path + "**/*.*", recursive=True) 
                image_files.sort()

                batch_size = 256
                dataset = ImageTextDataset(image_files, preprocess, blur_transform)
                dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

                model_name = original_model_name 


                if model_name == "ViT-B/32":
                    model_name = "ViT_B_32"
                elif model_name == "ViT-B/16":
                    model_name = "ViT_B_16"
                elif model_name == "ViT-L/14":
                    model_name = "ViT_L_14"
                elif model_name == 'ViT-H-14':
                    model_name = "ViT_H_14"
                print('model_name: ', model_name)

                image_latent_list = []
                with torch.no_grad():
                    for images in tqdm(dataloader):
                        images = images.to(device)

                        image_features = model.encode_image(images)
                        image_latent_list.append(image_features)
                
                image_latent = torch.cat(image_latent_list, dim=0)
                print('latent_features: ',image_latent.shape)


                if brain_data == 'EEG':
                    torch.save(image_latent, f"/your_data_path/visual_feature/blur/{model_name}_k{blur_kernel_size}_g{system_g}_{mode}.pt")
                    print(f"Latent features saved! Shape: {image_latent.shape}")
                else:
                    torch.save(image_latent, f"/your_data_path/visual_feature_MEG/blur/{model_name}_k{blur_kernel_size}_g{system_g}_{mode}.pt")
                    print(f"Latent features saved! Shape: {image_latent.shape}")
