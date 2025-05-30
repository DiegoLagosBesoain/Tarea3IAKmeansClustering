import torch
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import clip

# data_dir = '/hd_data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
# image_dir = os.path.join(data_dir, 'JPEGImages')
# val_file = 'data/voc_val.txt'
# data_dir = '/hd_data/Paris/'
# image_dir = os.path.join(data_dir, 'paris')
# val_file = 'data/val_paris.txt'
DATASET = 'VocPascal'
MODEL = 'DINO'
data_dir = 'VocPascal'
image_dir = os.path.join(data_dir, 'JPEGImages')
list_of_images = os.path.join(data_dir, 'train_voc.txt')
if __name__ == '__main__':
    #reading data
    with open(list_of_images, "r+") as file: 
        files = [f.split('\t') for f in file]
        
    # check GPU 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # defining the image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]),
        ])
    #load de model
    dim = 512     
    model = None
    if MODEL == 'resnet18' :
        model = models.resnet18(pretrained=True).to(device)
        model.fc = torch.nn.Identity() 
    if MODEL == 'resnet34' :
        model = models.resnet34(pretrained=True).to(device)
        model.fc = torch.nn.Identity() 
    #you can add more models
    if MODEL=="DINO":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        dim = 384
    if MODEL=="CLIP":
        model, preprocess_clip = clip.load("ViT-B/32", device=device) 
        
    
    #Pasamos la imagen por el modelo
    with torch.no_grad():        
        n_images = len(files)
        features = np.zeros((n_images, dim), dtype = np.float32)        
        for i, file in enumerate(files):                
            img_name = file[0].strip()
            label = file[1].strip()
            xmax, xmin, ymax, ymin = map(int, file[2:6])
            xmin, ymin, xmax, ymax = min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)

            filename = os.path.join(image_dir, img_name + ".jpg")

            # Cargar y recortar la imagen al bounding box
            image = Image.open(filename).convert('RGB')
            cropped = image.crop((xmin, ymin, xmax, ymax))  # (left, upper, right, lower)

            # Preprocesar
            if MODEL == "CLIP":
                image_tensor = preprocess_clip(cropped).unsqueeze(0).to(device)
                features[i, :] = model.encode_image(image_tensor).cpu().squeeze().numpy()
            elif MODEL == "DINO":
                image_tensor = preprocess(cropped).unsqueeze(0).to(device)
                features[i, :] = model(image_tensor).cpu().squeeze().numpy()
            else:
                image_tensor = preprocess(cropped).unsqueeze(0).to(device)
                features[i, :] = model(image_tensor).cpu()[0, :]
            
            if i % 100 == 0:
                print('{}/{}'.format(i, n_images))   
                        
        
        os.makedirs('data', exist_ok=True)  
        feat_file = os.path.join('data', 'feat_{}_{}.npy'.format(MODEL, DATASET))
        np.save(feat_file, features)
        print('saving data ok')