import torch
from torchvision import transforms, models
import numpy as np
from PIL import Image
import os
import clip

def extract_features(model_name="CLIP", split="val", dataset="VocPascal"):
    data_dir = dataset
    image_dir = os.path.join(data_dir, 'JPEGImages')
    list_file = os.path.join(data_dir, f'{split}_voc.txt')
    
    with open(list_file, "r") as file: 
        files = [f.split('\t') for f in file]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Preprocesamiento
    preprocess_std = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])

    dim = None
    model = None

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True).to(device)
        model.fc = torch.nn.Identity()
        dim = 512

    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True).to(device)
        model.fc = torch.nn.Identity()
        dim = 512

    elif model_name == 'DINO':
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
        dim = 384

    elif model_name == 'CLIP':
        model, preprocess_clip = clip.load("ViT-B/32", device=device)
        dim = 512

    else:
        raise ValueError("Modelo no soportado")

    model.eval()

    with torch.no_grad():
        n_images = len(files)
        features = np.zeros((n_images, dim), dtype=np.float32)

        for i, file in enumerate(files):
            img_name = file[0].strip()
            label = file[1].strip()
            xmax, xmin, ymax, ymin = map(int, file[2:6])
            xmin, ymin, xmax, ymax = min(xmin, xmax), min(ymin, ymax), max(xmin, xmax), max(ymin, ymax)

            filename = os.path.join(image_dir, img_name + ".jpg")

            try:
                image = Image.open(filename).convert('RGB')
            except FileNotFoundError:
                print(f"Imagen no encontrada: {filename}")
                continue

            cropped = image.crop((xmin, ymin, xmax, ymax))

            if model_name == "CLIP":
                image_tensor = preprocess_clip(cropped).unsqueeze(0).to(device)
                features[i, :] = model.encode_image(image_tensor).cpu().squeeze().numpy()

            else:
                image_tensor = preprocess_std(cropped).unsqueeze(0).to(device)
                output = model(image_tensor)
                if isinstance(output, tuple):  # Por si retorna varios outputs (como algunos DINO)
                    output = output[0]
                features[i, :] = output.cpu().squeeze().numpy()

            if i % 100 == 0:
                print(f'Procesado {i}/{n_images} imágenes...')

    os.makedirs('data', exist_ok=True)
    feat_file = os.path.join('data', f'feat_{model_name}_{split}_{dataset}.npy')
    np.save(feat_file, features)
    print(f'✅ Representaciones guardadas en {feat_file}')

#extract_features("CLIP","train","VocPascal")
#extract_features("CLIP","val","VocPascal")
#extract_features("DINO","train","VocPascal")
#extract_features("DINO","val","VocPascal")
#extract_features("resnet18","train","VocPascal")
#extract_features("resnet18","val","VocPascal")
extract_features("resnet34","train","VocPascal")
extract_features("resnet34","val","VocPascal")


