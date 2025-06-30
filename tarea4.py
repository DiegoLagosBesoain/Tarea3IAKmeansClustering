import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

# ---------------------------- MLP ----------------------------

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def train_mlp(X_train, y_train, X_val, y_val, num_classes):
    input_dim = X_train.shape[1]
    model = MLPClassifier(input_dim, num_classes)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    device = next(model.parameters()).device
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_val_tensor = X_val_tensor.to(device)

    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_val_tensor).argmax(dim=1).cpu().numpy()
    acc = accuracy_score(y_val, preds)
    return acc, preds

# ---------------------------- SVM ----------------------------

def train_svm(X_train, y_train, X_val, y_val):
    clf = SVC()  # usa kernel='rbf', C=1.0, gamma='scale' por defecto
    clf.fit(X_train, y_train)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc, preds

# ---------------------------- Random Forest ----------------------------

def train_rf(X_train, y_train, X_val, y_val):
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    return acc, preds

# ---------------------------- Métricas ----------------------------

def per_class_accuracy(y_true, y_pred, n_classes=20):
    report = classification_report(y_true, y_pred, output_dict=True)
    accs = [report[str(i)]['recall'] if str(i) in report else 0 for i in range(n_classes)]
    return accs

def plot_accuracy_bar(accs, model_name, encoder):
    plt.figure(figsize=(12, 5))
    plt.bar(range(len(accs)), accs)
    plt.xlabel("Clase")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy por clase - {model_name} con {encoder}")
    plt.xticks(range(len(accs)))
    plt.tight_layout()
    plt.savefig(f'data/acc_por_clase_{model_name}_{encoder}.png')
    plt.show()

def plot_conf_matrix(y_true, y_pred, model_name, encoder):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Matriz de confusión - {model_name} con {encoder}")
    plt.tight_layout()
    plt.savefig(f'data/conf_matrix_{model_name}_{encoder}.png')
    plt.show()

# ---------------------------- Obtener etiquetas ----------------------------

def load_labels_from_txt(split, dataset_path='VocPascal'):
    txt_file = os.path.join(dataset_path, f"{split}_voc.txt")
    with open(txt_file, "r") as file:
        lines = file.readlines()

    labels_str = [line.split('\t')[1].strip() for line in lines]

    # Crear diccionario de clases
    unique_classes = sorted(set(labels_str))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(unique_classes)}
    print("Diccionario de clases:", class_to_idx)

    labels = [class_to_idx[label] for label in labels_str]
    return np.array(labels)

# ---------------------------- Configuración ----------------------------

encoder = "resnet34"
model_type = "SVM"
dataset = "VocPascal"
n_classes = 20

# ---------------------------- Cargar representaciones y etiquetas ----------------------------

X_train = np.load(f"data/feat_{encoder}_train_{dataset}.npy")
X_val = np.load(f"data/feat_{encoder}_val_{dataset}.npy")

y_train = load_labels_from_txt("train", dataset_path=dataset)
y_val = load_labels_from_txt("val", dataset_path=dataset)

# ---------------------------- Entrenar ----------------------------

if model_type == "MLP":
    acc, preds = train_mlp(X_train, y_train, X_val, y_val, num_classes=n_classes)
elif model_type == "SVM":
    acc, preds = train_svm(X_train, y_train, X_val, y_val)
elif model_type == "RF":
    acc, preds = train_rf(X_train, y_train, X_val, y_val)
else:
    raise ValueError("Modelo no válido. Usa 'MLP', 'SVM' o 'RF'.")

print(f"\n✅ Accuracy total ({model_type} con {encoder}): {acc:.4f}")

accs = per_class_accuracy(y_val, preds, n_classes=n_classes)
plot_accuracy_bar(accs, model_type, encoder)
plot_conf_matrix(y_val, preds, model_type, encoder)
