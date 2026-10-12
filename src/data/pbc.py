import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]   # valores de ImageNet, que es con lo que se
STD = [0.229, 0.224, 0.225]    # preentrenó el modelo que usaremos


def get_dataloaders(data_dir, img_size=224, batch_size=32, seed=42, num_workers=0):
    # Con aumento de datos (solo para entrenar)
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # Sin aumento (validación y test)
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    train_full = datasets.ImageFolder(data_dir, transform=train_tf)
    eval_full = datasets.ImageFolder(data_dir, transform=eval_tf)

    # Split estratificado: cada subconjunto mantiene la proporción de clases
    idx = np.arange(len(train_full))
    y = np.array(train_full.targets)
    train_idx, tmp_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=0.50, stratify=y[tmp_idx], random_state=seed)

    def make(ds, indices, shuffle):
        return DataLoader(Subset(ds, indices), batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers)

    return (
        make(train_full, train_idx, True),
        make(eval_full, val_idx, False),
        make(eval_full, test_idx, False),
        train_full.classes,
    )