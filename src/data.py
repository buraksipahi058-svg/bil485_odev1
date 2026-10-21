"""MNIST veri setinin yüklenmesi ve train / validation / test ayrımı."""
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# MNIST eğitim setinin piksel ortalaması ve standart sapması (normalizasyon için)
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),                       # [0, 255] -> [0, 1]
    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
])


def get_datasets(root="data", val_size=10_000, seed=42):
    """60.000'lik resmi eğitim setini 50.000 train + 10.000 validation olarak ayırır.
    Resmi 10.000'lik test seti ayrı tutulur ve model geliştirmede kullanılmaz."""
    full_train = datasets.MNIST(root, train=True, download=True, transform=TRANSFORM)
    test = datasets.MNIST(root, train=False, download=True, transform=TRANSFORM)
    generator = torch.Generator().manual_seed(seed)  # her çalıştırmada aynı ayrım
    train, val = random_split(full_train, [len(full_train) - val_size, val_size], generator=generator)
    return train, val, test


def get_loaders(batch_size=64, root="data", seed=42):
    train, val, test = get_datasets(root, seed=seed)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True,
                   generator=torch.Generator().manual_seed(seed)),
        DataLoader(val, batch_size=1000),
        DataLoader(test, batch_size=1000),
    )
