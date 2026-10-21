"""Yapılandırılabilir Multi-Layer Perceptron (MLP). CNN kullanılmaz."""
from torch import nn


class MLP(nn.Module):
    """28x28 görüntüyü 784 boyutlu vektöre düzleştirip tam bağlantılı katmanlardan geçirir.

    hidden: gizli katman nöron sayıları, ör. [256, 128] -> 784-256-128-10
    dropout: her gizli katmandan sonra uygulanan dropout oranı (0 = kapalı)
    """

    def __init__(self, hidden=(256, 128), dropout=0.0, n_classes=10):
        super().__init__()
        layers, prev = [nn.Flatten()], 28 * 28
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))  # softmax CrossEntropyLoss içinde
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
