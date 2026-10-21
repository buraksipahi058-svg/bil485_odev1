"""Kendi el yazısı görüntülerini MNIST formatına getirip eğitilmiş modelle tahmin eder.

Kullanım:
  # Tek fotoğrafta yan yana yazılmış numara (rakamlar otomatik ayrılır)
  python src/handwriting.py handwriting/numara.jpg --expected 23120205088

  # Klasör: her dosyada tek rakam, dosya adının ilk karakteri gerçek etiket (ör. 3_a.png)
  python src/handwriting.py handwriting/rakamlar/

MNIST ön işleme adımları (LeCun ve ark.) taklit edilir:
  1. Gri tonlama, ışık/gölge düzeltme
  2. Arka plan siyah, rakam beyaz olacak şekilde ters çevirme
  3. Rakam sınırlayıcı kutusu, en-boy oranı korunarak 20x20'ye sığdırma
  4. 28x28 tuvalde ağırlık merkezine göre ortalama
  5. Eğitimdeki ile aynı normalizasyon (mean=0.1307, std=0.3081)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from scipy import ndimage

from data import MNIST_MEAN, MNIST_STD
from model import MLP

ROOT = Path(__file__).resolve().parent.parent


def load_gray(path, max_side=1600):
    img = ImageOps.exif_transpose(Image.open(path)).convert("L")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    return np.asarray(img, dtype=np.float32) / 255.0


def ink_map(gray, invert=False):
    """Kağıt üzerindeki koyu mürekkebi 0-1 arası 'mürekkep yoğunluğu'na çevirir."""
    if invert:  # açık renkli yazı / koyu zemin
        gray = 1.0 - gray
    # büyük bulanıklaştırma ile arka plan aydınlatmasını tahmin et, gölgeyi böl
    background = ndimage.gaussian_filter(gray, sigma=max(gray.shape) / 30)
    norm = np.clip(gray / (background + 1e-6), 0, 1)
    return 1.0 - norm  # mürekkep = 1, kağıt = 0


def otsu(values):
    hist, edges = np.histogram(values, bins=256, range=(0, 1))
    p = hist / hist.sum()
    omega, mu = np.cumsum(p), np.cumsum(p * edges[:-1])
    sigma_b = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega) + 1e-12)
    return edges[np.argmax(sigma_b)]


def segment_digits(ink):
    """Bağlantılı bileşenlerle rakamları bulur.
    Soldan sağa sıralı (y0, y1, x0, x1, maske) döndürür; maske yalnızca o rakamın pikselleridir."""
    mask = ink > max(otsu(ink), 0.15)
    mask = ndimage.binary_closing(mask, iterations=2)
    labels, n = ndimage.label(mask)
    boxes = []
    for i, sl in enumerate(ndimage.find_objects(labels), 1):
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        boxes.append([sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, (labels[sl] == i).sum(), {i}])
    if not boxes:
        return []
    # gürültüyü at: rakam yüksekliğinin medyanına göre çok küçük parçalar
    med_h = np.median([b[1] - b[0] for b in boxes if b[4] > 30] or [1])
    boxes = [b for b in boxes if (b[1] - b[0]) > 0.35 * med_h and b[4] > 30]
    # yatayda büyük ölçüde örtüşen parçaları birleştir (ör. kopuk yazılmış 5, 7)
    boxes.sort(key=lambda b: b[2])
    merged = []
    for b in boxes:
        if merged:
            m = merged[-1]
            overlap = min(m[3], b[3]) - max(m[2], b[2])
            if overlap > 0.5 * min(m[3] - m[2], b[3] - b[2]):
                merged[-1] = [min(m[0], b[0]), max(m[1], b[1]), min(m[2], b[2]), max(m[3], b[3]),
                              m[4] + b[4], m[5] | b[5]]
                continue
        merged.append(b)
    # komşu rakamların kırpmaya girmemesi için her rakamın kendi bileşen maskesi
    # (biraz genişletilir ki eşiğin altında kalan kenar pikselleri kaybolmasın)
    out = []
    for b in merged:
        own = ndimage.binary_dilation(np.isin(labels, list(b[5])), iterations=3)
        out.append((b[0], b[1], b[2], b[3], own))
    return out


def to_mnist(ink_crop, thicken=0):
    """Tek rakamlık mürekkep görüntüsünü 28x28 MNIST formatına getirir."""
    crop = np.clip((ink_crop - 0.1) / 0.9, 0, 1)  # zayıf kağıt dokusunu bastır
    if thicken:
        crop = ndimage.grey_dilation(crop, size=(thicken, thicken))
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS),
                       dtype=np.float32) / 255.0
    canvas = np.zeros((28, 28), dtype=np.float32)
    top, left = (28 - nh) // 2, (28 - nw) // 2
    canvas[top:top + nh, left:left + nw] = small
    # ağırlık merkezini (14, 14)'e kaydır
    cy, cx = ndimage.center_of_mass(canvas)
    canvas = ndimage.shift(canvas, (14 - cy, 14 - cx), order=1, mode="constant")
    return np.clip(canvas, 0, 1)


def load_model(weights=None):
    weights = Path(weights or ROOT / "results/models/best.pt")
    cfg_path = weights.with_suffix(".json")
    if not cfg_path.exists():
        cfg_path = weights.parent / "best.json"
    cfg = json.loads(cfg_path.read_text())
    model = MLP(cfg["hidden"], cfg.get("dropout", 0.0))
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model, cfg


def predict(model, digits28):
    x = torch.tensor(np.stack(digits28))[:, None]
    x = (x - MNIST_MEAN) / MNIST_STD
    with torch.no_grad():
        probs = torch.softmax(model(x), 1).numpy()
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="fotoğraf dosyası ya da tek-rakam görüntüleri klasörü")
    ap.add_argument("--expected", help="gerçek rakam dizisi (tek fotoğraf modu), ör. 23120205088")
    ap.add_argument("--weights", help="model ağırlıkları (varsayılan: results/models/best.pt)")
    ap.add_argument("--invert", action="store_true", help="koyu zemin üzerine açık yazı ise")
    ap.add_argument("--thicken", type=int, default=0, help="ince kalem çizgilerini kalınlaştır (ör. 3)")
    ap.add_argument("--out", default=str(ROOT / "results/handwriting"))
    args = ap.parse_args()

    out = Path(args.out)
    (out / "processed").mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.weights)
    src = Path(args.input)

    crops, digits, labels = [], [], []
    if src.is_dir():
        for f in sorted(p for p in src.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg")):
            ink = ink_map(load_gray(f), args.invert)
            boxes = segment_digits(ink)
            if not boxes:
                print(f"  {f.name}: rakam bulunamadı, atlandı")
                continue
            # tek rakam varsayımı: en büyük kutuyu al
            y0, y1, x0, x1, own = max(boxes, key=lambda b: (b[1] - b[0]) * (b[3] - b[2]))
            crops.append((ink * own)[y0:y1, x0:x1])
            labels.append(int(f.name[0]) if f.name[0].isdigit() else None)
    else:
        ink = ink_map(load_gray(src), args.invert)
        for y0, y1, x0, x1, own in segment_digits(ink):
            pad = max(2, (y1 - y0) // 10)
            crops.append((ink * own)[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad])
        if args.expected:
            if len(args.expected) != len(crops):
                print(f"UYARI: {len(crops)} rakam bulundu, beklenen {len(args.expected)}. "
                      "Fotoğrafta rakamlar birbirine değiyor olabilir.")
            labels = [int(ch) for ch in args.expected[:len(crops)]]
            labels += [None] * (len(crops) - len(labels))
        else:
            labels = [None] * len(crops)

    if not crops:
        raise SystemExit("Hiç rakam bulunamadı.")
    digits = [to_mnist(c, args.thicken) for c in crops]
    probs = predict(model, digits)
    preds = probs.argmax(1)

    for i, d in enumerate(digits):
        Image.fromarray((d * 255).astype(np.uint8)).save(out / "processed" / f"rakam_{i:02d}.png")

    n = len(digits)
    fig, axes = plt.subplots(2, n, figsize=(1.3 * n + 1, 3.6), squeeze=False)
    for i in range(n):
        axes[0, i].imshow(crops[i], cmap="gray")
        axes[1, i].imshow(digits[i], cmap="gray")
        ok = labels[i] is None or labels[i] == preds[i]
        title = f"{preds[i]} (%{probs[i, preds[i]] * 100:.0f})"
        if labels[i] is not None:
            title = f"gerçek {labels[i]}\ntahmin {title}"
        axes[1, i].set_title(title, fontsize=8, color="green" if ok else "red")
        for ax in axes[:, i]:
            ax.axis("off")
    axes[0, 0].set_ylabel("orijinal")
    fig.suptitle(f"Üst: fotoğraftan kırpılan rakam, alt: MNIST formatı (28x28) ve tahmin — model: {cfg['name']}",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "tahminler.png", dpi=150)
    plt.close(fig)

    known = [(l, p) for l, p in zip(labels, preds) if l is not None]
    acc = sum(l == p for l, p in known) / len(known) if known else None
    summary = {
        "model": cfg["name"], "input": str(src), "n_digits": n,
        "predicted": "".join(map(str, preds)),
        "expected": "".join(str(l) for l in labels if l is not None) or None,
        "accuracy": acc,
        "per_digit": [{"index": i, "label": labels[i], "pred": int(preds[i]),
                       "confidence": float(probs[i, preds[i]]),
                       "top3": [int(k) for k in probs[i].argsort()[::-1][:3]]} for i in range(n)],
    }
    (out / "sonuc.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Tahmin : {summary['predicted']}")
    if summary["expected"]:
        print(f"Gerçek : {summary['expected']}")
        print(f"Doğruluk: {acc * 100:.1f}% ({sum(l == p for l, p in known)}/{len(known)})")
    print(f"Görsel : {out / 'tahminler.png'}")


if __name__ == "__main__":
    main()
