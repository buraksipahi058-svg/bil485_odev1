"""Telefon fotoğrafını (HEIC/JPG) el yazısı tahmini için hazırlar: döndürme ve kırpma.

Kullanım:
  python src/fotograf_hazirla.py handwriting/gelen/43128.heic handwriting/numara.jpg \
      --crop 900 950 1530 3150 --rotate -90

--crop  : sol üst sağ alt (piksel). Numaranın dışındaki zarf kenarı vb. nesneleri dışarıda bırakır.
--rotate: derece (PIL yönü, -90 = saat yönünde 90°). Dikey çekilen yazıyı yatay yapar.
"""
import argparse

from PIL import Image, ImageOps

try:  # iPhone HEIC desteği
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("SOL", "UST", "SAG", "ALT"))
    ap.add_argument("--rotate", type=float, default=0)
    args = ap.parse_args()

    img = ImageOps.exif_transpose(Image.open(args.src)).convert("RGB")
    if args.crop:
        img = img.crop(tuple(args.crop))
    if args.rotate:
        img = img.rotate(args.rotate, expand=True)
    img.save(args.dst, quality=95)
    print(f"{args.dst} kaydedildi ({img.size[0]}x{img.size[1]})")


if __name__ == "__main__":
    main()
