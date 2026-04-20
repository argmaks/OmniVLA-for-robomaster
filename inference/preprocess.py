"""
Center-crop an image to a square and optionally resize it.

Usage:
    from inference.preprocess import center_crop_square
    img = center_crop_square(pil_image)          # crop to square, keep original side length
    img = center_crop_square(pil_image, size=224) # crop then resize to 224x224
"""

from PIL import Image


def center_crop_square(image: Image.Image, size: int | None = None) -> Image.Image:
    """Center-crop *image* to a square and optionally resize to *size*x*size*."""
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top  = (h - side) // 2
    image = image.crop((left, top, left + side, top + side))
    if size is not None:
        image = image.resize((size, size), Image.LANCZOS)
    return image


if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="Center-crop an image to a square.")
    parser.add_argument("input",          help="Input image path")
    parser.add_argument("output",         help="Output image path")
    parser.add_argument("--size", type=int, default=None,
                        help="Resize the square crop to SIZE x SIZE pixels")
    args = parser.parse_args()

    img = Image.open(args.input).convert("RGB")
    print(f"Input : {img.size} WxH")
    img = center_crop_square(img, size=args.size)
    print(f"Output: {img.size} WxH  →  {args.output}")
    img.save(args.output)
