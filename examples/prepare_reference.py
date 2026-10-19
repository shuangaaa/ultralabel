import argparse
import numpy as np
from PIL import Image
from ultralabel.data.sampling import sample_interior, sampling_interval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Image.open(args.mask) as image:
        mask = np.asarray(image.convert("L")) > 0
    interval = sampling_interval([mask.sum()])
    points = sample_interior(mask, interval)
    np.save(args.output, points)
    print(f"Saved {len(points)} interior reference points to {args.output}")


if __name__ == "__main__":
    main()
