import argparse
from polyspace.data import HMDB51Dataset, extract_features


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--split', default='train')
    p.add_argument('--out', required=True)
    p.add_argument('--frames', type=int, default=32)
    p.add_argument('--size', type=int, default=224)
    args = p.parse_args()

    ds = HMDB51Dataset(args.root, split=args.split)
    extract_features(ds, out_dir=args.out, frame_count=args.frames, size=args.size)


if __name__ == '__main__':
    main()
