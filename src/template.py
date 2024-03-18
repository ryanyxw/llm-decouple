import argparse


def main(args):
    print("yay!")


def parse_args():
    parser = argparse.ArgumentParser()

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)