import argparse


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration File'
    )
    return parser.parse_args()

