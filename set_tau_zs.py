import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Lyman-alpha optical depth maps along lightcone LOS."
    )
    parser.add_argument(
        "num_zs",
        type=int,
        help="Number of z0s"
    )
    parser.add_argument(
        "--z_min",
        type=float,
        default=None,
        help="Minimum z"
    )
    parser.add_argument(
        "--z_max", 
        type=float, 
        default=None,
        help="Maximum z."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    num_zs = args.num_zs
    z_min_arg = args.z_min
    z_max_arg = args.z_max
    z_min = 6
    if z_min_arg is not None:
        z_min = z_min_arg
    z_max = 13
    if z_max_arg is not None:
        z_max = z_max_arg
    zs = np.linspace(z_min, z_max, num_zs)
    np.savetxt('z0_list.txt', zs)

if __name__ == "__main__":
    main()