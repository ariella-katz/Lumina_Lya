import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Lyman-alpha optical depth maps along lightcone LOS."
    )
    parser.add_argument(
        "num_z0s",
        type=int,
        help="Number of z0s"
    )
    parser.add_argument(
        "--z0_min",
        type=float,
        default=None,
        help="Minimum z"
    )
    parser.add_argument(
        "--z0_max", 
        type=float, 
        default=None,
        help="Maximum z."
    )
    parser.add_argument(
        "--spread_depth", 
        type=float, 
        default=None,
        help="Depth of z spread for pdf plot."
    )
    parser.add_argument(
        "--spread_num", 
        type=int, 
        default=None,
        help="Resolution of z spread for pdf plot."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    num_z0s = args.num_z0s
    z0_min_arg = args.z0_min
    z0_max_arg = args.z0_max
    spread_depth_arg = args.spread_depth
    spread_num_arg = args.spread_num

    z0_min = 6
    if z0_min_arg is not None:
        z0_min = z0_min_arg
    z0_max = 13
    if z0_max_arg is not None:
        z0_max = z0_max_arg
    spread_num = 1
    if ((spread_num_arg is not None) and (spread_num_arg > 0)):
        spread_num = spread_num_arg
    spread_depth = 13
    if (spread_depth_arg is not None):
        spread_depth = spread_depth_arg

    CUTOFF = 4.75

    z0_spread_list = []
    z0s = np.linspace(z0_min, z0_max, num_z0s)
    for z0 in z0s:
        z_spread = np.linspace(z0, np.max([CUTOFF, z0 - spread_depth]), spread_num)
        z0_spread_list.append(z_spread)

    with open('z0_spreads.txt', 'w') as f:
        for z_spread in z0_spread_list:
            line = ','.join(f'{z:.4f}' for z in z_spread)
            f.write(line + '\n')

if __name__ == "__main__":
    main()