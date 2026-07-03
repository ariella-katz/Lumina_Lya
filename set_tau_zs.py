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
        default=6,
        help="Minimum z"
    )
    parser.add_argument(
        "--z0_max", 
        type=float, 
        default=13,
        help="Maximum z."
    )
    parser.add_argument(
        "--spread_depth", 
        type=float, 
        default=13,
        help="Depth of z spread for pdf plot."
    )
    parser.add_argument(
        "--spread_num", 
        type=int, 
        default=1,
        help="Resolution of z spread for pdf plot."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    num_z0s = args.num_z0s
    z0_min = args.z0_min
    z0_max = args.z0_max
    spread_depth = args.spread_depth
    spread_num = args.spread_num

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