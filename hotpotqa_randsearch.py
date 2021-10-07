from hyperparatuner.sentdrop_randomsearch import generate_random_search_bash
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--rand_seed", type=int, default=555)
    parser.add_argument("--task_num", type=int, default=5)
    args = parser.parse_args()
    generate_random_search_bash(task_num=args.task_num, seed=args.rand_seed)