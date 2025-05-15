import argparse
import os

from src.Environment import EnvironmentParams, Environment

from utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_config', action='store_true', help='Enable to write default config only')
    parser.add_argument('--config', default='config/manhattanfast32.json', help='Path to config file')
    parser.add_argument('--id', default=None, help='Overrides the logfile name and the save name')

    args = parser.parse_args()

    if args.generate_config:
        generate_config(EnvironmentParams(), "config/default.json")
        exit(0)

    if args.config is None:
        print("Config file needed!")
        exit(1)

    params = read_config(args.config)

    if args.id is not None:
        params.model_stats_params.save_model = "models/" + args.id
        params.model_stats_params.log_file_name = args.id

    env = Environment(params)

    env.run()
