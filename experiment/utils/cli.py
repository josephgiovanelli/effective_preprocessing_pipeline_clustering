import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='AutoClues')

    parser.add_argument("-s", "--scenario", nargs="?", type=str, required=True,
                        help="path to the scenario to execute")

    parser.add_argument("-p", "--pipeline", nargs="+", type=str, required=True,
                        help="steps of the pipeline to execute")

    parser.add_argument("-r", "--result_path", nargs="?", type=str, required=True,
                        help="path where put the results")

    args = parser.parse_args()

    return args