import json
import argparse

class Utils:

    def __init__(self, config_path=''):

        if config_path != '':
            try:
                f = open(config_path)
                f.close()
            except IOError:
                print('<!> Info: No configuration file')
                raise Exception('It is necessary to create a configuration file (.json) for some variables')

            with open(config_path) as json_file:
                self.config_variables = json.load(json_file)

    def get_config_variable(self, variable):
        return self.config_variables[variable]

    def parse_arguments(self):

        parser = argparse.ArgumentParser(description="Module to create an input of climate model for risk analysis for qms: Example ./python3 main.py 2 1000")

        parser.add_argument("number_bits", help="number of bits to execute zoom-in/zoom-out", type=int)
        parser.add_argument("number_iterations", help="number of iterations to execute the quantum metropolis", type=int)

        self.args = parser.parse_args()

        return self.args