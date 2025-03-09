import utils

print('\n#########################################################################################')
print('##                                    QMS Climate                                    ##')
print('##                                                                                     ##')
print('##      Hybrid quantum-classical algorithm of bayesian inference for climate risk analysis    ##')
print('#########################################################################################\n')


#Read config file with the QFold configuration variables
config_path = './config/config.json'
tools = utils.Utils(config_path)

args = tools.parse_arguments()