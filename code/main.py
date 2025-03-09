import utils
import flood_model
import classical_MCMC
import quantum_MCMC
from plotter import plotter_inference_results, plotter_climate_model

print('\n#########################################################################################')
print('##                                    QMS Climate                                    ##')
print('##                                                                                     ##')
print('##      Hybrid quantum-classical algorithm of bayesian inference for climate risk analysis    ##')
print('#########################################################################################\n')


#Read config file with the QFold configuration variables
config_path = './config/config.json'
tools = utils.Utils(config_path)

args = tools.parse_arguments()

climate_model_generator = flood_model.Flood_model(tools)
classical_agent = classical_MCMC.Classical_MCMC()
quantum_agent = quantum_MCMC.Quantum_MCMC()
plotter_climate = plotter_climate_model.Plotter_climate_model(tools)
plotter_inf = plotter_inference_results.Plotter_inference_results(tools)

# Generate flood model simulation
climate_model = climate_model_generator.flood_model_generation()

# Plot flood model results
plotter_climate.plot_flood_model()

# run classical MCMC
classical_agent.execute_MCMC(climate_model)

# plot MCMC training results
plotter_inf.plot_classical_MCMC()

# plot classical MCMC inference
plotter_climate.plot_classical_flooding_prob()


# run quantum MCMC
quantum_agent.run_quantum_algorithm()

# plot quantum MCMC inference
plotter_inf.plot_quantum_inference()

# plot quantum
plotter_climate.plot_quantum_flooding_prob()

# plot comparison quantum and classical inference
plotter_inf.plot_combined_inference()