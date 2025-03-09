import utils
import flood_model
import classical_MCMC
import quantum_MCMC
import plotter_inference_results
import plotter_climate_model

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
quantum_agent = quantum_MCMC.Quantum_MCMC(tools)
plotter_climate = plotter_climate_model.Plotter_climate_model(tools)
plotter_inf = plotter_inference_results.Plotter_inference_results(tools)

# Generate flood model simulation
climate_model_parameters = climate_model_generator.flood_model_generation()

# Plot flood model results
plotter_climate.plot_flood_model(climate_model_parameters)

# run classical MCMC
classical_MCMC_parameters =classical_agent.execute_MCMC(climate_model_generator.flood_model_generation)

# plot MCMC training results
plotter_inf.plot_classical_MCMC(classical_MCMC_parameters)

# plot classical MCMC inference
plotter_climate.plot_classical_flooding_prob(classical_MCMC_parameters)


# run quantum MCMC
quantum_MCMC_parameters = quantum_agent.run_qMCMC(tools.args.number_bits, tools.args.number_iterations)

# plot quantum MCMC inference
plotter_inf.plot_quantum_inference(quantum_MCMC_parameters)

# plot quantum
plotter_climate.plot_quantum_flooding_prob(quantum_MCMC_parameters)

# plot comparison quantum and classical inference
plotter_inf.plot_combined_inference(quantum_MCMC_parameters)