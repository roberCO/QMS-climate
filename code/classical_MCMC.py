import numpy as np
import jax.random as random
from numpyro.infer import MCMC, NUTS


class Classical_MCMC:
    
    def __init__(self):
        pass
    
    def execute_MCMC(self, climate_model):
        
        # MCMC Inference
        rng_key = random.PRNGKey(0)
        nuts_kernel = NUTS(climate_model)
        mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000)
        mcmc.run(rng_key, self.tools.get_config_variable('R_prev'), self.tools.get_config_variable('Q_obs'), self.tools.get_config_variable('H_obs'))
        
        # Extract posterior samples
        posterior_samples = mcmc.get_samples()
        Q_final = posterior_samples["QH_final"][:, 0]  # First component (discharge)
        H_final = posterior_samples["QH_final"][:, 1]  # Second component (water level)
        P_Q_posterior = posterior_samples["P_Q"]  # Probability of Q > threshold
        P_H_posterior = posterior_samples["P_H"]  # Probability of H > threshold
        
        # Stack Q_final and H_final into a single array for saving
        data_to_save = np.column_stack((Q_final, H_final))

        # Save the data as a text file with column headers
        np.savetxt(
            self.tools.get_config_variable('data_save_path')+'QH_classical_data.txt', data_to_save, fmt="%.4f", delimiter="\t",
            header="Q_final (m³/s)\tH_final (m)", comments=""
        )

        print(f"📄 Data saved successfully at: {self.tools.get_config_variable('data_save_path')+'QH_classical_data.txt'}")

        classical_MCMC_parameters = {
            'Q_final': Q_final,
            'H_final': H_final,
            'P_Q_posterior': P_Q_posterior,
            'P_H_posterior': P_H_posterior
        }
        
        return classical_MCMC_parameters
