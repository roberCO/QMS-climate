import numpyro
import jax.numpy as jnp
import numpyro.distributions as dist


class Flood_model:

    def __init__(self, tools):

        self.tools = tools

    # Bayesian flood model
    def flood_model_generation(self):
        """
        Simulates the evolution of discharge (Q) and water level (H) over time (in hours),
        given historical rainfall data. Uses a dynamic Bayesian approach.

        Parameters:
        - R_prev: Array of previous rainfall values (mm).
        - Q_obs: Initial observed discharge (m³/s).
        - H_obs: Initial observed water level (m).

        Outputs:
        - Plots the evolution of Q(t) and H(t) over time in hours.
        """

        # Initialize arrays for discharge (Q) and water level (H) evolution
        Q_t = jnp.zeros(self.tools.get_config_variable('T'))
        H_t = jnp.zeros(self.tools.get_config_variable('T'))
        
        # Set initial conditions
        Q_t = Q_t.at[0].set(self.tools.get_config_variable('Q_obs'))
        H_t = H_t.at[0].set(self.tools.get_config_variable('H_obs'))

        # Time-step simulation using a simple water balance model
        for t in range(1, self.tools.get_config_variable('T')):
            rainfall = self.tools.get_config_variable('R_prev')[t]  # Rainfall at time step t (hourly)
            # Update discharge: previous discharge + rainfall effect - drainage loss
            Q_t = Q_t.at[t].set(Q_t[t-1] + self.tools.get_config_variable('beta_Q') * rainfall - self.tools.get_config_variable('alpha_Q') * Q_t[t-1])
            # Update water level: previous level + effect of discharge change - drainage loss
            H_t = H_t.at[t].set(H_t[t-1] + self.tools.get_config_variable('beta_H') * (Q_t[t] - Q_t[t-1]) - self.tools.get_config_variable('alpha_H') * H_t[t-1])

        # 🔹 Compute dynamic variance based on data dispersion
        Q_mean = 0.7 * jnp.max(Q_t) + 0.3 * jnp.mean(Q_t)  # Weighted mean for Q
        H_mean = 0.7 * jnp.max(H_t) + 0.3 * jnp.mean(H_t)  # Weighted mean for H
        sigma_Q = jnp.maximum(5, 0.1 * jnp.std(Q_t))  # Ensuring minimum variance for Q
        sigma_H = jnp.maximum(0.5, 0.05 * jnp.std(H_t))  # Ensuring minimum variance for H
        
        # 🔹 Assume a covariance matrix. If Q and H are uncorrelated, covariance = 0.
        cov_QH = 0.5 * sigma_Q * sigma_H  # Adjust this based on the relationship between Q and H
        cov_matrix = jnp.array([[sigma_Q**2, cov_QH], [cov_QH, sigma_H**2]])

        # 🔹 Joint prior distribution for Q_final and H_final (2D Gaussian)
        mean_vector = jnp.array([Q_mean, H_mean])
        normal_bi = numpyro.sample("QH_final", dist.MultivariateNormal(mean_vector, cov_matrix))

        # Extract final values for Q and H
        Q_final, H_final = normal_bi[0], normal_bi[1]

        # 🔹 Compute the probability that discharge and water level exceed thresholds
        prob_Q = numpyro.deterministic("P_Q", 1 - dist.Normal(Q_final, sigma_Q).cdf(self.tools.get_config_variable('umbral_Q')))
        prob_H = numpyro.deterministic("P_H", 1 - dist.Normal(H_final, sigma_H).cdf(self.tools.get_config_variable('umbral_H')))

        climate_model_parameters = {
            'Q_t': Q_t,
            'Q_mean': Q_mean,
        }

        return climate_model_parameters