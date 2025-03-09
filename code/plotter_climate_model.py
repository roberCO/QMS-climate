import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt


class Plotter_climate_model:
    
    def __init__(self, tools):

        self.tools = tools

    def plot_flood_model(self, climate_model_parameters, show_plot=False):

        Q_t = climate_model_parameters['Q_t']
        Q_mean = climate_model_parameters['Q_mean']

        Q_t = jnp.zeros(self.tools.get_config_variable('T'))
        H_t = jnp.zeros(self.tools.get_config_variable('T'))

        # Visualization of discharge and water level evolution
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        time_hours = np.arange(self.tools.get_config_variable('T'))

        # Plot discharge evolution
        ax[0].plot(time_hours, Q_t, marker='o', color='b', linestyle='-', label='Q(t) (Discharge)')
        ax[0].axhline(Q_mean, color='r', linestyle='--', label=f'Q_mean = {Q_mean:.2f}')
        ax[0].set_title("Discharge Evolution Q(t)")
        ax[0].set_xlabel("Time (hours)")  # Updated to hours
        ax[0].set_ylabel("Discharge (m³/s)")
        ax[0].legend()
        ax[0].grid(True, linestyle='--', alpha=0.6)

        # Plot water level evolution
        ax[1].plot(time_hours, H_t, marker='s', color='g', linestyle='-', label='H(t) (Water Level)')
        ax[1].axhline(H_mean, color='r', linestyle='--', label=f'H_mean = {H_mean:.2f}')
        ax[1].set_title("Water Level Evolution H(t)")
        ax[1].set_xlabel("Time (hours)")  # Updated to hours
        ax[1].set_ylabel("Water Level (m)")
        ax[1].legend()
        ax[1].grid(True, linestyle='--', alpha=0.6)

        # Show the plots
        plt.savefig('./results/predictions_flood_model.png', dpi=300, bbox_inches="tight")
        if show_plot: plt.show()

    def plot_classical_flooding_prob(self, classical_MCMC_parameters):
        
        Q_final = classical_MCMC_parameters['Q_final']
        H_final = classical_MCMC_parameters['H_final']
        P_Q_posterior = classical_MCMC_parameters['P_Q_posterior']
        P_H_posterior = classical_MCMC_parameters['P_H_posterior']
        
        # Compute the joint probability that either Q > threshold_Q or H > threshold_H
        P_joint = (P_Q_posterior > 0.5) | (P_H_posterior > 0.5)  # Joint probability (at least one exceeds the threshold)

        # 🔹 Visualize the joint probability as a scatter plot
        fig, ax = plt.subplots(figsize=(7, 7))  # Adjusted figure size for better readability

        # Scatter plot of Q_final vs H_final with color representing joint probability
        scatter = ax.scatter(
            Q_final, H_final, c=P_joint, cmap="coolwarm", alpha=0.6, marker="o"
        )

        # Add reference threshold lines for flooding
        ax.axvline(self.tools.get_config_variable('umbral_Q'), color="r", linestyle="--", label="Flooding threshold (Q)")
        ax.axhline(self.tools.get_config_variable('umbral_H'), color="r", linestyle="--", label="Flooding threshold (H)")

        # Axis labels and title
        ax.set_xlabel("Final Discharge (Q) [m³/s]")
        ax.set_ylabel("Final Water Level (H) [m]")
        ax.set_title("Classical Joint Flooding Probability (Q > 455 or H > 20)")

        # Add legend
        ax.legend()

        # Add color bar to visualize probability scale
        cbar = fig.colorbar(scatter, ax=ax, label="Joint Probability")

        # Show the plot
        plt.savefig('./results/joint_classical_MCMC.png', dpi=300, bbox_inches="tight")
        plt.show()

        # Compute and display the mean joint probability of flooding
        P_joint_mean = jnp.mean(P_joint)
        print(f"📌 Joint probability of flooding (Q > {self.tools.get_config_variable('umbral_Q')} or H > {self.tools.get_config_variable('umbral_H')}): {P_joint_mean:.4f}")

    def plot_quantum_flooding_prob(self, quantum_MCMC_parameters):
        
        mean_Q_array = quantum_MCMC_parameters['mean_Q_array']
        mean_H_array = quantum_MCMC_parameters['mean_H_array']
        
        # 🔹 Compute the joint probability that either Q > threshold_Q or H > threshold_H
        P_joint = (mean_Q_array > self.tools.get_config_variable('umbral_Q')) | (mean_H_array > self.tools.get_config_variable('umbral_H'))  # Joint probability (at least one exceeds the threshold)

        # 🔹 Visualize the joint probability
        fig, ax = plt.subplots(figsize=(7, 7))

        # Create the scatter plot with colors indicating P_joint (0 or 1)
        scatter = ax.scatter(
            mean_Q_array, mean_H_array, c=P_joint, cmap="coolwarm", alpha=0.6, marker="o"
        )

        # Add reference threshold lines for flooding
        ax.axvline(self.tools.get_config_variable('umbral_Q'), color="r", linestyle="--", label="Flooding threshold (Q)")
        ax.axhline(self.tools.get_config_variable('umbral_H'), color="r", linestyle="--", label="Flooding threshold (H)")

        # Set axis labels and title
        ax.set_xlabel("Final Discharge (Q) [m³/s]")
        ax.set_ylabel("Final Water Level (H) [m]")
        ax.set_title("Quantum Joint Flooding Probability (Q > 455 or H > 20)")

        # 🔹 Adjust x and y axis limits
        ax.set_xlim(425, 510)  # 🔹 Set x-axis limits for Q
        ax.set_ylim(18, 21)    # 🔹 Set y-axis limits for H

        # Add legend
        ax.legend()

        # Add color bar to visualize probability scale
        cbar = plt.colorbar(scatter, ax=ax, label="Joint Probability (1 = exceeds threshold)")

        # Show the plot
        plt.savefig('./results/joint_quantum_MCMC.png', dpi=300, bbox_inches="tight")
        plt.show()

        # 🔹 Compute and display the mean joint probability of flooding
        P_joint_mean = np.mean(P_joint)
        print(f"📌 Joint probability of flooding (Q > {self.tools.get_config_variable('umbral_Q')} or H > {self.tools.get_config_variable('umbral_H')}): {P_joint_mean:.4f}")