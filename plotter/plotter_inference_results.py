import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

class Plotter_inference_results:

    def __init__(self, tools):

        self.tools = tools

    def plot_classical_MCMC(self, classical_MCMC_parameters):
        
        Q_final = classical_MCMC_parameters['Q_final']
        H_final = classical_MCMC_parameters['H_final']
        P_Q_posterior = classical_MCMC_parameters['P_Q_posterior']
        P_H_posterior = classical_MCMC_parameters['P_H_posterior']

        # Visualizing posterior distributions
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram for discharge
        ax[0].hist(Q_final, bins=40, density=True, alpha=0.6, color="b")
        ax[0].axvline(self.tools.get_config_variable('umbral_Q'), color="r", linestyle="--", label="Flooding threshold")
        ax[0].set_title("Classical Posterior Distribution of Q")
        ax[0].set_xlabel("Discharge (m³/s)")
        ax[0].set_ylabel("Probability density")
        ax[0].legend()

        # Histogram for water level
        ax[1].hist(H_final, bins=40, density=True, alpha=0.6, color="g")
        ax[1].axvline(self.tools.get_config_variable('umbral_H'), color="r", linestyle="--", label="Flooding threshold")
        ax[1].set_title("Classical Posterior Distribution of H")
        ax[1].set_xlabel("Water Level (m)")
        ax[1].set_ylabel("Probability density")
        ax[1].legend()

        plt.savefig('./results/marginal_classical_MCMC.png', dpi=300, bbox_inches="tight")
        plt.show()

        # Print results
        print(f"📌 Probability of extreme discharge P(Q > {self.tools.get_config_variable('umbral_Q')}): {jnp.mean(P_Q_posterior):.4f}")
        print(f"📌 Probability of extreme water level P(H > {self.tools.get_config_variable('umbral_H')}): {jnp.mean(P_H_posterior):.4f}")
    
    # Function to plot bitstring distributions
    def plot_distribution(self, C, indexes):
        """Plots the distribution of bitstrings with color differentiation."""
        C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
        color_dict = {key: "r" if key in indexes else "g" for key in C}

        plt.figure(figsize=(12, 6))
        plt.xlabel("Bitstrings")
        plt.ylabel("Counts")
        plt.bar(C.keys(), C.values(), width=0.5, color=color_dict.values())
        plt.xticks(rotation="vertical")
        plt.show()

    def plot_quantum_inference(self, quantum_MCMC_parameters):
        
        mean_Q_array = quantum_MCMC_parameters['mean_Q_array']
        mean_H_array = quantum_MCMC_parameters['mean_H_array']
        
        # 🔹 Create subplots for visualizing the posterior distributions of Q (discharge) and H (water level)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))  # Two plots side by side

        # 🔹 Histogram for discharge (Q)
        ax[0].hist(mean_Q_array, bins=30, density=True, alpha=0.6, color="b")  # Plot histogram with density normalization
        ax[0].axvline(self.tools.get_config_variable('umbral_Q'), color="r", linestyle="--", label="Flooding threshold")  # Add threshold line
        ax[0].set_title("Quantum Posterior Distribution of Q")  # Set plot title
        ax[0].set_xlabel("Discharge (m³/s)")  # X-axis label
        ax[0].set_ylabel("Probability Density")  # Y-axis label
        ax[0].legend()  # Display legend

        # 🔹 Histogram for water level (H)
        ax[1].hist(mean_H_array, bins=30, density=True, alpha=0.6, color="g")  # Plot histogram with density normalization
        ax[1].axvline(self.tools.get_config_variable('umbral_H'), color="r", linestyle="--", label="Flooding threshold")  # Add threshold line
        ax[1].set_title("Quantum Posterior Distribution of H")  # Set plot title
        ax[1].set_xlabel("Water Level (m)")  # X-axis label
        ax[1].set_ylabel("Probability Density")  # Y-axis label
        ax[1].legend()  # Display legend

        # 🔹 Save the plot as a high-resolution image
        plt.savefig('./results/marginal_quantum_MCMC.png', dpi=300, bbox_inches="tight")  

        # 🔹 Display the plot
        plt.show()

    def plot_combined_inference(self, quantum_MCMC_parameters):
        
        mean_Q_array = quantum_MCMC_parameters['mean_Q_array']
        mean_H_array = quantum_MCMC_parameters['mean_H_array']

        # 🔹 Define the path to the saved classical data file
        data_file_path = "./results/QH_classical_data.txt"

        # 🔹 Load the classical Q_final and H_final data from the text file
        data_loaded = np.loadtxt(data_file_path, delimiter="\t", skiprows=1)  # Skip header row
        Q_final_classical = data_loaded[:, 0]  # First column: Classical Q_final
        H_final_classical = data_loaded[:, 1]  # Second column: Classical H_final

        # 🔹 Visualizing posterior distributions with quantum vs classical comparison
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        # 🔹 Histogram for discharge (Q) - Quantum vs Classical
        ax[0].hist(mean_Q_array, bins=30, density=True, alpha=0.6, color="b", label="Quantum")
        ax[0].hist(Q_final_classical, bins=30, density=True, alpha=0.5, color="orange", label="Classical")
        ax[0].axvline(self.tools.get_config_variable('umbral_Q'), color="r", linestyle="--", label="Flooding threshold")
        ax[0].set_title("Comparison of Posterior Distributions for Q")
        ax[0].set_xlabel("Discharge (m³/s)")
        ax[0].set_ylabel("Probability Density")
        ax[0].set_xlim(435, 500)  # 🔹 Set x-axis limits for Q
        ax[0].legend()

        # 🔹 Histogram for water level (H) - Quantum vs Classical
        ax[1].hist(mean_H_array, bins=30, density=True, alpha=0.6, color="g", label="Quantum")
        ax[1].hist(H_final_classical, bins=30, density=True, alpha=0.5, color="purple", label="Classical")
        ax[1].axvline(self.tools.get_config_variable('umbral_H'), color="r", linestyle="--", label="Flooding threshold")
        ax[1].set_title("Comparison of Posterior Distributions for H")
        ax[1].set_xlabel("Water Level (m)")
        ax[1].set_ylabel("Probability Density")
        ax[1].set_xlim(18, 21)  # 🔹 Set x-axis limits for H
        ax[1].legend()

        # 🔹 Save the overlayed comparison plot
        plt.savefig('./results/marginal_comparison_MCMC.png', dpi=300, bbox_inches="tight")
        plt.show()