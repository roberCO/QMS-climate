import numpy as np
import jax.numpy as jnp
import numpyro.distributions as dist
from scipy.optimize import least_squares
from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator
from pulser.devices import DigitalAnalogDevice
from pulser.waveforms import InterpolatedWaveform
from tqdm import tqdm 


class Quantum_MCMC:
    
    def __init__(self, tools):
        
        self.tools = tools

    # 🔹 Function to generate binary mapping between real values (Q, H) and binary strings
    def generate_binary_mapping(self, Q_values, H_values):
        """Maps real values (Q, H) to binary strings for encoding."""
        combinations = [(q, h) for q in Q_values for h in H_values]
        n = len(combinations)
        num_bits = int(np.ceil(np.log2(n)))  # Compute the number of bits required
        bitstrings = [np.binary_repr(i, num_bits) for i in range(n)]
        return {combinations[i]: bitstrings[i] for i in range(n)}

    # 🔹 Function to compute probability of (Q, H) given the bivariate normal distribution
    def compute_probability(self, Q_value, H_value, normal_bi):
        """Computes the probability of a (Q, H) pair based on the bivariate normal distribution."""
        value = jnp.array([Q_value, H_value])
        probability = jnp.exp(normal_bi.log_prob(value))  # Convert log-probability to probability
        return float(probability)

    # 🔹 Function to estimate QUBO matrix
    def estimate_Q(self, X, f_values):
        """Estimates the QUBO matrix from the dataset using optimization."""
        n = X.shape[1]  # Number of variables
        Q_size = n * (n + 1) // 2  # Size of the upper triangular matrix

        # Define function for QUBO calculation
        def qubo_function(Q_vec, X):
            """Computes QUBO function values given a vectorized Q matrix."""
            Q = np.zeros((n, n))
            idx = np.triu_indices(n)
            Q[idx] = Q_vec
            return np.array([x @ Q @ x for x in X])

        # Define loss function for optimization
        def loss(Q_vec):
            """Loss function to minimize the difference between f(x) and x^T Q x."""
            return qubo_function(Q_vec, X) - f_values

        # Solve the optimization problem
        result = least_squares(loss, np.zeros(Q_size))

        # Reconstruct the Q matrix
        Q = np.zeros((n, n))
        idx = np.triu_indices(n)
        Q[idx] = result.x
        return Q

    # 🔹 Function to compute approximate function values using QUBO
    def approximate_f(self, X, Q):
        """Computes the approximated function values using the estimated QUBO matrix."""
        return np.array([x @ Q @ x for x in X])

    # 🔹 Function to generate values of Q and H for a given bit depth
    def generate_QH_values(self, num_bits):
        """Generates uniformly distributed random values for Q and H."""
        n = 2 ** num_bits  # Ensure values match the number of binary states
        Q_values = np.random.uniform(440, 500, n)  # Generate Q values in range
        H_values = np.random.uniform(18, 21, n)  # Generate H values in range
        return Q_values, H_values

    # 🔹 Function to compute weighted means based on bitstring counts
    def compute_weighted_means(self, counts, q_h_to_bin):
        """Computes the weighted means of Q and H based on bitstring counts."""
        total_counts = sum(counts.values())  # Total occurrences
        sum_Q, sum_H = 0, 0

        for (q, h), bitstring in q_h_to_bin.items():
            if bitstring in counts:
                sum_Q += q * counts[bitstring]
                sum_H += h * counts[bitstring]

        return sum_Q / total_counts, sum_H / total_counts  # Compute weighted averages

    # 🔹 Function to reduce the binary representation while maintaining mappings
    def reduce_bit_representation(self, counts, q_h_to_bin, new_num_bits):
        """
        Reduces the binary representation while maintaining the mapping between (Q, H) values and bitstrings.

        :param counts: Dictionary containing frequencies of each bitstring.
        :param q_h_to_bin: Original dictionary mapping (Q, H) to bitstrings.
        :param new_num_bits: New number of bits per parameter.
        :return: New dictionary with reduced-bit representation.
        """
        num_elements = 2 ** (2 * new_num_bits)  # Compute new number of elements
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)  # Sort by frequency
        selected_bitstrings = set(dict(sorted_counts[:num_elements]).keys())  # Select top elements

        # Create new mapping with reduced bit length
        filtered_q_h_to_bin = {k: v for k, v in q_h_to_bin.items() if v in selected_bitstrings}
        new_bitstrings = [np.binary_repr(i, 2 * new_num_bits) for i in range(num_elements)]
        
        return {k: new_bitstrings[i] for i, (k, _) in enumerate(filtered_q_h_to_bin.items())}
    
    '''
    Simulates a quantum algorithm to solve the QUBO problem using an adiabatic quantum computing approach.

    Parameters:
    - num_bits (int): The number of bits per parameter (used to determine qubit count).
    - estimated_q (numpy.ndarray): The estimated QUBO matrix.

    Returns:
    - count_dict (dict): Dictionary containing the sampled final state counts from the quantum emulator.
    '''
    def run_quantum_algorithm(self, num_bits, estimated_q):

        # Convert the estimated QUBO matrix into a negative form to align with minimization
        Q = -estimated_q  

        # Generate all possible binary bitstrings corresponding to the problem size
        bitstrings = [np.binary_repr(i, len(Q)) for i in range(2 ** len(Q))]
        costs = []  # List to store cost values for each bitstring

        # Compute the cost function for each bitstring (this is an exponential operation)
        for b in bitstrings:
            z = np.array(list(b), dtype=int)  # Convert bitstring to binary array
            cost = z.T @ Q @ z  # Compute cost using the QUBO matrix
            costs.append(cost)

        # 🔹 Pair each bitstring with its cost and sort them by increasing cost
        zipped = zip(bitstrings, costs)
        sort_zipped = sorted(zipped, key=lambda x: x[1])  

        # 🔹 Number of qubits (each parameter has num_bits qubits, considering two parameters Q and H)
        num_qubits = 2 * num_bits  

        # 🔹 Generate qubit positions in a circular layout
        angles = np.linspace(0, 2 * np.pi, num_qubits, endpoint=False)  # Evenly distribute qubits
        radii = 24  # Fixed radius to ensure uniform qubit placement

        # 🔹 Convert polar coordinates to Cartesian (x, y) coordinates
        x_coords = radii * np.cos(angles)
        y_coords = radii * np.sin(angles)
        coords = np.column_stack((x_coords, y_coords))  # Stack x and y into coordinate pairs

        # 🔹 Create the qubit register dictionary
        qubits = {f"q{i}": coord for i, coord in enumerate(coords)}

        # 🔹 Initialize the quantum register with qubit coordinates
        reg = Register(qubits)

        # 🔹 Define the adiabatic evolution parameters
        Omega = min(15, np.median(Q[Q > 0].flatten()))  # Select a median control parameter
        delta_0 = -5  # Initial detuning (must be negative)
        delta_f = 0.5  # Final detuning (must be positive)
        T = 8000  # Evolution time in nanoseconds (long enough for system propagation)

        # 🔹 Create the adiabatic pulse for quantum annealing
        adiabatic_pulse = Pulse(
            InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),  # Rabi frequency control
            InterpolatedWaveform(T, [delta_0, 0, delta_f]),  # Detuning control
            0,  # Phase of the pulse
        )

        # 🔹 Define the quantum sequence and declare the interaction channel
        seq = Sequence(reg, DigitalAnalogDevice)
        seq.declare_channel("ising", "rydberg_global")  # Rydberg global interaction
        seq.add(adiabatic_pulse, "ising")  # Add pulse to the system

        # 🔹 Simulate the quantum process using QuTiP emulator
        simul = QutipEmulator.from_sequence(seq)
        results = simul.run()

        # 🔹 Retrieve the final quantum state
        final = results.get_final_state()

        # 🔹 Sample the final state to obtain bitstring counts
        count_dict = results.sample_final_state()

        return count_dict  # Return the dictionary of bitstring frequencies

    def run_qMCMC(self, num_bits_max, num_repetitions):
        
        # Initialize Q_t (discharge) and H_t (water level) arrays
        Q_t = jnp.zeros(self.tools.get_config_variable('T')).at[0].set(self.tools.get_config_variable('Q_obs'))
        H_t = jnp.zeros(self.tools.get_config_variable('T')).at[0].set(self.tools.get_config_variable('H_obs'))

        # Simulate the evolution of Q and H over time
        for t in range(1, self.tools.get_config_variable('T')):
            rain = self.tools.get_config_variable('R_prev')[t]  # Rainfall at time step t
            # Compute new discharge considering rainfall contribution and drainage
            Q_t = Q_t.at[t].set(Q_t[t-1] + self.tools.get_config_variable('beta_Q') * rain - self.tools.get_config_variable('alpha_Q') * Q_t[t-1])
            # Compute new water level based on discharge variation and drainage
            H_t = H_t.at[t].set(H_t[t-1] + self.tools.get_config_variable('beta_H') * (Q_t[t] - Q_t[t-1]) - self.tools.get_config_variable('alpha_H') * H_t[t-1])

        # Compute statistical parameters for bivariate normal distribution
        Q_mean = 0.7 * jnp.max(Q_t) + 0.3 * jnp.mean(Q_t)  # Weighted mean for Q
        H_mean = 0.7 * jnp.max(H_t) + 0.3 * jnp.mean(H_t)  # Weighted mean for H
        sigma_Q = jnp.maximum(5, 0.1 * jnp.std(Q_t))  # Ensure minimum variance for Q
        sigma_H = jnp.maximum(0.5, 0.05 * jnp.std(H_t))  # Ensure minimum variance for H
        cov_QH = 0.5 * sigma_Q * sigma_H  # Assumed covariance between Q and H
        cov_matrix = jnp.array([[sigma_Q**2, cov_QH], [cov_QH, sigma_H**2]])  # Covariance matrix
        mean_vector = jnp.array([Q_mean, H_mean])

        # Define the bivariate normal distribution
        normal_bi = dist.MultivariateNormal(mean_vector, cov_matrix)
        
        # 🔹 Lists to store the mean values of Q (discharge) and H (water level)
        mean_Q_list = []
        mean_H_list = []

        # 🔹 Main simulation loop
        for _ in tqdm(range(num_repetitions), desc="Running simulations"):

            # 🔹 Generate random values for Q and H based on the specified number of bits
            Q_values, H_values = self.generate_QH_values(num_bits_max)

            # 🔹 Create a binary mapping from (Q, H) values to bitstrings
            q_h_to_bin = self.generate_binary_mapping(Q_values, H_values)

            # 🔹 Construct dataset in the format X (binary inputs) and f_values (objective function)
            X = np.array([list(map(int, q_h_to_bin[(q, h)])) for (q, h) in q_h_to_bin.keys()])
            f_values = 10000 * np.array([self.compute_probability(q, h, normal_bi) for (q, h) in q_h_to_bin.keys()])

            # 🔹 Estimate the initial QUBO matrix for optimization
            Q_estimada = self.estimate_Q(X, f_values)

            # 🔹 Iteratively reduce the bit representation in each loop
            for num_bits in range(num_bits_max, 0, -1):
                
                # 🔹 Execute the quantum algorithm to optimize the QUBO problem
                counts = self.run_quantum_algorithm(num_bits, Q_estimada)

                # 🔹 Reduce bit representation for the next iteration (except for the last step)
                if num_bits > 1:
                    q_h_to_bin = self.reduce_bit_representation(counts, q_h_to_bin, num_bits - 1)  # Reduce to (num_bits - 1)

                    # 🔹 Update dataset X and function values f_values with reduced representation
                    X = np.array([list(map(int, q_h_to_bin[(q, h)])) for (q, h) in q_h_to_bin.keys()])
                    f_values = 10000 * np.array([self.compute_probability(q, h, normal_bi) for (q, h) in q_h_to_bin.keys()])

                    # 🔹 Recalculate the QUBO matrix with the updated dataset
                    Q_estimada = self.estimate_Q(X, f_values)

                # 🔹 Compute the weighted mean values for Q and H at the final iteration
                if num_bits == 1:
                    mean_Q, mean_H = self.compute_weighted_means(counts, q_h_to_bin)

            # 🔹 Store the computed mean values for Q and H after each repetition
            mean_Q_list.append(mean_Q)
            mean_H_list.append(mean_H)

        # Convert lists to NumPy arrays for analysis
        mean_Q_array = np.array(mean_Q_list)
        mean_H_array = np.array(mean_H_list)

        # Compute boolean arrays for exceeding thresholds
        P_Q_exceed = mean_Q_array > self.tools.get_config_variable('umbral_Q')  # Boolean mask for Q > threshold
        P_H_exceed = mean_H_array > self.tools.get_config_variable('umbral_H')  # Boolean mask for H > threshold

        # Compute probabilities of exceeding thresholds
        P_Q_prob = np.mean(P_Q_exceed)  # Probability of exceeding discharge threshold
        P_H_prob = np.mean(P_H_exceed)  # Probability of exceeding water level threshold

        # 🔹 Print results
        print(f"📌 Probability of extreme discharge P(Q > {self.tools.get_config_variable('umbral_Q')}): {P_Q_prob:.4f}")
        print(f"📌 Probability of extreme water level P(H > {self.tools.get_config_variable('umbral_H')}): {P_H_prob:.4f}")

        # Stack Q_final and H_final into a single array for saving
        data_to_save = np.column_stack((mean_Q_array, mean_H_array))

        # Save the data as a text file with column headers
        np.savetxt(
            self.tools.get_config_variable('data_save_path')+"QH_quantum_data.txt", data_to_save, fmt="%.4f", delimiter="\t",
            header="Q_final (m³/s)\tH_final (m)", comments=""
        )

        print(f"📄 Data saved successfully at: {self.tools.get_config_variable('data_save_path')+"QH_quantum_data.txt"}")

        quantum_MCMC_parameters = {
            'mean_Q_array': mean_Q_array,
            'mean_H_array': mean_H_array
        }

        return quantum_MCMC_parameters