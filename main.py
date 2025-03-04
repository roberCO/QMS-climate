from pulser import Pulse, Sequence, Register
#from pulser.simulation import Simulation
import numpy as np

qubit_positions = {
    'q0': (0, 0),
    'q1': (0, 1),
    'q2': (0, 2),
    'q3': (0, 3),
    'q4': (0, 4)
}
register = Register(qubit_positions)

seq = Sequence(register, device='Chadoq2')

# Definir la duración y amplitud de los pulsos
duration = 1000  # en ns
amplitude = 1.0  # en unidades arbitrarias

# Crear un pulso que corresponda a la puerta de Hadamard
hadamard_pulse = Pulse.ConstantDetuning(amplitude, 0.0, duration, phase=0.0)

# Añadir el pulso a cada qubit
for qubit in register.qubits:
    seq.add(hadamard_pulse, qubit)

sim = Simulation(seq)
results = sim.run()
statevector = results.state_vector

# Mostrar el vector de estado
print(statevector)