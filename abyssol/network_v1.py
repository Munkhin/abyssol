#!NOTE time complexity is roughly O(n)

import numpy as np
import math
from typing import List

# --- Part 1: Synapse Definitions ---
# The core logic for synaptic transmission and plasticity.

class BaseSynapse:
    """
    ABSTRACT BASE CLASS for a synapse.
    
    This class handles shared logic for synaptic plasticity (STDP), while
    delegating the specific current calculation and conductance decay to its
    subclasses (fastSynapse for AMPA, slowSynapse for NMDA).
    """
    
    def __init__(self, host_compartment: 'Compartment', dt: float, initial_weight: float = 1.0):
        # --- Core Connections & State ---
        self.host_compartment = host_compartment
        self.weight = initial_weight
        self.synaptic_conductance = 0.0  # g(t), dynamic state

        # --- Spike Timing History (for STDP) ---
        self.last_presynaptic_spike_time = -np.inf

        # --- Plasticity Parameters ---
        self.A_plus = 0.03   # Base learning rate for potentiation (LTP)
        self.A_minus = 0.035 # Base learning rate for depression (LTD)
        self.t_plus = 20.0   # ms, time window for LTP
        self.t_minus = 20.0  # ms, time window for LTD
        self.max_weight = 4.0

        # --- PERFORMANCE: Pre-calculate decay factors ---
        # This avoids expensive math.exp() calls in the main simulation loop.
        self.conductance_decay_factor = math.exp(-dt / self.get_decay_tau())
        self.ltd_decay_factor = math.exp(-dt / self.t_minus)
        self.ltp_decay_factor = math.exp(-dt / self.t_plus)
        
    def get_decay_tau(self) -> float:
        """Abstract method to be implemented by children to provide their specific time constant."""
        raise NotImplementedError

    def receive_presynaptic_spike(self, t: float):
        """
        Processes an incoming spike. Triggers conductance increase and checks for LTD.
        This is a primary event in the simulation.
        """
        # 1. Update conductance: Set to 1.0, will decay over time.
        self.synaptic_conductance = 1.0 
        
        # 2. Record spike time for future LTP calculations.
        self.last_presynaptic_spike_time = t
        
        # 3. Check for LTD (Post-before-Pre timing).
        delta_t = t - self.host_compartment.last_postsynaptic_spike_time
        if delta_t > 0:
            # Homeostatic scaling: Weaken less if the neuron is already quiet.
            homeostatic_factor = self.host_compartment.get_homeostatic_scaling_factor()
            
            # Decrease weight based on the STDP rule.
            delta_w = self.A_minus * homeostatic_factor * math.exp(-delta_t / self.t_minus)
            self.weight = max(0, self.weight - delta_w)

    def receive_postsynaptic_spike(self, t: float):
        """
        Processes a postsynaptic spike (from the host compartment). Triggers LTP check.
        """
        # Check for LTP (Pre-before-Post timing).
        delta_t = t - self.last_presynaptic_spike_time
        if delta_t > 0:
            # Homeostatic scaling: Strengthen less if the neuron is already overactive.
            homeostatic_factor = self.host_compartment.get_homeostatic_scaling_factor()

            # Increase weight based on the STDP rule.
            delta_w = self.A_plus * homeostatic_factor * math.exp(-delta_t / self.t_plus)
            self.weight = min(self.max_weight, self.weight + delta_w)
            
    def get_current(self) -> float:
        """Abstract method for current calculation."""
        raise NotImplementedError

    def update_conductance(self):
        """
        Updates the conductance for one time step using the pre-calculated factor.
        This is a very fast multiplication operation.
        """
        self.synaptic_conductance *= self.conductance_decay_factor


class fastSynapse(BaseSynapse):
    """
    Models a fast, AMPA-like synapse.
    Characterized by rapid decay and a linear current response.
    """
    DECAY_TAU = 5.0  # ms

    def __init__(self, host_compartment: 'Compartment', dt: float, initial_weight: float = 1.0):
        self.V_rev = 0.0  # mV, excitatory reversal potential
        super().__init__(host_compartment, dt, initial_weight)

    def get_decay_tau(self) -> float:
        return self.DECAY_TAU

    def get_current(self) -> float:
        driving_force = self.V_rev - self.host_compartment.voltage
        return self.synaptic_conductance * self.weight * driving_force


class slowSynapse(BaseSynapse):
    """
    Models a slow, NMDA-like synapse.
    Characterized by slow decay and a voltage-dependent (non-linear) current.
    """
    DECAY_TAU = 50.0  # ms

    def __init__(self, host_compartment: 'Compartment', dt: float, initial_weight: float = 1.0):
        self.V_rev = 0.0  # mV
        # --- Parameters for the Mg2+ block sigmoid ---
        self.V_half = -30.0 # mV, voltage at which block is 50% removed
        self.steepness = 0.1 # Controls how sharp the transition is
        super().__init__(host_compartment, dt, initial_weight)

    def get_decay_tau(self) -> float:
        return self.DECAY_TAU

    def _get_unblock_fraction(self) -> float:
        """
        Calculates the voltage-dependent Mg2+ unblocking fraction (B(V)).
        Returns a value between 0.0 (fully blocked) and 1.0 (fully open).
        """
        # Using a try-except block is good practice but can be slow.
        # A numerically stable sigmoid is often preferred in performance-critical code.
        exponent = -self.steepness * (self.host_compartment.voltage - self.V_half)
        return 1.0 / (1.0 + math.exp(exponent))

    def get_current(self) -> float:
        unblock_fraction = self._get_unblock_fraction()
        driving_force = self.V_rev - self.host_compartment.voltage
        effective_conductance = self.synaptic_conductance * self.weight * unblock_fraction
        return effective_conductance * driving_force


# --- Part 2: Compartment & Neuron Definitions ---
# The structural elements of the neuron model.

class Compartment:
    """
    Represents a single, electrically-distinct segment of a neuron.
    Acts as a leaky integrator based on the cable equation.
    """
    RESTING_POTENTIAL = -70.0
    SPIKE_PEAK = 40.0
    SPIKE_RESET_POTENTIAL = -75.0

    def __init__(self, compartment_id: str, dt: float, parent: 'Compartment' = None, **kwargs):
        self.id = compartment_id
        self.voltage = self.RESTING_POTENTIAL
        
        # --- Biophysical Parameters (Tunable) ---
        self.threshold = kwargs.get('threshold', -55.0)
        self.Rm = kwargs.get('membrane_resistance', 10.0)
        self.Cm = kwargs.get('membrane_capacitance', 1.0)
        self.Ra = kwargs.get('axial_resistance', 5.0)
        
        # --- Structural ---
        self.parent = parent
        self.children = []
        self.synapses = []
        
        # --- Dynamic State ---
        self.synaptic_currents_this_step = []
        self.last_postsynaptic_spike_time = -np.inf

        # --- LEARNING: Homeostatic Plasticity Mechanism ---
        self.target_firing_rate = 5.0  # Hz, desired average firing rate
        self.homeostatic_time_constant = 10000.0 # ms, how quickly it adapts
        self.low_pass_filter_alpha = dt / self.homeostatic_time_constant
        self.average_firing_rate_tracker = 0.0 # Internal tracker, in Hz

    def add_child(self, child_compartment: 'Compartment'): self.children.append(child_compartment)
    def add_synapse(self, synapse_instance: 'BaseSynapse'): self.synapses.append(synapse_instance)
    def receive_synaptic_input(self, current: float): self.synaptic_currents_this_step.append(current)

    def get_homeostatic_scaling_factor(self) -> float:
        """Provides a scaling factor to modulate learning based on activity."""
        # If firing rate is too high, return a value < 1 to suppress learning.
        # If firing rate is too low, return a value > 1 to enhance learning.
        return self.target_firing_rate / (self.average_firing_rate_tracker + 1e-9) # add small epsilon to avoid division by zero

    def update_voltage(self, dt: float):
        leak_current = (self.RESTING_POTENTIAL - self.voltage) / self.Rm
        total_synaptic_current = sum(self.synaptic_currents_this_step)
        
        axial_current = 0.0
        if self.parent: axial_current += (self.parent.voltage - self.voltage) / self.Ra
        for child in self.children: axial_current += (child.voltage - self.voltage) / self.Ra
        
        I_total = leak_current + total_synaptic_current + axial_current
        self.voltage += (I_total / self.Cm) * dt
        self.synaptic_currents_this_step.clear()

    def check_for_spike(self, t: float, dt: float) -> bool:
        spiked = False
        if self.voltage >= self.threshold:
            self.voltage = self.SPIKE_PEAK
            self.last_postsynaptic_spike_time = t
            for synapse in self.synapses: synapse.receive_postsynaptic_spike(t)
            spiked = True
        elif self.voltage == self.SPIKE_PEAK:
            self.voltage = self.SPIKE_RESET_POTENTIAL

        # Update homeostatic tracker regardless of spiking
        # Instantaneous rate is 1/dt if spiked, 0 otherwise
        instantaneous_rate = (1000.0 / dt) if spiked else 0.0
        self.average_firing_rate_tracker += self.low_pass_filter_alpha * (instantaneous_rate - self.average_firing_rate_tracker)
        
        return spiked

class Neuron:
    """A container to build and manage a neuron's compartmental structure."""
    def __init__(self, soma_params: dict, dt: float):
        self.dt = dt
        self.soma = Compartment(compartment_id='soma', dt=dt, parent=None, **soma_params)
        self.all_compartments = {'soma': self.soma}
        self.all_synapses = []

    def add_compartment(self, new_id: str, parent_id: str, comp_params: dict):
        parent_comp = self.all_compartments[parent_id]
        new_comp = Compartment(new_id, self.dt, parent=parent_comp, **comp_params)
        parent_comp.add_child(new_comp)
        self.all_compartments[new_id] = new_comp
        return new_comp

    def add_synapse_to(self, compartment_id: str, synapse_type: str, initial_weight: float = 1.0):
        comp = self.all_compartments[compartment_id]
        if synapse_type.lower() == 'ampa':
            synapse = fastSynapse(comp, self.dt, initial_weight)
        elif synapse_type.lower() == 'nmda':
            synapse = slowSynapse(comp, self.dt, initial_weight)
        else:
            raise ValueError("Synapse type must be 'ampa' or 'nmda'")
        comp.add_synapse(synapse)
        self.all_synapses.append(synapse)
        return synapse

# --- Part 3: Information Flow & Example Implementation ---

class SpikeGenerator:
    """Generates Poisson-distributed spike trains for an input source."""
    def __init__(self, average_rate_hz: float, duration_ms: float):
        self.rate = average_rate_hz
        self.duration = duration_ms
        if self.rate > 0:
            # Generate spike times based on exponential inter-spike intervals
            num_spikes = int((self.rate / 1000.0) * self.duration)
            beta = 1.0 / (self.rate / 1000.0) # Convert Hz to ms/spike
            # Using numpy is much faster for this
            inter_spike_intervals = np.random.exponential(scale=beta, size=num_spikes)
            self.spike_times = np.cumsum(inter_spike_intervals).tolist()
        else:
            self.spike_times = []

    def get_spikes_in_window(self, t_start: float, t_end: float) -> List[float]:
        """Returns spikes that should fire in the current time step."""
        # In a real-time simulation, you would just check the head of the list.
        spikes_now = [t for t in self.spike_times if t_start <= t < t_end]
        return spikes_now

def run_simulation():
    """Example of building a neuron, feeding it patterned input, and letting it learn."""
    print("--- Setting up simulation ---")
    
    # --- Simulation Parameters ---
    DT = 0.1  # ms, time step
    SIMULATION_DURATION = 5000  # ms
    
    # --- Neuron Morphology and Parameters ---
    soma_params = {'threshold': -55.0, 'membrane_resistance': 5.0, 'membrane_capacitance': 3.0}
    dend_params = {'threshold': -50.0, 'membrane_resistance': 15.0}
    
    my_neuron = Neuron(soma_params=soma_params, dt=DT)
    my_neuron.add_compartment('dend1', 'soma', dend_params)
    my_neuron.add_compartment('dend2', 'dend1', dend_params)

    # --- Information Source Setup ---
    # We will create two input pathways (A and B).
    # Pathway A will be the "target" pattern we want the neuron to learn.
    # Pathway B will be uncorrelated noise.
    
    # Create 10 input spike generators for our target pattern (Pathway A)
    pathway_A_rate = 20.0 # Hz
    pathway_A_generators = [SpikeGenerator(pathway_A_rate, SIMULATION_DURATION) for _ in range(10)]

    # Create 10 input spike generators for noise (Pathway B)
    pathway_B_rate = 5.0 # Hz
    pathway_B_generators = [SpikeGenerator(pathway_B_rate, SIMULATION_DURATION) for _ in range(10)]

    # --- Connect Inputs to Synapses ---
    # Map each generator to a new synapse on the neuron's dendrite.
    # We use a dictionary to link a generator to its synapse.
    generator_to_synapse_map = {}
    print("Connecting Pathway A (target pattern)...")
    for i, gen in enumerate(pathway_A_generators):
        # Add a synapse and store the mapping
        synapse = my_neuron.add_synapse_to('dend2', 'ampa', initial_weight=0.5)
        generator_to_synapse_map[id(gen)] = synapse
    
    print("Connecting Pathway B (noise)...")
    for i, gen in enumerate(pathway_B_generators):
        synapse = my_neuron.add_synapse_to('dend2', 'ampa', initial_weight=0.5)
        generator_to_synapse_map[id(gen)] = synapse

    all_generators = pathway_A_generators + pathway_B_generators
    
    # --- Data Logging ---
    time_points = np.arange(0, SIMULATION_DURATION, DT)
    soma_voltage_log = []
    pathway_A_weights = []
    pathway_B_weights = []

    print("\n--- Starting simulation ---")
    
    # --- Main Simulation Loop ---
    for t_idx, t in enumerate(time_points):
        # --- 1. Deliver External Spikes ---
        for gen in all_generators:
            spikes_now = gen.get_spikes_in_window(t, t + DT)
            if spikes_now:
                synapse = generator_to_synapse_map[id(gen)]
                for spike_time in spikes_now:
                    synapse.receive_presynaptic_spike(spike_time)

        # --- 2. Update Synapses ---
        for synapse in my_neuron.all_synapses:
            synapse.update_conductance()
            current = synapse.get_current()
            synapse.host_compartment.receive_synaptic_input(current)

        # --- 3. Update Compartment Voltages ---
        for comp in my_neuron.all_compartments.values():
            comp.update_voltage(DT)

        # --- 4. Check for Spikes ---
        for comp in my_neuron.all_compartments.values():
            comp.check_for_spike(t, DT)
            
        # --- 5. Log Data periodically ---
        if t_idx % 100 == 0: # Log every 10ms
            soma_voltage_log.append(my_neuron.soma.voltage)
            # Get average weight for each pathway
            avg_A_weight = np.mean([generator_to_synapse_map[id(g)].weight for g in pathway_A_generators])
            avg_B_weight = np.mean([generator_to_synapse_map[id(g)].weight for g in pathway_B_generators])
            pathway_A_weights.append(avg_A_weight)
            pathway_B_weights.append(avg_B_weight)

    print("--- Simulation finished ---")

    # --- Analysis & Visualization ---
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(soma_voltage_log, label='Soma Voltage (mV)', color='black')
    ax1.set_ylabel('Voltage (mV)')
    ax1.set_title('Neuron Activity and Synaptic Weight Evolution')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(pathway_A_weights, label='Avg. Weight of Pathway A (Target)', color='blue', linewidth=2)
    ax2.plot(pathway_B_weights, label='Avg. Weight of Pathway B (Noise)', color='red', linestyle='--')
    ax2.set_xlabel('Time Steps (x10 ms)')
    ax2.set_ylabel('Average Synaptic Weight')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# --- Run the entire process ---
if __name__ == "__main__":
    run_simulation()