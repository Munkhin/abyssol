"""
!NOTE new infrastructure:
- environment stimulus detection
- predictive event queue
- network of vectorized neurons: sets of trait arrays across the synapses and neurons
"""

import numpy as np
import heapq


# --- Detect the stimulus --- #
def detect_stimulus()
# --- Manage event queue asynchronously with the model updating --- #
event_counter = 0
event_queue = []


def create_event(time, event_type, data):
    global event_counter
    event_counter += 1
    new_event = (time, event_counter, event_type, data)
    heapq.heappush(event_queue, new_event)
    return
 

class Abyssol:
    """
    Manages the entire simulation state for a population of neurons using a
    Structure of Arrays (SoA) data model.
    
    This class pre-allocates large NumPy arrays for all component properties
    to ensure high-performance, vectorized computation and avoid dynamic resizing.
    """
    
    def __init__(self, dt: float,
                 # --- Capacity Parameters --- #
                 # These define the maximum size of the pre-allocated arrays.
                 max_neurons: int = 100,
                 avg_comps_per_neuron: int = 20,
                 avg_syns_per_neuron: int = 1000):
        
        print("--- Initializing Network ---")
        self.dt = dt
        
        # --- 1. Calculate Total Capacities --- #
        # Determine the total size needed for the global arrays.
        self.capacity_neurons = max_neurons
        self.capacity_comps = max_neurons * avg_comps_per_neuron
        self.capacity_syns = max_neurons * avg_syns_per_neuron
        
        print(f"  - Pre-allocating memory for:")
        print(f"    - Max {self.capacity_neurons} neurons")
        print(f"    - Max {self.capacity_comps} compartments")
        print(f"    - Max {self.capacity_syns} synapses")

        # --- 2. Pointers to Track Used Space --- #
        # These integers track how much of the pre-allocated space is currently in use.
        # When we add a new neuron, we increment these pointers.
        self.neuron_count = 0
        self.comp_count = 0
        self.syn_count = 0

        # --- 3. Allocate Global Property Arrays --- #
        # We initialize them with sensible default values.
        # Using float32 is more memory-efficient than the default float64.
        
        # -- Compartment Arrays (size = capacity_comps) -- #
        self.comp_voltages    = np.full(self.capacity_comps, -70.0, dtype=np.float32)
        self.comp_last_spikes = np.full(self.capacity_comps, -np.inf, dtype=np.float32)
        self.comp_Rm          = np.zeros(self.capacity_comps, dtype=np.float32)
        self.comp_Cm          = np.zeros(self.capacity_comps, dtype=np.float32)
        self.comp_Ra          = np.zeros(self.capacity_comps, dtype=np.float32)
        self.comp_thresholds  = np.zeros(self.capacity_comps, dtype=np.float32)
        # Homeostasis tracker
        self.comp_avg_firing_rate = np.zeros(self.capacity_comps, dtype=np.float32)

        # -- Synapse Arrays (size = capacity_syns) -- #
        self.syn_conductances = np.zeros(self.capacity_syns, dtype=np.float32)
        self.syn_weights      = np.zeros(self.capacity_syns, dtype=np.float32)
        self.syn_last_spikes  = np.full(self.capacity_syns, -np.inf, dtype=np.float32)
        # Masks for different synapse types
        self.syn_is_ampa_mask = np.zeros(self.capacity_syns, dtype=bool)
        self.syn_is_nmda_mask = np.zeros(self.capacity_syns, dtype=bool)
        # Pre-calculated decay factors for performance
        self.syn_decay_factors = np.zeros(self.capacity_syns, dtype=np.float32)

        # --- 4. Allocate Global Connectivity Map Arrays --- #
        # These define the graph structure of the entire network.
        
        # Maps a compartment index to its parent compartment index.
        # -1 indicates no parent (i.e., it's a soma).
        self.comp_parent_indices = np.full(self.capacity_comps, -1, dtype=np.int32)
        
        # Maps a synapse index to its host compartment index.
        self.syn_host_comp_indices = np.zeros(self.capacity_syns, dtype=np.int32)
        
        # Maps a compartment index to the neuron ID it belongs to.
        # This is useful for neuron-level analysis.
        self.comp_neuron_id_map = np.full(self.capacity_comps, -1, dtype=np.int32)
        
        # Maps a synapse index to the neuron ID it belongs to.
        self.syn_neuron_id_map = np.full(self.capacity_syns, -1, dtype=np.int32)

        # --- 5. High-Level Neuromodulation State ---
        self.global_dopamine = 1.0 # Example neuromodulator level

        print("--- Network Initialized Successfully ---")

    
    def add_neuron(self, comp_params, syn_host_comp_indices, syn_params, comp_neuron_ids, syn_neuron_ids):
        # Add a neuron into the arrays
        pass
    
    
    def add_synapse(self, syn_host_comp_index, syn_params, syn_neuron_id):
        # Add a synapse into the array
        pass
    
    def handle_event(self, event):
        pass