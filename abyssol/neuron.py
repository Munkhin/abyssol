# This file defines the structure of a single neuron of the network

import numpy as np
from typing import Callable, List
from abc import ABC, abstractmethod

neuron = {
    "synapses": dendritic_tree,
    "soma": soma
}

# In a dendiritic tree there are two types of parts, either a synapse or a branch

import numpy as np
import math
from abc import ABC, abstractmethod


class BaseSynapse(ABC):
    """
    PURPOSE: An abstract base class for different types of synapses.
    Manages shared properties like synaptic weight and plasticity rules (STDP).
    It cannot be used directly; it must be subclassed by AMPASynapse or NMDASynapse.
    """
    
    def __init__(self, host_compartment, initial_weight: float = 1.0):
        
        # --- Internal State Variables ---
        self.host_compartment = host_compartment # The dendritic compartment it connects to
        self.weight = initial_weight             # The synapse's effective strength
        self.synaptic_conductance = 0.0          # Current level of channel opening, g(t)
        
        # --- Plasticity (STDP) Variables ---
        self.learning_rate = 0.03       # Learning rate for potentiation (LTP)
        self.forgetting_rate = 0.035     # Learning rate for depression (LTD), should be grater than learning rate for the synapse death
        self.t_plus = 20.0     # Time window for LTP (ms)
        self.t_minus = 20.0    # Time window for LTD (ms)
        self.max_weight = 4.0    # Maximum allowed weight
        
        # Time of the last spike from the presynaptic neuron connected to this synapse
        self.last_presynaptic_spike_time = -np.inf # set to negative infinity
        
    
    def receive_presynaptic_spike(self, t: float):
        """
        Called when the presynaptic neuron fires an action potential.
        This triggers a conductance increase and an LTD check.
        """
        # 1. Set conductance to maximum, ready to decay
        self.synaptic_conductance = 1.0 
        
        # 2. Record the time of this spike
        self.last_presynaptic_spike_time = t
        
        # 3. Check for LTD (Post-before-Pre)
        # We need the time the host compartment last fired a spike
        delta_t = t - self.host_compartment.last_postsynaptic_spike_time
        if delta_t > 0:
            # Weaken the synapse
            self.weight -= self.forgetting_rate * math.exp(-delta_t / self.t_minus)
            self.weight = max(0, self.weight) # Ensure weight doesn't go below zero
            

    def receive_postsynaptic_spike(self, t: float):
        """
        Called by the host_compartment when the postsynaptic neuron fires.
        This triggers an LTP check.
        """
        # Check for LTP (Pre-before-Post)
        delta_t = t - self.last_presynaptic_spike_time
        if delta_t > 0:
            # Strengthen the synapse
            self.weight += self.learning_rate * math.exp(-delta_t / self.t_plus)
            self.weight = min(self.max_weight, self.weight) # Cap the weight
            
    
    @abstractmethod
    def get_current(self) -> float:
        """
        ABSTRACT METHOD: Calculates the synaptic current.
        This MUST be implemented by child classes (AMPA, NMDA) because
        their current calculations are different.
        """
        pass

    @abstractmethod
    def update_conductance(self, dt: float):
        """
        ABSTRACT METHOD: Decays the conductance over a time step 'dt'.
        This MUST be implemented by child classes because their decay
        rates (tau) are different.
        """
        pass
        
        
class fastSynapse(BaseSynapse):
    """
    PURPOSE: the fast synapse for fast information transmission
    linear volatge calculation
    """
    def __init__(self, host_compartment, initial_weight=1.0):
        # Call the parent constructor
        super().__init__(host_compartment, initial_weight)
        
        # AMPA-specific constants
        self.V_rev = 0.0
        self.decay_t = 5.0 # fast decay

    def get_current(self):
        # AMPA's specific linear current calculation
        driving_force = self.V_rev - self.host_compartment.voltage
        return self.synaptic_conductance * self.weight * driving_force

    def update_conductance(self, dt):
        # AMPA's specific fast decay
        self.synaptic_conductance *= math.exp(-dt / self.decay_t)


class slowSynapse(BaseSynapse):
    """
    PURPOSE: slow synapse for further information gathering and memory
    sigmoid calculation of voltage.
    """
    def __init__(self, host_compartment, initial_weight=1.0):
        # Call the parent constructor
        super().__init__(host_compartment, initial_weight)
        
        # NMDA-specific constants
        self.V_rev = 0.0
        self.decay_t = 50.0 # slow decay
        # ... and V_HALF, STEEPNESS for the magnesium block

    def determine_open_chanels(self):
        """
        Determine the fraction of voltage input channels thqt are open
        Returns a value between 0.0 (fully blocked) and 1.0 (fully open).
        """
        try:
            # The sigmoid function that models the voltage dependency
            exponent = -self.STEEPNESS * (self.host_compartment.voltage - self.V_HALF)
            unblock_fraction = 1.0 / (1.0 + math.exp(exponent))
            return unblock_fraction
        except OverflowError:
            # Handles extreme negative voltages to prevent math errors
            return 0.0

    def get_current(self):
        """
        Calculates the NMDA synaptic current.
        
        This is a non-linear calculation where the conductance is first scaled
        by the voltage-dependent magnesium unblocking factor.
        """
        # Step 1: Get the voltage-gating factor (a value from 0 to 1)
        unblock_fraction = self.determine_open_chanels(self)
        
        # Step 2: Calculate the effective conductance
        # This is the key step: g_effective = g(t) * B(V) * w
        # where B(V) is the unblock_fraction
        effective_conductance = self.synaptic_conductance * self.weight * unblock_fraction
        
        # Step 3: Calculate the driving force
        driving_force = self.V_rev - self.host_compartment.voltage
        
        # Step 4: Calculate the final current using Ohm's Law (I = g * V)
        current = effective_conductance * driving_force
        
        return current

    def update_conductance(self, dt):
        # NMDA's specific slow decay
        self.synaptic_conductance *= math.exp(-dt / self.decay_t)

class compartment:
    """"
    PURPOSE: represents a dendritic branch
    """
    
    def __self__(self, compartment_id: str, location_type: str, parent=None, threshold: float):
        
        self.id = compartment_id
        self.location_type = location_type  # e.g., "soma", "proximal_dendrite", "distal_dendrite"
        self.voltage = self.RESTING_POTENTIAL
        self.threshold = threshold
        
        # Graph structure
        self.parent = parent
        self.children = []
        
        # Inputs for the current time step
        self.synaptic_inputs = []
    
    
    # add a child compartment
    def add_child(self, child_compartment):

        self.children.append(child_compartment)

    # gets the final current from the child component
    def receive_synaptic_input(self, current: float):

        self.synaptic_inputs.append(current)


    def update_voltage(self, coupling_factor=0.3, dt):
        """
        Compute the resultant current based on: 
        1. Its own synaptic inputs.
        2. Voltage flowing from its children.
        3. A "leak" back towards resting potential over some dt
        """
        
        
        

        
        
class soma:
    """
    PURPOSE: computes the activation potential, if the threshold is reached the neuron is updated.
    tunes the sensitivity of the neuron
    """
    def __self__(self, dendritic_tree):
        self.dendritic_tree = 