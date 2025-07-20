# This file defines the structure of a single neuron of the network

import numpy as np

neuron = {
    "synapses": synapses,
    "dendritic_tree": dendritic_tree,
    "soma": soma
}


class synapse:
    """
    PURPOSE: converts a neuron firing spike into a current
    Manages its own weight and plasticity
    """
    
    # initialize all of the attributes of a synapse
    def __init__(self, host_compartment, strength, decay_factor):
        
        # internal variables
        self.host_component = host_compartment # host neuron/dendritic tree
        self.strength = strength # initial strength of the synapse
        self.decay_factor = decay_factor # measures how fast the current decays
        self.synaptic_conductance = 0.0 # measures how conductive the synapse is to the spike
        
        # plasticity variables
        strength_plus = 0.03
        strength_minus = 0.035
        t_plus = 20.0 # max time betweeen firing to trigger a stengthening of the pathway
        t_minus = 20.0 # max time between firing to trigger a weakening of the pathway
        
        self.last_spike_t = -np.inf # time for the latest signal from the presynaptic neuron
        
    
    # tunes params for a  presynaptic spike
    def presynaptic_spike(self, t):
        
        self.synaptic_conductance += 1.0 # synapse is now active
        self.last_spike_t = t
        
        # if the post synaptic neuron fires before the presynaptic neuron, LTD occurs
        delta_t = t - self.host.neuron_last_time # time now - host fire time
        if delta_t > 0:
            self.weight -= self.strength_minus * np.exp(-delta_t / self.t_minus)
            self.weight = max(0, self.weight)
            

    # tunes params for a postsynaptic spike
    def postsynaptic_spike(self, t):
        
        # if the presynaptic neuron fires before the postsynaptic neuron, LTP occurs
        delta_t = t - self.last_spike_t # time now - hostee fire time
        if delta_t > 0:
            self.weight += self.strength_plus * np.exp(-delta_t / self.t_plus)
            self.weight = min(4.0, self.weight) # cap at 4.0
            
    
    # get the output current of the synapse
    def get_current(self):
        # I = g * w * (V_rev - V_post)
        V_rev_excitatory = 0.0  # Reversal potential for excitatory synapses
        i = self.synaptic_conductance * self.weight * (V_rev_excitatory - self.host.potential) # current intensity
        return i

    # decay the conductance(how effective the current flow is for some dt after the presynaptic spike)
    def update(self.synaptic_conductance, self.decay_factor, dt):
        self.synaptic_conductance -= (self.synaptic_conductance/self. decay_factor) * dt
        return
        