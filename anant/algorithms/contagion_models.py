"""
Contagion Models for Hypergraphs
================================

Implementation of various contagion and spreading models on hypergraphs,
including discrete-time and continuous-time models for epidemic spreading,
information diffusion, and collective behaviors.

Based on HyperNetX contagion algorithms but adapted for Anant's architecture.
Supports both individual and collective contagion mechanisms.
"""

import random
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional, Union, Tuple
import time


def threshold(nodes_states, threshold_value):
    """
    Threshold function for collective contagion.
    
    Parameters
    ----------
    nodes_states : dict
        Current states of nodes in an edge
    threshold_value : float
        Threshold value for activation
        
    Returns
    -------
    bool
        Whether threshold is met
    """
    if not nodes_states:
        return False
    
    infected_count = sum(1 for state in nodes_states.values() if state == 'I')
    total_count = len(nodes_states)
    
    return (infected_count / total_count) >= threshold_value


def majority_vote(nodes_states):
    """
    Majority vote function for collective contagion.
    
    Parameters
    ----------
    nodes_states : dict
        Current states of nodes in an edge
        
    Returns
    -------
    bool
        Whether majority are infected
    """
    if not nodes_states:
        return False
    
    infected_count = sum(1 for state in nodes_states.values() if state == 'I')
    total_count = len(nodes_states)
    
    return infected_count > (total_count / 2)


def individual_contagion(hg, node_states, tau, gamma, return_event_history=False):
    """
    Individual contagion model - nodes can be infected by any infected neighbor.
    
    Parameters
    ----------
    hg : Hypergraph
        The hypergraph to run contagion on
    node_states : dict
        Initial states of nodes ('S', 'I', 'R')
    tau : float
        Transmission rate
    gamma : float
        Recovery rate
    return_event_history : bool, default False
        Whether to return detailed event history
        
    Returns
    -------
    dict
        Final states of nodes, optionally with event history
    """
    current_states = node_states.copy()
    event_history = [] if return_event_history else None
    step = 0
    
    # Get node neighborhoods (edges containing each node)
    node_edges = {}
    for node in hg.nodes:
        node_edges[node] = list(hg.get_node_edges(node))
    
    changed = True
    while changed:
        changed = False
        step += 1
        new_states = current_states.copy()
        
        for node in hg.nodes:
            current_state = current_states[node]
            
            if current_state == 'S':
                # Check if any neighbor in any edge is infected
                infected_neighbors = False
                for edge in node_edges[node]:
                    edge_nodes = hg.get_edge_nodes(edge)
                    for neighbor in edge_nodes:
                        if neighbor != node and current_states[neighbor] == 'I':
                            infected_neighbors = True
                            break
                    if infected_neighbors:
                        break
                
                # Transmission with probability tau
                if infected_neighbors and random.random() < tau:
                    new_states[node] = 'I'
                    changed = True
                    if return_event_history:
                        event_history.append({
                            'step': step,
                            'node': node,
                            'transition': 'S->I',
                            'mechanism': 'individual'
                        })
            
            elif current_state == 'I':
                # Recovery with probability gamma
                if random.random() < gamma:
                    new_states[node] = 'R'
                    changed = True
                    if return_event_history:
                        event_history.append({
                            'step': step,
                            'node': node,
                            'transition': 'I->R',
                            'mechanism': 'recovery'
                        })
        
        current_states = new_states
        
        # Safety check to prevent infinite loops
        if step > 1000:
            break
    
    result = {'final_states': current_states, 'steps': step}
    if return_event_history:
        result['event_history'] = event_history
    
    return result


def collective_contagion(hg, node_states, tau, gamma, activation_function=majority_vote, return_event_history=False):
    """
    Collective contagion model - nodes are infected based on collective behavior in edges.
    
    Parameters
    ----------
    hg : Hypergraph
        The hypergraph to run contagion on
    node_states : dict
        Initial states of nodes ('S', 'I', 'R')
    tau : float
        Transmission rate
    gamma : float
        Recovery rate
    activation_function : callable
        Function to determine if an edge is "active" for transmission
    return_event_history : bool, default False
        Whether to return detailed event history
        
    Returns
    -------
    dict
        Final states of nodes, optionally with event history
    """
    current_states = node_states.copy()
    event_history = [] if return_event_history else None
    step = 0
    
    # Get edge nodes for each edge
    edge_nodes = {}
    for edge in hg.edges:
        edge_nodes[edge] = list(hg.get_edge_nodes(edge))
    
    changed = True
    while changed:
        changed = False
        step += 1
        new_states = current_states.copy()
        
        # First, determine which edges are "active" based on activation function
        active_edges = []
        for edge in hg.edges:
            nodes_in_edge = edge_nodes[edge]
            edge_states = {node: current_states[node] for node in nodes_in_edge}
            
            if activation_function(edge_states):
                active_edges.append(edge)
        
        # Then, apply contagion in active edges
        for edge in active_edges:
            nodes_in_edge = edge_nodes[edge]
            
            for node in nodes_in_edge:
                if current_states[node] == 'S' and random.random() < tau:
                    new_states[node] = 'I'
                    changed = True
                    if return_event_history:
                        event_history.append({
                            'step': step,
                            'node': node,
                            'transition': 'S->I',
                            'mechanism': 'collective',
                            'edge': edge
                        })
        
        # Apply recovery
        for node in hg.nodes:
            if current_states[node] == 'I' and random.random() < gamma:
                new_states[node] = 'R'
                changed = True
                if return_event_history:
                    event_history.append({
                        'step': step,
                        'node': node,
                        'transition': 'I->R',
                        'mechanism': 'recovery'
                    })
        
        current_states = new_states
        
        # Safety check to prevent infinite loops
        if step > 1000:
            break
    
    result = {'final_states': current_states, 'steps': step}
    if return_event_history:
        result['event_history'] = event_history
    
    return result


def discrete_SIR(hg, initial_infected=None, tau=0.1, gamma=0.1, contagion_type='individual', 
                 activation_function=majority_vote, max_steps=100, return_full_data=False):
    """
    Discrete-time SIR (Susceptible-Infected-Recovered) model on hypergraphs.
    
    Parameters
    ----------
    hg : Hypergraph
        The hypergraph to run the model on
    initial_infected : list, optional
        List of initially infected nodes. If None, randomly select 1 node
    tau : float, default 0.1
        Transmission rate
    gamma : float, default 0.1
        Recovery rate
    contagion_type : str, default 'individual'
        Type of contagion ('individual' or 'collective')
    activation_function : callable
        Function for collective contagion activation
    max_steps : int, default 100
        Maximum number of simulation steps
    return_full_data : bool, default False
        Whether to return full simulation data
        
    Returns
    -------
    dict
        Simulation results including final states and statistics
    """
    # Initialize states
    nodes = list(hg.nodes)
    if initial_infected is None:
        initial_infected = [random.choice(nodes)]
    
    node_states = {node: 'S' for node in nodes}
    for node in initial_infected:
        if node in node_states:
            node_states[node] = 'I'
    
    # Track simulation
    history = []
    step = 0
    
    current_states = node_states.copy()
    
    while step < max_steps:
        # Count states
        counts = Counter(current_states.values())
        history.append({
            'step': step,
            'S': counts.get('S', 0),
            'I': counts.get('I', 0),
            'R': counts.get('R', 0)
        })
        
        # Stop if no infected nodes
        if counts.get('I', 0) == 0:
            break
        
        # Run one step of contagion
        if contagion_type == 'individual':
            result = individual_contagion(hg, current_states, tau, gamma, return_event_history=return_full_data)
        else:
            result = collective_contagion(hg, current_states, tau, gamma, activation_function, return_full_data)
        
        current_states = result['final_states']
        step += 1
        
        # Check for convergence
        if step > 1 and history[-1]['I'] == counts.get('I', 0):
            break
    
    # Final counts
    final_counts = Counter(current_states.values())
    history.append({
        'step': step,
        'S': final_counts.get('S', 0),
        'I': final_counts.get('I', 0),
        'R': final_counts.get('R', 0)
    })
    
    result = {
        'final_states': current_states,
        'history': history,
        'total_infected': final_counts.get('I', 0) + final_counts.get('R', 0),
        'peak_infected': max(h['I'] for h in history),
        'duration': step
    }
    
    if return_full_data:
        result['full_simulation_data'] = result  # For compatibility
    
    return result


def discrete_SIS(hg, initial_infected=None, tau=0.1, gamma=0.1, contagion_type='individual',
                 activation_function=majority_vote, max_steps=100, return_full_data=False):
    """
    Discrete-time SIS (Susceptible-Infected-Susceptible) model on hypergraphs.
    
    Parameters
    ----------
    hg : Hypergraph
        The hypergraph to run the model on
    initial_infected : list, optional
        List of initially infected nodes. If None, randomly select 1 node
    tau : float, default 0.1
        Transmission rate
    gamma : float, default 0.1
        Recovery rate (back to susceptible)
    contagion_type : str, default 'individual'
        Type of contagion ('individual' or 'collective')
    activation_function : callable
        Function for collective contagion activation
    max_steps : int, default 100
        Maximum number of simulation steps
    return_full_data : bool, default False
        Whether to return full simulation data
        
    Returns
    -------
    dict
        Simulation results including final states and statistics
    """
    # Initialize states
    nodes = list(hg.nodes)
    if initial_infected is None:
        initial_infected = [random.choice(nodes)]
    
    node_states = {node: 'S' for node in nodes}
    for node in initial_infected:
        if node in node_states:
            node_states[node] = 'I'
    
    # Track simulation
    history = []
    step = 0
    
    current_states = node_states.copy()
    
    while step < max_steps:
        # Count states
        counts = Counter(current_states.values())
        history.append({
            'step': step,
            'S': counts.get('S', 0),
            'I': counts.get('I', 0)
        })
        
        # Stop if no infected nodes
        if counts.get('I', 0) == 0:
            break
        
        # SIS modification: recovery goes back to S instead of R
        new_states = current_states.copy()
        
        if contagion_type == 'individual':
            # Individual transmission
            for node in nodes:
                if current_states[node] == 'S':
                    # Check for infected neighbors
                    infected_neighbors = False
                    for edge in hg.get_node_edges(node):
                        edge_nodes = hg.get_edge_nodes(edge)
                        for neighbor in edge_nodes:
                            if neighbor != node and current_states[neighbor] == 'I':
                                infected_neighbors = True
                                break
                        if infected_neighbors:
                            break
                    
                    if infected_neighbors and random.random() < tau:
                        new_states[node] = 'I'
        
        else:
            # Collective transmission
            for edge in hg.edges:
                edge_nodes = list(hg.get_edge_nodes(edge))
                edge_states = {node: current_states[node] for node in edge_nodes}
                
                if activation_function(edge_states):
                    for node in edge_nodes:
                        if current_states[node] == 'S' and random.random() < tau:
                            new_states[node] = 'I'
        
        # Recovery to S (not R)
        for node in nodes:
            if current_states[node] == 'I' and random.random() < gamma:
                new_states[node] = 'S'
        
        current_states = new_states
        step += 1
        
        # Check for equilibrium
        if step > 10:
            recent_infected = [h['I'] for h in history[-5:]]
            if len(set(recent_infected)) == 1:  # Steady state
                break
    
    # Final counts
    final_counts = Counter(current_states.values())
    history.append({
        'step': step,
        'S': final_counts.get('S', 0),
        'I': final_counts.get('I', 0)
    })
    
    result = {
        'final_states': current_states,
        'history': history,
        'endemic_level': final_counts.get('I', 0),
        'peak_infected': max(h['I'] for h in history),
        'duration': step
    }
    
    if return_full_data:
        result['full_simulation_data'] = result  # For compatibility
    
    return result


def run_contagion_analysis(hg, model_type='SIR', num_simulations=10, **model_params):
    """
    Run multiple contagion simulations and analyze results.
    
    Parameters
    ----------
    hg : Hypergraph
        The hypergraph to analyze
    model_type : str, default 'SIR'
        Type of model ('SIR' or 'SIS')
    num_simulations : int, default 10
        Number of simulations to run
    **model_params
        Parameters to pass to the model
        
    Returns
    -------
    dict
        Analysis results including statistics across simulations
    """
    results = []
    
    for i in range(num_simulations):
        if model_type == 'SIR':
            result = discrete_SIR(hg, **model_params)
        elif model_type == 'SIS':
            result = discrete_SIS(hg, **model_params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        results.append(result)
    
    # Analyze results
    if model_type == 'SIR':
        total_infected_list = [r['total_infected'] for r in results]
        peak_infected_list = [r['peak_infected'] for r in results]
        duration_list = [r['duration'] for r in results]
        
        analysis = {
            'model_type': model_type,
            'num_simulations': num_simulations,
            'statistics': {
                'total_infected': {
                    'mean': np.mean(total_infected_list),
                    'std': np.std(total_infected_list),
                    'min': np.min(total_infected_list),
                    'max': np.max(total_infected_list)
                },
                'peak_infected': {
                    'mean': np.mean(peak_infected_list),
                    'std': np.std(peak_infected_list),
                    'min': np.min(peak_infected_list),
                    'max': np.max(peak_infected_list)
                },
                'duration': {
                    'mean': np.mean(duration_list),
                    'std': np.std(duration_list),
                    'min': np.min(duration_list),
                    'max': np.max(duration_list)
                }
            },
            'all_results': results
        }
    
    else:  # SIS
        endemic_level_list = [r['endemic_level'] for r in results]
        peak_infected_list = [r['peak_infected'] for r in results]
        
        analysis = {
            'model_type': model_type,
            'num_simulations': num_simulations,
            'statistics': {
                'endemic_level': {
                    'mean': np.mean(endemic_level_list),
                    'std': np.std(endemic_level_list),
                    'min': np.min(endemic_level_list),
                    'max': np.max(endemic_level_list)
                },
                'peak_infected': {
                    'mean': np.mean(peak_infected_list),
                    'std': np.std(peak_infected_list),
                    'min': np.min(peak_infected_list),
                    'max': np.max(peak_infected_list)
                }
            },
            'all_results': results
        }
    
    return analysis


# Export main functions
__all__ = [
    'threshold',
    'majority_vote',
    'individual_contagion',
    'collective_contagion',
    'discrete_SIR',
    'discrete_SIS',
    'run_contagion_analysis'
]