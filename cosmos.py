import numpy as np
from typing import Dict, List, Tuple, Optional

class COSMOS:
    """
    COSMOS (Coordinated Optimization for Slicing with Multiple Objective Scheduling)
    Inter-slice resource allocation algorithm using primal-dual optimization
    """
    def __init__(
        self, 
        total_rbs: int,
        num_slices: int,
        weights: Dict[str, float],
        learning_rate: float = 0.01,
        max_iter: int = 100,
        eps: float = 1e-6,
        alpha: float = 1.0
    ):
        """
        Initialize COSMOS optimizer
        
        Args:
            total_rbs: Total number of resource blocks available
            num_slices: Number of network slices
            weights: Dictionary of weights for different objectives per slice
            learning_rate: Step size for gradient updates
            max_iter: Maximum number of primal-dual iterations
            eps: Small constant for numerical stability
            alpha: Packet loss surrogate sharpness parameter
        """
        self.total_rbs = total_rbs
        self.num_slices = num_slices
        self.weights = weights
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.eps = eps
        self.alpha = alpha
        self.lambda_dual = 0.0
        
    def throughput_surrogate(self, rb: float, se: float) -> float:
        """
        Throughput surrogate function using log-sum-exp
        
        Args:
            rb: Number of resource blocks
            se: Spectral efficiency
            
        Returns:
            Surrogate value for throughput objective
        """
        return -np.log(np.sum(np.exp(-rb * se)))
        
    def latency_surrogate(self, rb: float, queue_length: float) -> float:
        """
        Latency surrogate function using queue length and RB allocation
        
        Args:
            rb: Number of resource blocks
            queue_length: Current queue length
            
        Returns:
            Surrogate value for latency objective
        """
        return np.maximum(queue_length / (rb + self.eps), 1.0)
        
    def packet_loss_surrogate(self, rb: float) -> float:
        """
        Packet loss surrogate function using exponential
        
        Args:
            rb: Number of resource blocks
            
        Returns:
            Surrogate value for packet loss objective
        """
        return np.exp(-self.alpha * rb)
        
    def slice_objective(
        self,
        rb: float,
        se: float,
        queue_length: float,
        slice_type: str
    ) -> float:
        """
        Combined objective function for a single slice
        
        Args:
            rb: Number of resource blocks
            se: Spectral efficiency
            queue_length: Current queue length
            slice_type: Type of slice (embb, urllc, be)
            
        Returns:
            Combined objective value for the slice
        """
        w_r = self.weights[f"{slice_type}_throughput"]
        w_l = self.weights[f"{slice_type}_latency"] 
        w_p = self.weights[f"{slice_type}_packetloss"]
        
        return (
            w_r * self.throughput_surrogate(rb, se) +
            w_l * self.latency_surrogate(rb, queue_length) +
            w_p * self.packet_loss_surrogate(rb)
        )
    
    def compute_gradient(
        self,
        rb: float,
        se: float,
        queue_length: float,
        slice_type: str
    ) -> float:
        """
        Compute gradient of slice objective using finite differences
        
        Args:
            rb: Number of resource blocks
            se: Spectral efficiency
            queue_length: Current queue length
            slice_type: Type of slice
            
        Returns:
            Gradient of the objective
        """
        obj_plus = self.slice_objective(rb + self.eps, se, queue_length, slice_type)
        obj_minus = self.slice_objective(rb - self.eps, se, queue_length, slice_type)
        return (obj_plus - obj_minus) / (2 * self.eps)
        
    def optimize_slice_allocation(
        self,
        slice_types: List[str],
        spectral_efficiencies: np.ndarray,
        queue_lengths: np.ndarray,
        initial_allocation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Optimize RB allocation across slices using primal-dual method
        
        Args:
            slice_types: List of slice types
            spectral_efficiencies: Array of spectral efficiencies per slice  
            queue_lengths: Array of queue lengths per slice
            initial_allocation: Optional initial RB allocation
            
        Returns:
            Optimized RB allocation array
        """
        # Initialize RB allocation
        if initial_allocation is None:
            rb_allocation = np.ones(self.num_slices) * (self.total_rbs / self.num_slices)
        else:
            rb_allocation = initial_allocation.copy()
            
        # Reset dual variable
        self.lambda_dual = 0.0
        
        # Primal-dual iteration
        for _ in range(self.max_iter):
            # Primal update for each slice
            for s in range(self.num_slices):
                gradient = self.compute_gradient(
                    rb_allocation[s],
                    spectral_efficiencies[s],
                    queue_lengths[s],
                    slice_types[s]
                )
                rb_allocation[s] -= self.learning_rate * (gradient + self.lambda_dual)
                rb_allocation[s] = np.maximum(rb_allocation[s], 0)
            
            # Dual variable update
            violation = np.sum(rb_allocation) - self.total_rbs
            self.lambda_dual += self.learning_rate * violation
            self.lambda_dual = np.maximum(self.lambda_dual, 0)
            
            # Project onto feasible set
            if np.sum(rb_allocation) > self.total_rbs:
                rb_allocation *= self.total_rbs / np.sum(rb_allocation)
                
        return np.round(rb_allocation).astype(int)