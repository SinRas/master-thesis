"""Weight Updater is in this part.

Weight assigning receives:
- loss vector
- concept_drift vector
then updates weights inside itself,
then it can be asked to return weights.

Weight Updater is connected to `changedetection` and also receives `loss_vector`
from `forecaster`.
"""

# Modules
from copy import deepcopy # Deep Copy of objects
import numpy as np # Numpy


# Globals


# Methods


# Classes
## Base
class BaseWeightUpdater:
    """Base Weight Updater class and it's functionalities.
    A fully functional BaseWeightUpdater class should implement the following:

    - memory : memory of reduced losses (based on change_status)
    - weight_repr_vector : representation of weight vector. this is to be
        transformed before being interpreted as weight
    - weight_vector( normalize = False ) : compute the weight_vector from the stored repr.
    - reset() : Reset/Initialize the parameters
    - update( loss_vector, change_status ) : update the weight_repr_vector based
        on arrived information.
    """
    # Constructor
    def __init__( self, N, max_memory = np.inf ):
        """Default Constructor.

        N : number of weights to store.
        """
        # Parameters
        ## Max Memory Length
        self.max_memory = max_memory
        ## Number of Weights
        assert isinstance(N, int), "WeightUpdater: BaseWeightUpdater: N must be an integer."
        assert N > 0, "WeightUpdater: BaseWeightUpdater: N must be an greater than 0."
        self.N = N
        # Reset
        self.reset()
        # Return
        return
    # Reset
    def reset( self ):
        """(Re-)Initialize the parameters.
        """
        # Parameters
        self.weight_repr_vector = None
        self.memory = []
        # Return
        return
    # Weight Vector
    def weight_vector( self, normalize = False ):
        """Transform the Representation of Weights into weight_vector.
        """
        # Return
        return
    # Update
    def update( self, loss_vector, change_status ):
        """Update the weights based on the loss vector and change status.

        loss_vector   : vector of losses incurred by agents
        change_status : change status vector as reported by changedetection module.
        """
        # Return
        return

## ExponentiallyDynamic
class ExpWM( BaseWeightUpdater ):
    """Update the (real) weights based on Exponential Weighting.

    Each time an expert makes a mistake, it is scaled down by some constant factor.
    Representation is Stored as sum of coefficients of each exponent.
    w(i) = exp( sum(losses_by_expert) * -beta ) => repr(i) = sum(losses_by_expert)
    """
    # Constructor
    def __init__( self, N, beta, gamma = 0.5, max_memory = np.inf ):
        """Constructor for ExpWM.

        N    : number of agents to handle
        beta : decaying factor.
        gamma: lingering effect on negative/positive changes detected.
        """
        # Parameters
        ## Lingering Effect
        self.gamma = gamma
        ## Decaying Factor
        assert beta >= 0, "WeightUpdater: ExpWM: beta must be a non-negative real number."
        self.beta = beta
        # Super
        super().__init__( N = N, max_memory = max_memory )
        # Return
        return

    # Reset
    def reset( self ):
        """Reset the weight representation vector.
        """
        # Super
        super().reset()
        # Parameters
        self.weight_repr_vector = np.zeros( self.N )
        # Return
        return

    # Weight Vector
    def weight_vector( self, normalize = False ):
        """Calculate the weight vector.

        weight_vector = exp( beta * weight_repr_vector )
        """
        # Offset
        weight_repr_vector = deepcopy( self.weight_repr_vector ) - np.min( self.weight_repr_vector )
        # Weights
        weight_vector = np.exp( -self.beta * weight_repr_vector )
        if( normalize ):
            weight_vector /= np.sum( weight_vector )
        # Return
        return( weight_vector )

    # Update
    def update( self, loss_vector, change_status ):
        """Update the weights based on the loss vector and change status.

        loss_vector   : vector of losses incurred by agents
        change_status : change status vector as reported by changedetection module.
        """
        # Update Representation
        repr_delta = loss_vector - change_status * self.gamma
        # Add to Memory
        self.memory.append( deepcopy(repr_delta) )
        ## Pop if required
        if( len(self.memory) > self.max_memory ):
            self.memory.pop(0)
        # Update Repr
        self.weight_repr_vector = np.sum( self.memory, axis = 0 )
        # Return
        return
































