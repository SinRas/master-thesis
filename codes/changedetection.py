"""Change Detection handling.
Everything related to chage detection (individual or process) is to be implemented
here.
"""
# Modules
from scipy import stats # For Two-Sample Kolmogorov-Sminoff Test
import numpy as np # For nd-array manipulation
from copy import deepcopy # Deep Copy of objects


# Globals


# Methods


# Classes
## Base
class BaseChangeDetection:
    """Base class implementing the structor of a ChangeDetection class.
    A fully functional ChangeDetection Class should implement:

    - losses : list of loss_vectors for calculation of p-values
    - p_values : list of p_value_vectors
    - change_status : current status of changed vector
        +1 means significant improvement
         0 means no significant change
        -1 means significant worsening
    - reset() : reset everything to default value
    - update( loss_vector ) : add arrived loss_vector and calculate p_values and change_status
    - infer( losses = None ): inference function from the sequence of loss_vectors
    """
    # Constructor
    def __init__( self, calc_series_p_value, infer_from_vector ):
        """Default Constructor.

        calc_series_p_value: a function which takes a sequence of losses and
            calculate a p_value for change_has_not_happened
        infer_from_vector: a function which takes a p_value_vector and returns
            the change_status (signed detected changes).
        """
        # Check
        assert callable(calc_series_p_value), "ChangeDetection: BaseChangeDetection: calc_series_p_value should be a callable."
        assert callable(infer_from_vector), "ChangeDetection: BaseChangeDetection: infer_from_vector should be a callable."
        # Parameters
        self.calc_series_p_value = calc_series_p_value
        self.infer_from_vector = infer_from_vector
        # Reset
        self.reset()
        # Return
        return

    # Reset
    def reset( self ):
        """Reset/Define parameters to/with default values.
        """
        # Parameters
        self.change_status = None
        self.losses = []
        self.p_values = []
        # Return
        return

    # Update
    def update( self, loss_vector ):
        """Update the current status.
        """
        # Store
        self.losses.append( deepcopy(loss_vector) )
        # Infer
        change_status = self.infer()
        # Return
        return( change_status )

    # Inference Function
    def infer( self, losses = None ):
        """Infer from the losses list.
        """
        # Losses
        losses = losses if not( losses is None ) else self.losses
        # Sanity & No Update Required
        ## Sanity Check
        assert len(self.losses) >= len(self.p_values), "ChangeDetection: BaseChangeDetection: length mismatch."
        ## No Data
        if( len(self.losses) == 0 ):
            return
        ## No Update
        if( len(self.losses) == len(self.p_values) and not( self.change_status is None ) ):
            return( deepcopy( self.change_status ) )
        # Calculate P-Values
        p_value_vector = self.calc_p_values( losses )
        # Infer Based on p_value_vector
        self.change_status = self.infer_from_vector( p_value_vector )
        # Return
        return( deepcopy(self.change_status) )

    # P-Values
    def calc_p_values( self, losses = None ):
        """Calculate p_value vector based on given losses list of loss_vectors.
        """
        # Losses
        losses = losses if not( losses is None ) else self.losses
        # Empty Losses
        if( len(losses) == 0 ):
            return
        # Exctract and Calculate Series
        losses = np.array( losses )
        p_value_vector = np.zeros( losses.shape[1] )
        for i in range( len(p_value_vector) ):
            p_value_vector[i] = self.calc_series_p_value( losses[:,i] )
        # Store
        self.p_values.append( deepcopy(p_value_vector) )
        # Return
        return( p_value_vector )


## Kolmogorov-Smirnof + BenjaminHochberg
class KS_BenjaminiHochberg( BaseChangeDetection ):
    """Change Detection Instance which uses
    - Simple delayed autocorrelation.
    - Benjamini-Hochberg correction for inference from p_value_vector
    """
    # P_Adjust_BH
    # Creadit: https://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python/33532498#33532498
    @staticmethod
    def p_adjust_bh( p_value_vector ):
        """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
        p_value_vector = np.asfarray( p_value_vector )
        by_descend = p_value_vector.argsort()[::-1]
        by_orig = by_descend.argsort()
        steps = float(len(p_value_vector)) / np.arange(len(p_value_vector), 0, -1)
        q = np.minimum(1, np.minimum.accumulate(steps * p_value_vector[by_descend]))
        return q[by_orig]

    # Constructor
    def __init__( self, p_value_threshold = 0.1, windows_sizes = (5, 10) ):
        """Constructor for thresholds and values.
        """
        # Methods
        ## Benjamini-Hochberg
        def infer_from_vector( p_value_vector ):
            changed = KS_BenjaminiHochberg.p_adjust_bh( np.abs(p_value_vector) ) < p_value_threshold
            signs = np.sign( p_value_vector )
            return( signs * changed )

        ## Kolmogorov-Smirnoff
        def calc_series_p_value( loss_series ):
            # Not Enough Data
            if( sum(windows_sizes) > len(loss_series) ):
                return( 1.0 )
            # Windows
            window_recent = loss_series[ -windows_sizes[0]: ]
            window_reference = loss_series[ -sum(windows_sizes): ][ :windows_sizes[1] ]
            direction = np.sign( np.mean(window_reference) - np.mean(window_recent) ) # +1 means improving
            _,p_value = stats.ks_2samp( window_recent, window_reference )
            # Return
            return( direction * p_value )
        # Super
        super().__init__( calc_series_p_value, infer_from_vector )
        # Return
        return
































