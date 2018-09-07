"""Agent Module aims at defining all possible 'action' prediction strategies.
This includes any type of prediciton procedure.
"""
# Modules
from copy import deepcopy # Deep Copy of objects
import numpy as np # Numpy
import talib # For Technical Indicators


# Globals


# Methods


# Classes
## Base
class BaseAgent:
    """Base Agent class and functionality.
    A fully functional `Agent` class should implement the following:

    - time : last time reported by environment
    - memory : array of previously seen states
    - receive_state( state ): receive new state and manage memory
    - action( memory = None ) : make prediction about next action
    - reset() : reset the time and memory
    """
    # Constructor
    def __init__( self ):
        """Default Constructore.
        """
        # Reset
        self.reset()
        # Return
        return
    # Reset
    def reset( self ):
        """Reset time and memory.
        """
        # Parameters
        self.time = None
        self.memory = []
        # Return
        return
    # Receive State
    def receive_state( self, state ):
        """Receive state and manage memory.
        """
        # Get Time
        self.time = state['time']
        # Store
        self.memory.append( deepcopy(state) )
        # Return
        return
    # Action
    def action( self, memory = None ):
        """Action to be done next, based on current memory.
        """
        # Return
        return

## Delayed
class Delayed( BaseAgent ):
    """Agent that predicts 'None' or the lagged state value (assuming that state_space is a subset of action_space).
    """
    # Constructor
    def __init__( self, lag = 1, transformation = None, transformation_length = None, transformation_wrap = False ):
        """Set `lag` parameter.

        lag: must be an integer greater than 0.
        """
        # Parameters
        ## Transformation
        if( (transformation is None)  ):
            self.transformation = lambda x: x
            self.transformation_length = 0
        else:
            # Check Sanity
            assert not(transformation_length is None), "When transformation is provided, transformation_length should also be provided."
            assert transformation_length >= 0, "Transformation Length must be greater than zero."
            # Assign
            self.transformation = transformation
            if( transformation_wrap ):
                self.transformation = lambda x: transformation( np.array([ entry['state'] for entry in x ]) )
            self.transformation_length = transformation_length
        ## Lag
        assert isinstance( lag, int ), "Agent: Delayed: `lag` must be simple int."
        assert lag > 0, "Agent: Delayed: lag must be greater than 0."
        self.lag = lag
        # Reset
        super().reset()
        # Return
        return

    # Receive State
    def receive_state( self, state ):
        """Store new state and pop states that won't be used later.

        state: state as reported by environment.
        """
        # Check Length
        if( len(self.memory) >= (self.lag + self.transformation_length) ):
            self.memory.pop(0)
        # Store
        super().receive_state( state )
        # Return
        return

    # Action
    def action( self, memory = None ):
        """Predict the lagged state received.

        memory: memory to be used for choosing action. Default: self.memory
        """
        # Memory
        memory = memory if not(memory is None) else self.memory
        # Check Action
        if( len(memory) < (self.lag + self.transformation_length) ):
            return
        # Action
        information = deepcopy( memory[ -(self.lag+self.transformation_length): ][ :self.transformation_length ] )
        action_next =  self.transformation( information ) # From -(lag + transformation_length) to -lag
        # Return
        return( deepcopy( action_next ) )

## Delayed_MFI
class DelayedMFI( Delayed ):
    """Wrapper for Delayed which uses MFI as transformation function with given thresholds.
    """
    # Constructor
    def __init__( self, timeperiod, lag = 1, upper_threshold = 70, lower_threshold = 30 ):
        """MFI agent will calculate the delayed MFI value acts based on:
                         70 < mfi : +1
              30 < mfi < 70       : 0
        mfi < 30                  : -1
        """
        # Defin Transformation Function
        def mfi_from_state( memory ):
            # Extract Info
            high = np.array([ entry['High'] for entry in memory ], dtype=np.float)
            low = np.array([ entry['Low'] for entry in memory ], dtype=np.float)
            close = np.array([ entry['Close'] for entry in memory ], dtype=np.float)
            volume = np.array([ entry['Volume'] for entry in memory ], dtype=np.float)
            # Calculate
            mfi = talib.MFI( high, low, close, volume, timeperiod=timeperiod )[-1]
            # Action
            action = 1 if (mfi > upper_threshold) else ( -1 if mfi < lower_threshold else 0 )
            return( action )
        # Initialize Instance
        # DEBUG notice 'timeperiod+1' is requiring one extra memory entry. this is due to the definition of `talib.MFI`
        # UPDATE: understood the problem. here the lag should be reduced by one. think about it.
        super().__init__( lag = lag, transformation = mfi_from_state, transformation_length = (timeperiod+1) )
        # Return
        return






























