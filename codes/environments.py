"""Environements Module is intended to simulate environment for each run.
As in Game Theory, Reinforcement Learning, Control Theory, ... environment is
defined to be able to interact with agents.
For simplicity we currently assume that environment treats all the agents alike.
"""

# Modules
import os # Directory and file checks
import pandas as pd # Dataframe manipulation
from copy import deepcopy # Deep Copy of objects


# Globals


# Methods


# Classes
## Base
class BaseEnvironment:
    """Base Environemnet class and functionality.
    A fully functional `Environment` class should implement the following:

    - action_space : Space of all possible actions
    - actions : List of all recieved actions
    - state_space : Space of all possible states
    - current_state : Current state of environment
    - states : List of all seen states (required for `states_map`)
    - states_map( states ) : Map states to new state (map to subset of state space)
    - reset() : Reset environment
    - update( action ) : Given action from players, update the environment
    """
    # Contructor
    def __init__( self ):
        """Default constructor.
        """
        # Reset
        self.reset()
        # Return
        return
    # State Map
    def states_map( self, states ):
        """Map current states into a new state (maps new state + more information
        into a subset of state space)

        states: ordered list of states (last being current_state)
        """
        # Return
        return( deepcopy(self.current_state) )
    # Reset
    def reset( self ):
        """Reset environment to it's initial values (may be random initialization).
        """
        # Default
        ## Time
        self.time = 0
        ## States
        self.current_state = None
        self.state_space = None
        self.states = []
        ## Actions
        self.actions = []
        self.action_space = None
        # Return
        return
    # Update
    def update( self, action ):
        """Given the action and current state, update the environment state.

        action: a point in action space.
        """
        # Return
        return

## Stock: Daily
class StockDaily( BaseEnvironment ):
    """Environment for Daily stock evolutions.
    A stock price is reported daily and actions have no effect on it.

    state_space: information about the stock market at current day
        - time, state
        - high, low, close, open, typical, volume
    action_space: non-negatice real numbers (price)
    """
    # Constructor
    def __init__( self, file_path, relative = False, relative_sign = False ):
        """Initialize the environment for daily stock environment.

        file_path: path to data file to load.
        """
        # Load Data
        df = pd.read_csv( file_path, parse_dates=['Date'] )
        df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
        # Create `state`
        df['state'] = df['Typical']
        if( relative_sign ):
            df['state'] = pd.np.sign( df['state'].diff() )
        elif( relative ):
            df['state'] = df['state'].diff()
        # Sort and ReIndex
        df.sort_values( by = ['Date'], ascending = True, inplace = True )
        df.reset_index( drop = True, inplace = True )
        self.df = df
        # Reset
        self.reset()
        # Return
        return

    # Make State
    def __make_state( self, idx ):
        """Create state for given index.
        """
        # Get df data
        state = self.df.iloc[ self.time, : ].to_dict()
        state['time'] = self.time
        # Return
        return( state )

    # Reset
    def reset( self ):
        """Reset the values.
        """
        # Parameters
        ## Time
        self.time = 0
        ## State
        self.current_state = self.__make_state( 0 )
        self.states = [ deepcopy(self.current_state) ]
        self.state_space = {
            'time': 'Environment time (step number).',
            'state': 'Typical price of the market.',
            'other_info': 'There are other info sent along with the state.'
        }
        ## Action
        self.actions = []
        self.action_space = 'Non-Negative Reals'
        # Return
        return

    # State Map
    def states_map( self, states ):
        """Identity map. -> Return last state in states
        """
        # Check if states is compatible
        assert states[-1] == self.current_state, "State Map: current state should be the last element in states."
        # Return
        return( deepcopy(states[-1]) )

    # Update
    def update( self, action ):
        """Independent of the given action, move one step further in time.
        """
        # Check Finished Criteria
        if( self.time >= ( len(self.df) - 1 ) ):
            print('Simulation finished')
            return
        # Parameters
        ## Actions
        self.actions.append( deepcopy(action) )
        ## Time
        self.time += 1
        ## State
        self.current_state = self.__make_state( self.time )
        self.states.append( deepcopy(self.current_state) )
        # Return: return mapped states
        return( self.states_map( self.states ) )


























