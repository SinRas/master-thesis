# Modules
import talib
import numpy as np

# Style
row_top = '='*20
row_bottom = '===^-^== *-* ==^-^==='
header = '===  {}'

# Methods
## Sign
def sign(x):
    if( x < 0 ):
        return( -1 )
    elif( x > 0 ):
        return( 1 )
    else:
        return( 0 )

# Classes
## True
class Correct:
    """True knowledge about given idx.
    """
    # Constructor
    def __init__( self, series, name = 'Correct' ):
        # Paramters
        self.name = name
        self.series = series
        self.desc = 'True Values'
        # Return
        return
    # Get
    def get( self, idx ):
        # Index must be valid
        assert 0 <= idx <= len(self.series), "Invalid index: {}".format( idx )
        # No Knowledge
        if( idx == 0 ):
            return( 0 )
        # Relative Sign
        result = sign( self.series[idx] - self.series[idx-1] )
        # Return
        return(result)

## Expert (base)
class Expert:
    """Base class for expert.
    """
    # Constructor
    def __init__(self, series = None, name = 'Expert', lag = 1 ):
        # Paramters
        self.name = name
        assert lag >= 1, "'lag' must be at leas 1."
        self.lag = lag
        if( not( series is None ) ):
            self.series = series
        # Generate Decsription
        self.desc = '{}_lag{}'.format( name, lag )
        # Return
        return

## Single Series with Delays
class SingleSeries(Expert):
    """Signals based on a single series and previous one.
    """
    # Constructor
    def __init__( self, series, name = 'SingleSeries-unnamed', lag = 1 ):
        """An instance of 'SingleSeries' based on time-series given by 'series'.
        """
        # Super
        Expert.__init__( self, series, name, lag )
        # Return
        return
    # Get Data for given index
    def get( self, idx ):
        """Get the prediction for given idx.
        """
        # No Information yet
        if( idx <= self.lag ):
            return( 0 ) # No Signal
        # Sign of change in idx-lag relative to its previous one
        target = self.series[ idx - self.lag ]
        prev = self.series[ idx - self.lag - 1 ]
        result = sign( target - prev )
        # Special Cases
        if( prev == 0 ):
            return( 0 )
        # Return
        return( result )

## RSI
class RSI(Expert):
    """Using RSI as signaling strategy.
    """
    # Constructor
    def __init__( self, series, name = 'RSI-unnamed', lag = 1, timeperiod = 7, sell_threshold = 70, buy_thershold = 30 ):
        # Super
        Expert.__init__( self, series, name, lag )
        self.desc += '_timeperiod{}_sell{}_buy{}'.format( timeperiod, sell_threshold, buy_thershold )
        # Paramters
        assert timeperiod > 0, "'timeperiod' must be at least 1"
        self.timeperiod = timeperiod
        self.sell_threshold = sell_threshold
        self.buy_thershold = buy_thershold
        # Return
        return
    # Get Data for given index
    def get( self, idx ):
        """Get the prediction for given idx.
        """
        # No Information yet
        if( idx <= ( self.lag + self.timeperiod ) ):
            return( 0 ) # No Signal
        # Calculate RSI
        rsi = talib.RSI( self.series[ (idx - self.lag - self.timeperiod):(idx - self.lag + 1) ], timeperiod = self.timeperiod )[-1]
        # Result
        if( rsi >= self.sell_threshold ):
            result = -1
        elif( rsi <= self.buy_thershold ):
            result = 1
        else:
            result = 0
        # Return
        return( result )

## MFI
class MFI(Expert):
    """Consider MFI for signaling.
    """
    # Constructor
    def __init__( self, series_high, series_low, series_close, series_volume, name = 'MFI', lag = 1, timeperiod = 7, sell_threshold = 70, buy_thershold = 30 ):
        # Super
        Expert.__init__( self, None, name, lag )
        self.desc += '_timeperiod{}_sell{}_buy{}'.format( timeperiod, sell_threshold, buy_thershold )
        # Paramters
        assert timeperiod > 0, "'timeperiod' must be at least 1"
        self.timeperiod = timeperiod
        self.sell_threshold = sell_threshold
        self.buy_thershold = buy_thershold

        self.series_high = series_high
        self.series_low = series_low
        self.series_close = series_close
        self.series_volume = series_volume.astype( 'float' )
        # Return
        return
    # Get Data for given index
    def get( self, idx ):
        """Get the prediction for given idx.
        """
        # No Information yet
        if( idx <= ( self.lag + self.timeperiod ) ):
            return( 0 ) # No Signal
        # Calculate mfi
        mfi = talib.MFI(
            self.series_high[ (idx - self.lag - self.timeperiod):(idx - self.lag + 1) ],
            self.series_low[ (idx - self.lag - self.timeperiod):(idx - self.lag + 1) ],
            self.series_close[ (idx - self.lag - self.timeperiod):(idx - self.lag + 1) ],
            self.series_volume[ (idx - self.lag - self.timeperiod):(idx - self.lag + 1) ],
            timeperiod = self.timeperiod
        )[-1]
        # Result
        if( mfi >= self.sell_threshold ):
            result = -1
        elif( mfi <= self.buy_thershold ):
            result = 1
        else:
            result = 0
        # Return
        return( result )