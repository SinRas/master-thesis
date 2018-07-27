# Modules
import os # listdir
import pandas as pd # DataFrame manipulation

# Styles
row_top = '='*20
row_bottom = '===^-^== *-* ==^-^==='
header = '===  {}'

# Run
def run():
    """Load the data, filter by the number of available entries.
    """
    # Header
    print( row_top )
    print( header.format( 'loaddata.py' ) )
    # Data directory
    data_dir = os.path.join( 'data', 'iran' )
    # List of files
    file_names = os.listdir( data_dir )
    # Load the data
    dataframes = dict()
    for file_name in file_names:
        # File Info
        file_id = int( file_name[:-8] )
        file_path = os.path.join( data_dir, file_name )
        # Load
        df = pd.read_csv( file_path, parse_dates=[0] )
        # df = df[ df.Count != 0 ].reset_index()
        dataframes[ file_id ] = df.copy()
    # Find Common Dates
    common_dates = set()
    for file_id in dataframes:
        dataframe_dates = set( dataframes[file_id].Date )
        if( len(common_dates) == 0 ):
            common_dates.update( dataframe_dates )
        else:
            common_dates = common_dates.intersection(dataframe_dates)
    # Check Continuous Date
    continuois_dates = set()
    df = dataframes[ file_id ] # Get on Data Frame
    df = df[ df.Date.map( lambda x: x in common_dates ) ].reset_index( drop = True ) # Filter Dates
    date_diffs = df.Date - df.Date.shift(-1) # Caclulate Date Diffrerences
    ## Find first discontinuity
    for idx, d in enumerate(date_diffs):
        if( d.days > 10 ):
            break
    ## Filter Dates
    continuois_dates = set( df.Date[:idx] )
    # Filter Dates
    for file_id in dataframes:
        df = dataframes[file_id]
        dataframes[file_id] = df[ df.Date.map( lambda x: x in continuois_dates ) ].reset_index( drop = True )
    # Sort Dates
    for file_id in dataframes:
        dataframes[file_id] = dataframes[file_id].sort_values( by=['Date'], ascending=True ).reset_index( drop = True )
    # Reports
    print( 'Symbols Loaded:\n\t{}\n------'.format( '\n\t'.join( [ str(entry) for entry in list(dataframes) ] ) ) )
    print( '{} Common Dates\n------'.format( len(common_dates) ) )
    print( '{} Continuous Dates {{break is less than 10 days}}\n------'.format( len(continuois_dates) ) )
    # Return
    print( row_bottom )
    return( dataframes )






























