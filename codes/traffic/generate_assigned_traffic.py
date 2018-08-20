# Modules
## External
import os, time, itertools, ast, datetime
import multiprocessing as mp
import numpy as np
## Internal
import TNTP, assign_traffic

# Methods
## Intermediate
def intermediate( file_path, new_od_matrix, od_pairs, graph ):
    # Check if file exists
    assert not os.path.exists( file_path ), "File should not exist!"
    # Create & Assign Traffic
    traffic_assigner = assign_traffic.Averaging( graph, new_od_matrix, alpha = 0.1, nominal = 1, od_pairs = od_pairs )
    traffic_assigner.assign_till_converge( stop_threshold = 0.1, decay_alpha = True, decay_rounds = 15, verbose = False )
    # Edge Costs
    edge_costs = traffic_assigner.edge_costs.copy()
    # Store
    with open( file_path, 'w' ) as out_file:
        out_file.write( str(
            {
                'od_matrix': str(od_matrix.tolist()),
                'edge_costs': str(edge_costs),
            }
        ) )
    # Return
    return

# Variables
## TNTP Instance
tntp_instance = TNTP.TNTPData( city='Berlin-Friedrichshain', base_path='data/tntp/' )
# OD Pairs
zone_nodes = list(range( int(tntp_instance.meta_data_dict['numberofzones']) ) )
od_pairs = []
for u,v in itertools.product( zone_nodes, zone_nodes ):
    if( u != v ):
        od_pairs.append( (u+1,v+1) )
## Graph
graph = tntp_instance.nx_graph
## OD Matrix
od_matrix = tntp_instance.trip_data
## Config
config_file = 'config.json'


if __name__ == '__main__':
    # Time Stamp
    print('|\n|-Time: {}\n|'.format( datetime.datetime.now() ) )
    # Prepare Workspace
    print( '|-Prepare Workspace:' )
    print( '| |' )
    ## Load Factors
    with open( config_file, 'r' ) as in_file:
        ## Load Paramters
        data_dict = ast.literal_eval( in_file.read() )
    ## Extract Parameters
    data_dir = data_dict['data_dir']
    factors = data_dict['factors']
    N_samples = data_dict['N_samples']
    ## Pool
    N_proc = data_dict['N_proc']
    mp_pool = mp.Pool(N_proc)
    ## Create Folders
    for factor in factors:
        folder_path = os.path.join( data_dir, str(factor) )
        if( not os.path.exists(folder_path) ):
            os.makedirs( folder_path )
            print('| |  created  {}'.format(folder_path))
        else:
            print('| |  exists   {}'.format(folder_path))
    # Initialization
    print( '|' )
    print( '|-Defined Variables:' )
    print( '| |' )
    print( '| |  N_proc: {}'.format(N_proc) )
    print( '| |  Factors: {}'.format( factors ) )
    print( '| |  N_samples: {}'.format( N_samples ) )
    print( '| |  OD Matrix (shape): {}'.format(od_matrix.shape) )
    print( '| |' )
    print( '| |  TNTP Instance: {}'.format(tntp_instance) )
    print( '| |  Graph: {}'.format( repr(graph) ) )
    print( '| |  mp_pool: {}'.format(mp_pool) )
    print( '|' )
    # Loop: Main
    print( '|' )
    print( '|-# Main Loop' )
    print( '|' )
    N_factors = len(factors)
    for i,factor in enumerate(factors):
        # Report: Factor
        print( '|\n|-Factor: {}\n| |'.format( factor ) )
        # Loop: Files
        ## Folder Path
        folder_path = os.path.join( data_dir, str(factor) )
        ## Existing Files
        existing_file_names = set([ file_name for file_name in os.listdir( folder_path ) if( file_name[-4:] == '.dat' ) ])
        ## New Files
        to_be_file_names = [ '{}.dat'.format(idx) for idx in range(N_samples) ]
        new_file_names = [ file_name for file_name in to_be_file_names if( not file_name in existing_file_names ) ]
        # Generate New OD Matrix
        print( '| |-Tasks' )
        new_tasks = [ [
            os.path.join(folder_path, str(file_name)), factor * np.random.poisson(lam = od_matrix), od_pairs, graph.copy()
            ] for file_name in new_file_names ]
        print( '| |  | {} tasks created.'.format( len(new_tasks) ) )
        print( '| |  | assigning traffic (in progress)', end = '\r', flush=True )
        # Assign New Traffic
        _start = time.time()
        edge_costs_list = mp_pool.starmap( intermediate, new_tasks )
        _end = time.time()
        _duration = _end - _start
        # Report
        print( '| |  | assigning traffic (finished)   ' )
        print( '| |  | Duration: {:.0f} s'.format( _duration ) )
    # Report: Final
    print( '|' )
    print( '|-Finished' )
    print( '|\n' )