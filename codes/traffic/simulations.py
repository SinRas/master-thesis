# Modules
## External
import time,itertools
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import multiprocessing as mp
## Internal
import assign_traffic

# Styles


# Methods
## Intermediate
def intermediate( new_od_matrix, od_pairs, graph ):
    traffic_assigner = assign_traffic.Averaging( graph, new_od_matrix, alpha = 0.1, nominal = 1, od_pairs = od_pairs )
    traffic_assigner.assign_till_converge( stop_threshold = 0.1, decay_alpha = True, decay_rounds = 15, verbose = False )
    return( traffic_assigner.edge_costs.copy() )
## Path Join
def path_join( path1, path2 ):
    found = False
    for i,v in enumerate(path1):
        try:
            j = path2.index(v)
            found = True
            break
        except:
            pass
    #
    if( found ):
        path = path1[:i] + path2[j:]
        return( path )
    else:
        return( None )
## Paths to Depth
def paths_to_depth( graph, src, dst, depth ):
    src_neighbors = set( [src] )
    for _ in range(depth):
        src_neighbors_copy = src_neighbors.copy()
        for v in src_neighbors_copy:
            src_neighbors.update( graph.adj[v] )
    #
    src_neighbors.remove(src)
    #
    src_neighbors = list( src_neighbors )
    #
    paths_src_dst = []
    for intermediate in src_neighbors:
        # Find Paths
        path_src_intermediate = nx.dijkstra_path( graph, src, intermediate )
        path_intermediate_dst = nx.dijkstra_path( graph, intermediate, dst )
        # Add
        paths_src_dst.append( tuple(path_join( path_src_intermediate, path_intermediate_dst )) )
    # Unique
    paths_src_dst = set( paths_src_dst )
    #
    result = dict()
    initial_weight = 1 / len(paths_src_dst)
    for path_src_dst in paths_src_dst:
        result[ path_src_dst ] = {
            'weight': initial_weight,
            'weights': [ initial_weight ],
        }
    #
    return( result )
## All Paths to Depth
def gen_od_pairs_paths( graph, od_pairs, depth, verbose = False ):
    od_pairs_paths = dict()
    for i,od_pair in enumerate(od_pairs):
        src, dst = od_pair
        od_pairs_paths[ od_pair ] = paths_to_depth( graph, src, dst, depth )
        if( verbose ):
            print( '{}/{}'.format( i+1, len(od_pairs) ), end = '\r' )
    if(verbose):
        print( '{}/{}'.format( i+1, len(od_pairs) ) )
    # Return
    return( od_pairs_paths )

# Classes
## Weighted Majority
class DeterministicWeightedMajority:
    """Deterministic Weighted Majority algorithm for Traffic.
    """
    # Constructor
    def __init__( self, tntp_instance, beta = 1.0, depth = 10 ):
        # Parameters
        self.tntp_instance = tntp_instance
        self.beta = beta
        self.depth = depth
        # OD Pairs
        zone_nodes = list(range( int(tntp_instance.meta_data_dict['numberofzones']) ) )
        self.od_pairs = []
        for u,v in itertools.product( zone_nodes, zone_nodes ):
            if( u != v ):
                self.od_pairs.append( (u+1,v+1) )
        # Get Graph
        self.graph = tntp_instance.nx_graph
        # Get OD Data
        self.od_matrix = tntp_instance.trip_data
        # First run to find paths
        print('Finding path candidates ...', end='\r')
        ## Traffic Assigner
        self.traffic_assigner = assign_traffic.Averaging( self.graph, self.od_matrix, alpha = 0.1, nominal = 1, od_pairs = self.od_pairs )
        self.traffic_assigner.assign_till_converge( stop_threshold = 0.1, decay_alpha = True, decay_rounds = 20, verbose = False )
        ## Find Paths
        self.od_pairs_paths = gen_od_pairs_paths( self.graph, self.od_pairs, self.depth )
        print('Finding path candidates ... done')
        # Return
        return
    # Path Cost
    def path_cost( self, edge_costs, path ):
        if( len(path) < 2 ):
            return(0)
        sum_costs = 0
        for i in range(len(path)-1):
            src, dst = path[i], path[i+1]
            cost = edge_costs[ (src, dst) ]
            sum_costs += cost
        return(sum_costs)
    # Loss
    def loss( self, real, pred ):
        result = np.exp( - self.beta * ( pred - real ) )
        return( result )
    # Simulate
    def simulate( self, N_max = 120, N_proc = 6, verbose = True ):
        # Intermediate Function
        graph = self.graph
        od_pairs = self.od_pairs
        mp_pool = mp.Pool(N_proc)
        # Run
        for i in range(N_max//N_proc):
            print('Round: {:>5}/{:>5} started ...'.format( i+1, N_max//N_proc ))
            _start = time.time()
            # Generate New OD Matrix
            new_od_matrices = [ [np.random.poisson(lam = self.od_matrix), od_pairs, graph.copy() ] for _ in range(N_proc) ]
            # Assign New Traffic
            edge_costs_list = mp_pool.starmap( intermediate, new_od_matrices )
            # For each resulting edge_cost
            for edge_costs in edge_costs_list:
                # Pair Shortest Paths
                for od_pair in self.od_pairs_paths:
                    ## Dijkstra Shortest Path Length
                    src, dst = od_pair
                    shortest_path_cost = nx.dijkstra_path_length( self.graph, src, dst, weight='cost' )
                    ## Update Path Weights
                    for current_path in self.od_pairs_paths[od_pair]:
                        # Path Cost
                        current_path_cost = self.path_cost( edge_costs, current_path )
                        # Weight Update Factor
                        weight_factor = self.loss( shortest_path_cost, current_path_cost )
                        # New weight
                        new_weight = self.od_pairs_paths[od_pair][current_path]['weight'] * weight_factor
                        # Store
                        self.od_pairs_paths[od_pair][current_path]['weight'] = new_weight
                        self.od_pairs_paths[od_pair][current_path]['weights'].append( new_weight )
            #
            _end = time.time()
            duration = _end - _start
            print('Duration: {:.0f} s\n----'.format(duration))






















