"""
This module will assign traffic assuming 'User Equilibrium' and 'Wardrop Criteria'.
"""
# Internal Modules

# External Modules
import ast, time
import numpy as np
import networkx as nx
import itertools

# Global Functions
def path_gen( city_graph, pair, cutoff = np.inf, pair_flows = None ):
    ## From u to v
    u, v = pair
    ## u != b
    assert u != v, 'Origin and Destination must be non-equal!'
    ## paths dictionary
    path_values = {
        'paths' : [ [u] ],
        'values' : [ 0.0 ]
    }
    ## While True
    while( len(path_values['paths']) > 0 ):
        # Find Shortest Value Path
        idx = np.argmin( path_values['values'] )
        path = path_values['paths'][idx]
        last_node = path[-1]
        value = path_values['values'][idx]
        # For All Adjacents
        # Finished Path
        if( (last_node != v) ):
            # Pop Path
            path_values['paths'].pop(idx)
            path_values['values'].pop(idx)
            # Add New Paths
            for node in city_graph[last_node]:
                # Repeated Node
                if( (node in path) ):
                    continue
                # New Paths
                new_path = path.copy() + [node]
                ## Get New Value
                if( pair_flows is None ):
                    edge_cost = city_graph[last_node][node]['distance']
                else:
                    edge_cost = pair_flows[ (last_node,node) ]
                new_value = value + edge_cost
                # Add New Paths
                if( new_value < cutoff ):
                    path_values['paths'].append( new_path.copy() )
                    path_values['values'].append( new_value )
        # If No Addition, This Path is a DeadEnd!
        if( (last_node == v) ): # This Path is Finished
            ## Remove from Current Stack
            path_values['paths'].pop(idx)
            path_values['values'].pop(idx)
            ## Yield
            yield( ( path.copy(), value ) )
    ## If Finished
    yield( (None,None) )

# Averaging Iterations
class Averaging():
    """This class implements the heuristic algorithm of all-or-nothing assigment and averaging over two iterations.
    """
    # Constructor
    def __init__( self, city_graph, od_matrix, alpha = 0.5, od_pairs = None, nominal = 0 ):
        """
        'City Graph' must be a graph where nodes have 'pos' attributes (position of nodes),
        edges must have 'distance', 'FreeFlowTime', 'Capacity', 'B', 'Power'.
        """
        # Parameters
        self.nominal = nominal
        self.city_graph = city_graph
        self.od_matrix = od_matrix
        self.alpha = alpha
        self.initial_assignement = True
        ## Create Nodes
        self.graph_nodes = list(self.city_graph.nodes)
        ## Create Pairs
        ### OD Pairs
        if( od_pairs is None ):
            self.od_pairs = []
            for u,v in itertools.product( self.graph_nodes, self.graph_nodes ):
                if( u != v ):
                    self.od_pairs.append( (u,v) )
        else:
            self.od_pairs = od_pairs
        ### All Existing Edges
        self.graph_edges = []
        for u,v in list( self.city_graph.edges ):
            self.graph_edges.append( (u,v) )
            self.graph_edges.append( (v,u) )
        # Flows
        ## Edge Flows
        self.edge_flows = dict( zip( self.graph_edges, [0.0]*len(self.graph_edges) ) )
        ## Edge Costs
        self.edge_costs = dict( zip( self.graph_edges, [0.0]*len(self.graph_edges) ) )
        ## Path Place Holders
        self.path_flows = dict()
        self.path_gens = dict()
        for pair in self.od_pairs:
            self.path_flows[ pair ] = dict()
            self.path_gens[ pair ] = None
        # Return
        return
    # Order Pair
    def order_pair( self, pair ):
        u,v = pair
        if( u > v ):
            return( (v,u) )
        else:
            return( pair )
    # Beuro of Pulbic Road
    def bpr( self, freeflowtime, current_flow, capacity, b, power ):
        """Return the Beuro of Public Road Function value:
        S_a( v_a ) = t_a ( 1 + b * ( v_a / c_a )^power )
        where:
        t_a : freeflowtime
        v_a : current_flow
        c_a : capacity
        b : b
        power: power
        """
        # Result
        result = freeflowtime * ( 1 + b * ( current_flow / capacity )**power )
        # Return
        return( result )
    # Reset Map
    def reset_flows( self, to_value = 0.0 ):
        """Reset all flows on edges and paths to given value (default 0.0).
        This will remove any path from the dictionary of path since in the intial step, there are no paths.
        """
        # Reset
        ## Edges Flows
        self.edge_flows = dict( zip( self.graph_edges, [0.0]*len(self.graph_edges) ) )
        ## Edge Costs
        self.edge_costs = dict( zip( self.graph_edges, [0.0]*len(self.graph_edges) ) )
        ## Path
        for pair in self.od_pairs:
            self.path_flows[pair] = dict()
            self.path_gens[pair] = None
        # Return
        return
    # Random Flows
    def random_edge_flows( self, random_generator = np.random.rand ):
        """Randomly generate flows for edges with given random generator ( default np.random.rand )
        """
        # Populate Edges
        for edge in self.graph_edges:
            self.edge_flows[edge] = random_generator()
        # Return
        return
    # Find S_a
    def calc_edge_costs( self ):
        """Calculate the cost function for edges.
        """
        # Calculate Cost Function
        for edge in self.graph_edges:
            # Edge Parameters
            edge_dict = dict(self.city_graph.edges[edge])
            b = edge_dict['B']
            capacity = edge_dict['Capacity']
            power = edge_dict['Power']
            freeflowtime = edge_dict['FreeFlowTime']
            edge_flow = self.edge_flows[edge]
            # Calculate Cost
            current_cost = self.bpr( freeflowtime = freeflowtime, current_flow = edge_flow, capacity = capacity, b = b, power = power )
            # Add Current Cost
            self.edge_costs[edge] = current_cost
        # Set Graph Edge Costs
        nx.set_edge_attributes( self.city_graph, self.edge_costs, name='cost' )
        # Return
        return
    # Assign Traffic Once
    def assign_once( self ):
        """Find the shortest path for each pair of O-D, and assign all the demand to it.
        """
        # Parameters
        if( self.initial_assignement ):
            alpha = 1
            self.initial_assignement = False
        else:
            alpha = self.alpha
        # Update Edge Costs
        self.calc_edge_costs()
        # New Path/Edge Flows
        ## Edges
        self.edge_flows =  dict( zip( self.graph_edges, [0.0]*len(self.graph_edges) ) )
        ## Paths
        new_path_flows = dict()
        for pair in self.od_pairs:
            new_path_flows[ pair ] = dict()
        # Create Path Generators
        N = len(self.path_flows)
        for i,pair in enumerate(self.path_flows):
            # DEBUG
            # print( 'Find New Solution: {} / {}'.format( i+1, N ), end='\r' )
            ## Define and Get Shortest Path from Generator
            # shortest_path = next( path_gen( city_graph  = self.city_graph, pair = pair, pair_flows = self.edge_costs ) )[0]
            source, target = pair
            ## Sub Graph with other OD Nodes removed
            city_subgraph = self.city_graph.copy()
            for k in range(1,self.od_matrix.shape[0]):
                if( k == source or k == target ):
                    continue
                city_subgraph.remove_node( k )
            ## Get Shortest Path
            shortest_path = nx.dijkstra_path( city_subgraph, source, target, weight='cost' )
            ## OD for Pair
            u,v = pair
            od_value = self.od_matrix[u-self.nominal][v-self.nominal]
            ## Add New Path to Flow
            new_path_flows[pair][ str(shortest_path) ] = od_value
        # print( 'Find New Solution: {} / {}'.format( i+1, N ) )
        # Update Path Flows
        for pair in self.path_flows:
            # DEBUG
            # print( 'Update Path Flows: {} / {}'.format( i+1, N ), end='\r' )
            all_paths = set( list(self.path_flows[pair]) + list(new_path_flows[pair]) )
            # Loop All Paths
            for path in all_paths:
                # Calculate New Value
                if( not(path in self.path_flows[pair]) ): # New Path
                    new_value = alpha * new_path_flows[pair][path]
                elif( not(path in new_path_flows[pair]) ): # Removed Path
                    new_value = (1-alpha) * self.path_flows[pair][path]
                else: # Combine Old and New
                    new_value = (1-alpha) * self.path_flows[pair][path] + alpha * new_path_flows[pair][path]
                # Set Value
                self.path_flows[pair][path] = new_value
        # DEBUG
        # print( 'Update Path Flows: {} / {}'.format( i+1, N ) )
        # Update Edge Values
        for pair in self.path_flows:
            # DEBUG
            # print( 'Update Edge Values: {} / {}'.format( i+1, N ), end='\r' )
            # Loop through Paths
            for path in self.path_flows[pair]:
                path_flow = self.path_flows[pair][path]
                path = ast.literal_eval( path )
                # Loop through Edges
                for i, u in enumerate(path[:-1]):
                    if( i == (len(path)-1) ):
                        break
                    v = path[i+1]
                    self.edge_flows[ (u,v) ] += path_flow
        # Set Edge Flow Attribute
        nx.set_edge_attributes( self.city_graph, self.edge_flows, name='flow' )
        # DEBUG
        # print( 'Update Edge Values: {} / {}'.format( i+1, N ) )
        # Update Costs
        self.calc_edge_costs()
        # Return
        return
    # Assign Traffic Once
    def assign_till_converge( self, stop_threshold = 0.1, decay_alpha = False, decay_rounds = 10, verbose = True ):
        """Do Assignment untill some sort of convergence is acquired.
        """
        prev_rel_delta = np.inf
        # Loop Till Convergence
        i = 0
        while(True):
            # Time
            start_time = time.time()
            # Calculate Diff Norm
            diff_norm = 0.0
            sum_norm = 0.0
            # Copy Costs
            prev_edge_costs = self.edge_costs.copy()
            # Assign Once
            self.assign_once()
            # Update max_rel_delta
            for pair in self.edge_costs:
                sum_norm += prev_edge_costs[pair]**2
                diff_norm += (prev_edge_costs[pair] - self.edge_costs[pair])**2
            # Rel Difference
            diff_norm = np.sqrt( diff_norm )
            sum_norm = np.sqrt( sum_norm )
            rel_diff = diff_norm / (sum_norm + 0.1)
            # Freeze
            if( (decay_alpha) and ( (i+1)%decay_rounds == 0 ) ):
                self.alpha *= 0.9
            # Time
            end_time = time.time()
            duration = end_time - start_time
            # Report
            if( verbose ):
                print('Round: {:3.0f}, Alpha: {:.6f}, Diff Norm:\t{:.7f}, Duration: {:.2f}s  '.format( i+1, self.alpha, diff_norm , duration ), end = '\r' )
            # Increment
            i += 1
            # Check Stopping Criteria
            if( diff_norm < stop_threshold ):
                break
        # Last Report
        if( verbose ):
            print('Round: {:3.0f}, Alpha: {:.6f}, Diff Norm:\t{:.7f}, Duration: {:.2f}s  '.format( i+1, self.alpha, diff_norm , duration ) )
        # Return
        return
    # Wardrop Measure
    def wardrop_measure( self ):
        """Check how much deviation there is from Wardrop Equilibrium.
        """
        od_deviations = dict()
        for pair in self.od_pairs:
            # Pair Parameters
            min_pair = np.inf
            max_pair = 0.0
            for path in self.path_flows[pair]:
                # Calculate Path Cost
                path_cost = 0.0
                path = ast.literal_eval( path )
                for i, u in enumerate(path[:-1]):
                    v = path[i+1]
                    path_cost += self.edge_costs[ (u,v) ]
                # Update Max/Min Pair
                min_pair = min( min_pair, path_cost )
                max_pair = max( max_pair, path_cost )
            # Dijkstra Path
            source, target = pair
            path = nx.dijkstra_path( self.city_graph, source, target, weight='cost' )
            path_cost = 0.0
            for i, u in enumerate(path[:-1]):
                v = path[i+1]
                path_cost += self.edge_costs[ (u,v) ]
            # DEBUG
            # print('Min Flows: {}, Dijkstra: {}'.format( min_pair, path_cost ))
            min_pair = min( min_pair, path_cost )
            # Calculate Deviation
            od_deviations[pair] = max_pair - min_pair
        # Save
        self.od_deviations = od_deviations
        # Return
        return

# Fast and Convergent Paper
