##################################################
"""Main Module for TNTP
"""
# External Modules
import os, sys, time
import pathos.multiprocessing as multiprocessing
import pandas as pd
import numpy as np
import scipy as sp
import networkx as nx
# import matplotlib.pyplot as plt
import re
if sys.version_info[0] < 3:
    from StringIO import StringIO
else:
    from io import StringIO
# Main Class


class TNTPData:
    """Main Class Implementation for TNTP Parser
    """
    #########################################
    # Intermediate
    @staticmethod
    def intermediate( city_name, adj_mat, pairs ):
        file_path = city_name + '/cache/' + str(pairs[0]) + '.dat'
        if( os.path.exists( file_path ) ):
            with open(file_path, 'r') as in_file:
                results = ast.literal_eval( in_file.read() )
                in_file.close()
            #
        else:
            G = nx.Graph( adj_mat )
            #
            results = []
            for i,j in pairs:
                results.append( (i,j,nx.dijkstra_path_length( G, i-1, j-1 )) )
            #
            with open(file_path, 'w') as out_file:
                out_file.write( str(results) )
                out_file.close()
            #
        return( results )
    ##########################################
    # Constructor
    def __init__(self, city='Birmingham-England', base_path='./', N_proc = 8, verbose = False):
        """Default Constructure.
        """
        self.verbose = verbose
        self.base_path = base_path
        folder_dir = base_path + city + '/'
        cache_dir = folder_dir + 'cache'
        if( not os.path.exists( cache_dir ) ):
            os.mkdir( cache_dir )
        folder_files = os.listdir(folder_dir)
        folder_file = folder_files[0]
        self.N_proc = N_proc
        ##################################
        net_files = [entry for entry in folder_files if (
            '_net.' in entry.lower())]
        node_files = [entry for entry in folder_files if (
            '_node.' in entry.lower())]
        trip_files = [entry for entry in folder_files if (
            '_trips.' in entry.lower())]
        ##################################
        # Meta
        self.city_name = city
        self.meta_data_dict = None
        # Node
        self.node_data = None
        # Net
        self.net_data = None
        self.adj_mat = None
        self.weight_mat = None
        # Trip
        self.trip_data = None
        # Graph
        self.distances = dict()
        self.nx_graph = None
        ##################################
        # Net
        assert len(net_files) < 2, "Multiple '_Net' Files!"
        if( self.verbose ):
            print('Loading "Net" Data ...', end='\r')
        if(len(net_files) == 1):
            to_read = folder_dir + net_files[0]
            self.get_data_net(to_read)
        if( self.verbose ):
            print('Loading "Net" Data Done!')
        ##########################
        # Nodes
        assert len(node_files) < 2, "Multiple '_Node' Files!"
        if( self.verbose ):
            print('Loading "Nodes" Data ...', end='\r')
        if(len(node_files) == 1):
            to_read = folder_dir + node_files[0]
            self.get_data_node(to_read)
        if( self.verbose ):
            print('Loading "Nodes" Data Done!')
        ##########################
        # Trip
        if( self.verbose ):
            print('Loading "Trip" Data ...', end='\r')
        assert len(trip_files) < 2, "Multiple '_Trip' Files!"
        if(len(trip_files) == 1):
            to_read = folder_dir + trip_files[0]
            self.get_data_trip(to_read)
        if( self.verbose ):
            print('Loading "Trip" Data Done!')
        ##########################
        # Add Nodes to Net
        if( self.verbose ):
            print('Adding "Nodes" to "Net": ...', end='\r')
        if(not(self.node_data is None) and not(self.net_data is None)):
            # From Coordinates
            self.net_data = pd.merge(
                left=self.net_data, right=self.node_data, left_on='From', right_on='Node')
            self.net_data.rename(
                columns={'X': 'From_X', 'Y': 'From_Y'}, inplace=True)
            self.net_data.drop('Node', axis=1, inplace=True)
            # To Coordinates
            self.net_data = pd.merge(
                left=self.net_data, right=self.node_data, left_on='To', right_on='Node')
            self.net_data.rename(
                columns={'X': 'To_X', 'Y': 'To_Y'}, inplace=True)
            self.net_data.drop('Node', axis=1, inplace=True)
            # Add Distances
            self.net_data['Distance'] = np.sqrt(
                (self.net_data.From_X - self.net_data.To_X)**2 + (self.net_data.From_Y - self.net_data.To_Y)**2)
        ##########################
        # Create Weight Matrice
        if(not(self.node_data is None) and not(self.net_data is None)):
            N = int(self.meta_data_dict['numberofnodes'])
            # Weight Matrix
            self.weight_mat = np.zeros((N, N))
            for i, j, w in zip(self.net_data.From, self.net_data.To, self.net_data.Distance):
                self.weight_mat[i - 1][j - 1] = w
            self.weight_mat = sp.sparse.csr_matrix(self.weight_mat)
        if( self.verbose ):
            print('Adding "Nodes" to "Net": Done!')
        ##########################
        # Create Networkx Object
        if( self.verbose ):
            print('Creating "NX_GRAPH": ...', end='\r')
        self.create_nx()
        if( self.verbose ):
            print('Creating "NX_GRAPH": Done!')
        ##########################
        # Calculate Distances
        if( self.verbose ):
            print('Calculating Distances: ...')
        self.calc_distances()
        if( self.verbose ):
            print('Calculating Distances: Done!')
        return
    ######################################
    # Get Net
    def get_data_net(self, to_read):
        ##################################
        # Net
        # Read Data
        with open(to_read, 'r') as in_file:
            data = in_file.read()
            in_file.close()
        data = data.replace('\t', '    ')
        # Parse
        # Meta Data
        meta_data = data[:data.rfind('~')]
        meta_data = re.sub(' +', '', meta_data).lower()
        meta_data = meta_data[:meta_data.rfind('<endofmetadata>')]
        meta_data = meta_data.split('\n')
        tmp = []
        for entry in meta_data:
            if(len(entry) == 0):
                continue
            attr, val = entry.split('>')
            attr = attr[1:]
            try:
                val = np.float(val)
            except:
                pass
            tmp.append((attr, val))
        meta_data = tmp.copy()
        meta_data_dict = dict(meta_data)
        # Store
        if(self.meta_data_dict is None):
            self.meta_data_dict = meta_data_dict.copy()
        else:
            self.meta_data_dict.update(meta_data_dict)
        # Table
        table_data = data[(data.rfind('~') + 1):]
        table_data = re.sub('\n +', '\n', table_data)
        table_data = table_data.split('\n')[1:]
        table_data = [re.sub(' +', ',', entry)[:-2] for entry in table_data]
        col_names = ','.join([
            'From', 'To', 'Capacity', 'Length', 'FreeFlowTime', 'B', 'Power', 'SpeedLimit', 'Toll', 'Type'
        ])
        table_data = '\n'.join(table_data)
        table_data = table_data.replace(',\n', '\n')
        self.debug = table_data  # DEBUG!
        table_data = pd.read_csv(StringIO(col_names + '\n' + table_data))
        # Adj/Weight Matrices
        N = int(self.meta_data_dict['numberofnodes'])
        self.adj_mat = np.zeros((N, N))
        for i, j in zip(table_data.From, table_data.To):
            self.adj_mat[i - 1][j - 1] = 1
        self.adj_mat = sp.sparse.csr_matrix(self.adj_mat)
        # Store Values
        self.net_data = table_data.copy()
        # Return
        return
    ######################################
    # Get Node
    def get_data_node(self, to_read):
        ##################################
        # Nodes
        # Read Data
        with open(to_read, 'r') as in_file:
            data = in_file.read()
            in_file.close()
        data = data.replace('\t', '    ')
        # Parse
        # Table
        node_list = re.sub(' +', ',', data)
        node_list = node_list.replace(',\n', '\n')
        node_list = node_list.replace(',;', '')
        # Replace Header
        if(node_list[0].lower() == 'n'):
            node_list = "Node,X,Y\n" + node_list.split('\n', 1)[1]
        elif( node_list[0].isdecimal() ):
            node_list = "Node,X,Y\n" + node_list
        node_list = pd.read_csv(StringIO(node_list))
        # Store Data
        self.node_data = node_list.copy()
        # Return
        return
    ######################################
    # Get Trip
    def get_data_trip(self, to_read):
        ##################################
        # Trip
        # Read Data
        with open(to_read, 'r') as in_file:
            data = in_file.read().lower()
            in_file.close()
        data = data.replace('\t', '    ')
        # Parse
        meta_data = data[:data.rfind('<end of metadata>')]
        meta_data = re.sub(' +', '', meta_data)
        meta_data = meta_data.split('\n')
        tmp = []
        for entry in meta_data:
            if(len(entry) == 0):
                continue
            attr, val = entry.split('>')
            attr = attr[1:]
            try:
                val = np.float(val)
            except:
                pass
            tmp.append((attr, val))
        meta_data = tmp.copy()
        meta_data_dict = dict(meta_data)
        # Store
        if(self.meta_data_dict is None):
            self.meta_data_dict = meta_data_dict.copy()
        else:
            self.meta_data_dict.update(meta_data_dict)
        # Tabular
        table_data = re.sub(' +', '', data).split('origin')[1:]
        N = int(meta_data_dict['numberofzones'])
        tmp = np.zeros((N, N))
        counter = 0
        for entry in table_data:
            raws = entry.split('\n')
            row_idx = int(raws[0]) - 1
            for raw in raws[1:]:
                columns = raw.split(';')[:-1]
                for column in columns:
                    col_raw = column.split(':')
                    # Get Info
                    col_idx = int(col_raw[0]) - 1
                    val = np.float(col_raw[1])  # Get Column Index
                    # Set Value
                    tmp[row_idx][col_idx] = val
                    counter += 1
        table_data = tmp.copy()
        # Store
        self.trip_data = table_data.copy()
        # Return
        return
    #######################################
    # Calculate Distances
    def calc_distances(self):
        N = int(self.meta_data_dict['numberofzones'])
        if( self.verbose ):
            print('\tNumebr of Operations: {}'.format(N**2))
        ##############################
        # Euclidean Distance
        if( self.verbose ):
            print('\tEuclidean: ...')
        file_path = os.path.join( self.base_path, self.city_name, 'euclidean.txt' )
        if(os.path.exists(file_path)):
            self.distances['Euclidean'] = np.loadtxt(file_path, delimiter=',')
        else:
            if(not(self.node_data is None)):
                start_time = time.time()
                ##
                tmp = np.zeros((N, N))
                for i in range(N):
                    r_i = self.node_data.iloc[i, 1:].values
                    for j in range(i + 1, N):
                        r_j = self.node_data.iloc[j, 1:].values
                        d_ij = np.linalg.norm(r_i - r_j)
                        tmp[i][j] = d_ij
                # Save
                self.distances['Euclidean'] = tmp.transpose() + tmp
                ##
                duration = time.time() - start_time
                if( self.verbose ):
                    print('Total Time "Euclidean" Paths: {}s'.format( duration ))
                    print('Avg Time Per "Euclidean" Path: {}s'.format(duration / N**2))
                # Store
                np.savetxt(
                    file_path, self.distances['Euclidean'], fmt='%.3f', delimiter=',')
        if( self.verbose ):
            print('\tEuclidean: Done!')
        ##############################
        # Free-Flow Distance
        if( self.verbose ):
            print('\tManhattan: ...')
        file_path = os.path.join( self.base_path, self.city_name, 'manhattan.txt' )
        if(os.path.exists(file_path)):
            self.distances['Manhattan'] = np.loadtxt(file_path, delimiter=',')
        else:
            if(not(self.nx_graph is None)):
                start_time = time.time()
                ##
                tmp = np.zeros((N, N))
                ## Intermediate Function
                adj_mat = self.adj_mat.copy()
                ##
                pairs = []
                for i in range(1,N+1):
                    for j in range(i+1,N+1):
                        pairs.append( (i,j) )
                ##
                N_chunk = 10**3
                N_pairs = int( len(pairs) / N_chunk ) + 1
                ##
                tasks = []
                for i in range( N_pairs ):
                    selected_pairs = pairs[ (i*N_chunk):((i+1)*N_chunk) ]
                    tasks.append( ( self.city_name, adj_mat, selected_pairs ) )
                ## Pool
                if( self.verbose ):
                    print('\tBefore Pool')
                mp_pool = multiprocessing.Pool( self.N_proc )
                results_list = mp_pool.starmap( TNTPData.intermediate, tasks )
                if( self.verbose ):
                    print('\tAfter Pool')
                ## Fill
                for results in results_list:
                    for i,j,l in results:
                        tmp[i-1][j-1] = l
                ##
                self.distances['Manhattan'] = tmp.transpose() + tmp
                ##
                duration = time.time() - start_time
                if( self.verbose ):
                    print('Total Time "Manhattan" Paths: {}s'.format( duration ))
                    print('Avg Time Per "Manhattan" Path: {}s'.format(duration / N**2))
                # Store
                np.savetxt(
                    file_path, self.distances['Manhattan'], fmt='%.3f', delimiter=',')
        if( self.verbose ):
            print('\tManhattan: Done!')
        return
    ######################################
    # Create Networkx
    def create_nx(self):
        if(self.net_data is None):
            return
        self.nx_graph = nx.Graph()
        if(self.node_data is None):
            # Add Edges
            for i, j in zip(self.net_data.From, self.net_data.To):
                self.nx_graph.add_edge(i, j)
        else:
            # Add Nodes
            ## From
            for i, x, y in zip(self.net_data.From, self.net_data.From_X, self.net_data.From_Y):
                self.nx_graph.add_node(i, pos=(x, y))
            ## To
            for i, x, y in zip(self.net_data.To, self.net_data.To_X, self.net_data.To_Y):
                self.nx_graph.add_node(i, pos=(x, y))
            # Add Edges
            for i, j, w in zip(self.net_data.From, self.net_data.To, self.net_data.Distance):
                self.nx_graph.add_edge(i, j, weight=w)
            # Add Attributes
            for column_name in self.net_data.columns[2:10]:
                values = dict( zip( zip(self.net_data.From , self.net_data.To), self.net_data[column_name] ) )
                nx.set_edge_attributes( self.nx_graph, values, name = column_name )
        return








































# End of File! :D for stupid ATOM :)
