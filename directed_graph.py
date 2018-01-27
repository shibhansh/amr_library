from __future__ import division
from collections import defaultdict
import operator
import networkx as nx
import numpy
import scipy

from grasshopper import *

# data structure is represented as a 'dict of sets'
# for the directed case - The connection - (node1,node2) represents an arc from node1 to node2
# {'A': {'B'}} represents a self._graph with a single arc from A->B

class Graph(object):
    """ Graph data structure, directed by default.
        Initializatin expects input of type - [['A', 'B'], ['B', 'C'], ['B', 'D']]
    """

    def __init__(self, connections,nodes=[], directed=True, edge_lables=dict(), weights=dict(),var_to_sent={}):
        self._graph = defaultdict(list)
        self._directed = directed
        if self._directed:
            self.sent_in_degree = {}
            self.in_degree = {}
            self.out_degree = {}
            self.reverse_graph = defaultdict(list)
            self.topological_order = {}

        # weights only used in nx graph
        self.weights = weights
        self.var_to_sent = var_to_sent
        self.sent_to_var = self.get_sent_to_var()

        self.add_nodes(nodes)
        self.add_connections(connections)

        self.edge_lables = self.fix_edge_lables(edge_lables=dict(edge_lables))
        # get sent_in_degree, no. of edges coming from different sents
        self.get_sent_in_degree()
        self.topological_ordering()

        # Add a 'nx' graph
        self.generate_nx_graph()
        # get all_pair_shortest_paths
        self.all_pairs_shortest_paths = nx.all_pairs_shortest_path(self.undirected_nx_graph)

    # ranking functions
    def rank_pairs(self,ranks=[],weights=[],pairs_to_rank=10,to_remove_parent_transfer=False):
        # find pairs with maximum weights - 
        if weights==[]: return []
        pair_weights = defaultdict(str)
        for index_n1, n1 in enumerate(ranks):
            for index_n2, n2 in enumerate(ranks):
                if n1!=n2 and n2 in self.all_pairs_shortest_paths[n1].keys(): 
                    total_weight = weights[index_n1] + weights[index_n2]
                    # if to_remove_parent_transfer:
                    #     if n1 in self.get_ancestor_list(n2):
                    #         total_weight -= self.transfer_ratio(n1,n2)*weights[index_n1]
                    #     if n2 in self.get_ancestor_list(n1):
                    #         total_weight -= self.transfer_ratio(n2,n1)*weights[index_n2]
                    pair_weights[n1+' '+n2] = total_weight#/len(self.all_pairs_shortest_paths[n1][n2])
                else: pair_weights[n1+' '+n2] = 0

        ordered_pairs = sorted(pair_weights.items(), key=operator.itemgetter(1))
        ordered_pairs.reverse()
        final_ordered_pairs = defaultdict(str)

        # need to return only specific num of pairs; removing repretetive pairs
        rank_order_pairs = []
        for pair,weight in ordered_pairs:
            first_var = pair.split()[0]
            second_var = pair.split()[1]
            if len(final_ordered_pairs.keys()) == pairs_to_rank:   break
            if second_var + ' ' + first_var not in final_ordered_pairs.keys():
                final_ordered_pairs[first_var+' '+second_var] = weight
                rank_order_pairs.append(first_var+' '+second_var)
        return rank_order_pairs
        # return final_ordered_pairs.keys()

    def max_imp_path(self,ordered_pairs=[]):
        current_path = []
        connections = []
        # traverse in decreasing order of weights
        paths_to_return = []
        # This step is probably too slow here
        for pair in ordered_pairs:
            first_var = pair.split()[0]
            second_var = pair.split()[1]

            try:    sent = min(list(set(self.var_to_sent[first_var]).intersection(self.var_to_sent[second_var])))
            except: sent = -1
            if sent != -1:
                # find path in this sentence
                sent_connections = self.get_connections(nodes=self.sent_to_var[sent])
                sent_sub_graph = Graph(sent_connections,nodes=self.sent_to_var[sent])
                current_path = sent_sub_graph.all_pairs_shortest_paths[first_var][second_var]
                # current_path = [first_var,second_var]
            # if sent not found, return the shortes_path
            else:
                current_path = self.all_pairs_shortest_paths[first_var][second_var]
            if len(current_path):# and len(current_path) < 9: 
                connections = self.get_connections(current_path)
                to_add_current_path = True

                if to_add_current_path:
                    paths_to_return.append([current_path,Graph(connections=connections,
                                                                nodes=set(current_path)),sent])
            # if num_paths_to_return == len(paths_to_return):
            #     return paths_to_return
        return paths_to_return

    def get_grasshopper_ranks(self,):
        # Returns ranks based on the 'grasshopper' algorithm
        W = []
        for node1 in self._graph:
            # temp is the list of transistion probabilities for the current matrix
            temp = []
            for node2 in self._graph:
                if node2 in self._graph[node1]: temp.append(1)
                else:   temp.append(0)
            temp_sum = sum(temp)
            if all(item == 0 for item in temp): temp = [1/len(temp)]*len(temp)
            else: temp = [item/temp_sum for item in temp]
            W.append(temp)

        W = np.array(W)
        r = np.array( [1/len(self._graph)]*len(self._graph) )
        ranks = grasshopper(W, r, _lambda=1, k=len(self._graph.keys()))
        # For now weight given to each node is based on just 'index'
        return [(self._graph.keys()[location],index) for index,location in enumerate(reversed(ranks))]

    def rank_sent_in_degree(self,):
        # Returns ranks using 'in_degree'
        order_vars = sorted(self.sent_in_degree.items(), key=operator.itemgetter(1))
        return order_vars

    def rank_in_degree(self,):
        # Returns ranks using 'in_degree'
        order_vars = sorted(self.in_degree.items(), key=operator.itemgetter(1))
        return order_vars

    def rank_in_plus_out_degree(self,):
        # Returns ranks using 'in_degree' + 'out_degree'
        order_vars = [(node, self.in_degree[node] + self.out_degree[node]) for node in self.in_degree.keys()]
        order_vars = sorted(order_vars, key=operator.itemgetter(1))
        return order_vars

    def get_hits_ranks(self,to_return='both'):
        # Returns ranks using 'hits' algorithm
        hubs, authorities = nx.hits_numpy(self.nx_graph)
        hubs = hubs.items()
        authorities = authorities.items()
        hubs = sorted(hubs, key=lambda x: x[1])
        authorities = sorted(authorities, key=lambda x: x[1])
        if to_return == 'both':  return hubs,authorities
        if to_return == 'hubs':  return hubs
        if to_return == 'authorities':   return authorities

    def get_page_ranks(self,reverse=False):
        # Ranks the nodes using PageRank algorithm
        if reverse: pr = nx.pagerank_numpy(self.reverse_nx_graph,alpha=0.9)
        else:       pr = nx.pagerank_numpy(self.nx_graph,alpha=0.9)
        order_vars = sorted(pr.items(), key=operator.itemgetter(1))
        return order_vars

    def max_in_degree(self,):
        """Returns the node with the maximum incoming arcs/edges"""
        node_key = max((self.in_degree).iteritems(), key=operator.itemgetter(1))[0]
        return node_key

    def second_max_in_degree(self,):
        """Returns the node with the second maximum incoming arcs/edges"""
        temp_graph = self.in_degree.copy()        
        max_in_degree = max((temp_graph).iteritems(), key=operator.itemgetter(1))[0]
        del temp_graph[max_in_degree]
        node_key = max((temp_graph).iteritems(), key=operator.itemgetter(1))[0]
        return node_key

    # Auxillary Functions (Independent) 
    def connect_unconnected_components(self,nodes):
        # given a list of nodes, find other nodes such that the final set of nodes is connected
        # step-1 : get connected components
        connections = self.get_connections(nodes=nodes)
        sub_graph = Graph(nodes=nodes,connections=connections)
        graphs = list(nx.connected_component_subgraphs(sub_graph.undirected_nx_graph))
        connected_components = [graph.nodes() for graph in graphs]
        while len(connected_components) > 1:
            # print 'num_connected_components ', len(connected_components), connected_components
            # Find the shortest path between all the non-connected components to get a pair of components to connect
            # just a very long path
            shortest_path = [i for i in range(100)]
            for index_component_1, component_1 in enumerate(connected_components):
                for index_component_2, component_2 in enumerate(connected_components[index_component_1+1:]):
                    for node_1 in component_1:
                        for node_2 in component_2:
                            if len(self.all_pairs_shortest_paths[node_1][node_2]) < len(shortest_path):
                                shortest_path = self.all_pairs_shortest_paths[node_1][node_2]

            # print self.all_pairs_shortest_paths
            nodes = list(set(nodes + shortest_path))
            connections = self.get_connections(nodes=nodes)
            sub_graph = Graph(nodes=nodes,connections=connections)
            graphs = list(nx.connected_component_subgraphs(sub_graph.undirected_nx_graph))
            connected_components = [graph.nodes() for graph in graphs]
            # print 'num_connected_components ', len(connected_components), connected_components

        return nodes

    def get_edge_lable(self,node,child):
        # Returns 'edge_lable' between node -> child
        if node+child in self.edge_lables.keys(): return self.edge_lables[node+child]
        return -1

    def transfer_ratio(self,node1,node2):
        path_n1_n2 = nx.shortest_path(self.nx_graph,source=node1,target=node2)
        transfer_ratio = 1
        while path_n1_n2 != []:
            # pop from the start of the remaining path
            new_node = path_n1_n2.pop(0)
            if new_node == node2:   return transfer_ratio
            transfer_ratio /= len(self._graph[new_node])
        return transfer_ratio

    def get_connections(self,nodes=[]):
        # Returns edges among the members of the given set of nodes
        connections = []
        nodes = set(nodes)
        for key in self._graph.keys():
            if key in nodes:
                for node in self._graph[key]:
                    if node in nodes:
                        connections.append([key,node])
        return connections

    def is_connected(self, node1, node2):
        """ Is node1 directly connected to node2 [i.e. There is an arc node1->node2] """
        return node1 in self._graph and node2 in self._graph[node1]

    def get_ancestor_list(self,node):
        """List of all the possible ancestors in the directed graph"""
        ancestor_list = [node]
        if len(self.reverse_graph[node]) == 0:   return ancestor_list
        parent_list = list(self.reverse_graph[node])
        while len(parent_list):
            current_parent = parent_list.pop(0)
            # print node, ancestor_list
            ancestor_list.extend(self.get_ancestor_list(current_parent))
        return ancestor_list

    def lowest_common_ancestor(self,node1,node2):
        # Returs the lowest common ancestor for node1 and node2
        node1_ancestor_list = self.get_ancestor_list(node1)
        node2_ancestor_list = self.get_ancestor_list(node2)
        if len(list(set(node1_ancestor_list).intersection(node2_ancestor_list))) == 0:  return node1
        else:   return list(set(node1_ancestor_list).intersection(node2_ancestor_list))[0]

    # Helper functions - Subtree and Topological ordering
    def get_sub_tree(self,node):
        # 'node' acts as the root
        # find all the connections using BFS
        start = node
        visited, queue = set(), [start]
        connections = []
        while queue:
            vertex = queue.pop(0)
            if vertex not in visited:
                visited.add(vertex)
                queue.extend([x for x in self._graph[vertex] if x not in visited])
                for node in self._graph[vertex]:
                    connections.append([vertex,node])

        return Graph(connections=connections,nodes=list(visited))

    def get_subtree_path(self,nodes,constraint=[]):
        # Returns all nodes in the subtrees of all nodes in path
        # one possibility for constraints is list of 'ARG's
        selected_nodes = set([])
        for node in nodes:
            selected_nodes.update(self.get_subtree( node=node,
                                                    selected_nodes=list(selected_nodes),
                                                    constraint=constraint))
        connections = self.get_connections(selected_nodes)
        return selected_nodes, Graph(connections=connections,nodes=set(selected_nodes))

    def get_name_path(self,nodes=[]):
        selected_nodes = set(nodes)
        for node in nodes:
            for child in self._graph[node]:
                if self.edge_lables[node+child].startswith(':name'):
                    selected_nodes.add(child)
        connections = self.get_connections(selected_nodes)
        return selected_nodes, Graph(connections=connections,nodes=set(selected_nodes))

    def get_subtree(self,node,selected_nodes=[],constraint=[]):
        # Returns the subtree for the given node
        selected_nodes.append(node)
        for child in self._graph[node]:
            if child not in selected_nodes:
                if constraint != []:
                    if self.get_edge_lable(node,child) in constraint:
                        self.get_subtree(node=child,selected_nodes=selected_nodes,constraint=constraint)
                else: self.get_subtree(node=child,selected_nodes=selected_nodes,constraint=constraint)
        return selected_nodes

    def get_topological_order_sub_graph(self,nodes=[]):
        topological_order_sub_graph = {}
        index = 0
        for key in sorted(self.topological_order):
            if self.topological_order[key] in nodes:
                topological_order_sub_graph[index] = self.topological_order[key]
                index += 1
        return topological_order_sub_graph

    def topological_ordering(self,):
        # add temp node
        temp_in_degree = self.in_degree.copy()
        queue = []
        for node in temp_in_degree.keys():
            if temp_in_degree[node] == 0:
                queue.append(node)
        # for node in self.in_degree.keys():  if self.in_degree[node] == 0:   queue.append(node)
        self.topological_order = {}
        topological_index = 1
        while(len(queue)):
            temp_node = queue.pop()
            self.topological_order[topological_index] = temp_node
            topological_index += 1
            for node in self._graph[temp_node]:
                temp_in_degree[node] -= 1
                if temp_in_degree[node] == 0:
                    queue.append(node)

        if False:
            # Find height of each element in the topologically sorted tree
            self.height = {}
            for i in range(topological_index,1,-1):
                new_node = self.topological_order[i]
                self.height[new_node] = max([self.height[node] for node in self._graph[new_node]]) \
                                             if len(self._graph[new_node]) != 0 else 0

            not_sorted = True
            # sort nodes in the order of height
            # in the same subsets, reorder topological order
            # Implemented like bubble sort
            while not_sorted:
                for i in range(1,len(self.topological_order)-1):
                    if self.height[self.topological_order[i]] > self.height[self.topological_order[i+1]]:   break

                # not_sorted = False if (i == len(self.topological_order)-1)
                for i in range(1,len(self.topological_order)-1):
                    if self.height[self.topological_order[i]] > self.height[self.topological_order[i+1]]:
                        temp_node = self.topological_order[i+1]
                        self.topological_order[i+1] = self.topological_order[i]
                        self.topological_order[i] = temp_node

    # Path functions
    def find_path(self, start, end, path=[],undirected=False):
        """ Find any path between start and end (may not be shortest) """
        path = path + [start]
        if start == end:
            return path
        if not self._graph.has_key(start):
            return []
        neighbours = self._graph[start]
        if undirected:  neighbours = neighbours + self.reverse_graph[start]
        neighbours = set(neighbours)
        for node in neighbours:
            if node not in path:
                newpath = self.find_path(node, end, path,undirected=False)
                if newpath: return newpath
        return []

    def find_all_paths(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return [path]
        if not self._graph.has_key(start):
            return []
        paths = []
        for node in self._graph[start]:
            if node not in path:
                newpaths = self.find_all_paths(node, end, path)
                for newpath in newpaths:
                    paths.append(newpath)
        return paths

    def find_shortest_path(self, start, end, path=[],undirected=False):
        # returns any shortest path between start and end
        path = path + [start]
        if start == end:
            return path
        if not self._graph.has_key(start):
            return None
        shortest = []
        neighbours = self._graph[start]
        if undirected:  neighbours = neighbours + self.reverse_graph[start]
        neighbours = set(neighbours)
        for node in neighbours:
            if node not in path:
                newpath = self.find_shortest_path(node,end,path=path,undirected=undirected)
                if len(newpath) > 0:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest

    # Construction Functions
    def fix_edge_lables(self,edge_lables={}):
        if edge_lables == {}:    return edge_lables
        for node in self._graph.keys():
            for child in self._graph[node]:
                if node+child not in edge_lables.keys() and child+node in edge_lables.keys():
                    edge_lables[node+child] = edge_lables[child+node]
                    del edge_lables[child+node]
        return edge_lables

    def get_sent_in_degree(self,):
        for var in self.var_to_sent:
            self.sent_in_degree[var] = len(set(self.var_to_sent[var]))

    def get_sent_to_var(self,):
        sent_to_var = {}
        for key in self.var_to_sent:
            for sent in self.var_to_sent[key]:
                try: sent_to_var[sent].append(key)
                except: sent_to_var[sent] = [key]
        return sent_to_var

    def generate_nx_graph(self,):
        # Generates the nx_graph for our graph
        # add edges in the graph
        self.nx_graph = nx.DiGraph()
        self.reverse_nx_graph = nx.DiGraph()
        self.undirected_nx_graph = nx.Graph()
        for key in self._graph.keys():
            self.nx_graph.add_node(key)
            self.undirected_nx_graph.add_node(key)
            for var in self._graph[key]: 
                try: self.nx_graph.add_edge(key,var,weight=self.weights[str(key)+' '+str(var)])
                except: self.nx_graph.add_edge(key,var)
                try: self.undirected_nx_graph.add_edge(key,var,weight=self.weights[str(key)+' '+str(var)])
                except: self.undirected_nx_graph.add_edge(key,var)
        for key in self.reverse_graph.keys():
            self.reverse_nx_graph.add_node(key)
            for var in self.reverse_graph[key]:
                try: self.reverse_nx_graph.add_edge(key,var,weight=self.weights[str(key)+' '+str(var)])
                except: self.reverse_nx_graph.add_edge(key,var)

    def add_nodes(self, nodes):
        """ Add nodes self._graph """
        for node in nodes:
            self._graph[node] = []
            if self._directed:
                self.reverse_graph[node] = []
                self.in_degree[node] = 0
                self.out_degree[node] = 0

    def add_connections(self, connections):
        """ Add connections (list of tuple pairs) to self._graph """
        for connection in connections:
            node1, node2 = connection[0], connection[1]
            self.add(node1, node2)
            if self._directed:
                self.in_degree[node2] += 1
                self.out_degree[node1] += 1

    def add(self, node1, node2):
        """ Add connection between node1 and node2 """

        self._graph[node1].append(node2)
        if self._directed:
            self.reverse_graph[node2].append(node1)
        else:
            self._graph[node2].append(node1)

    def remove(self, node):
        """ Remove all references to node """

        for n, cxns in self._graph.iteritems():
            if node in cxns: cxns.remove(node)
        for n, cxns in self.reverse_graph.iteritems():
            if node in cxns: cxns.remove(node)
        try:
            del self._graph[node]
            del self.reverse_graph[node]
        except KeyError:
            pass
        self.generate_nx_graph()

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
