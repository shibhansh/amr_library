from __future__ import division
from collections import defaultdict
import operator
import networkx as nx
import numpy
import scipy
import collections

from grasshopper import *
# data structure is represented as a 'dict of sets'
# for the directed case - The connection - (node1,node2) represents an arc from node1 to node2
# {'A': {'B'}} represents a self._graph with a single arc from A->B

num_sent_mis_match = 0

class Graph(object):
    """ Graph data structure, directed by default.
        Initializatin expects input of type - [['A', 'B'], ['B', 'C'], ['B', 'D']]
    """

    # Temp - Try to preserve the order of connections
    def __init__(self, connections,nodes=[], directed=True, edge_lables=dict(), weights=dict(),
                    var_to_sent={},common_text={},document_amr=False,text_index_to_var={},root=''):
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
        self.var_to_sent = dict(var_to_sent)
        self.sent_to_var = self.get_sent_to_var()

        self.add_nodes(nodes)
        self.add_connections(connections)

        self.edge_lables = self.fix_edge_lables(edge_lables=dict(edge_lables))
        self.common_text = common_text
        # get sent_in_degree, no. of edges coming from different sents
        self.get_sent_in_degree()
        self.topological_ordering()

        self.document_amr=document_amr
        # text_index_to_var - dict from text_index to 'var_set'
        self.text_index_to_var = text_index_to_var
        for index in self.text_index_to_var:
            self.text_index_to_var[index] = list(set(self.text_index_to_var[index]))

        self.root = root
        self.depth_dict = {}
        self.get_depth_dict_graph()

        # Add a 'nx' graph
        self.generate_nx_graph()
        # get all_pair_shortest_paths
        self.all_pairs_shortest_paths = {}

    # ranking functions
    def rank_pairs(self,ranks=[],weights=[],pairs_to_rank=10,to_remove_parent_transfer=False):
        if self.all_pairs_shortest_paths == {}:
            self.all_pairs_shortest_paths = nx.all_pairs_shortest_path(self.undirected_nx_graph)
        # find pairs with maximum weights - 
        if weights==[]: return []
        pair_weights = defaultdict(str)
        for index_n1, n1 in enumerate(ranks):
            for index_n2, n2 in enumerate(ranks):
                if n1!=n2 and n2 in self.all_pairs_shortest_paths[n1].keys(): 
                    total_weight = weights[index_n1] + weights[index_n2]
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

    def max_imp_path(self,ordered_pairs=[]):
        if self.all_pairs_shortest_paths == {}:
            self.all_pairs_shortest_paths = nx.all_pairs_shortest_path(self.undirected_nx_graph)
        current_path = []
        connections = []
        # traverse in decreasing order of weights
        paths_to_return = []
        # This step is probably too slow here
        for pair in ordered_pairs:
            first_var = pair.split()[0]
            second_var = pair.split()[1]
            # find the first sentence containing both the vars
            common_sents = sorted(list(set(self.var_to_sent[first_var]).intersection(self.var_to_sent[second_var])))
            print self.var_to_sent[first_var], self.var_to_sent[second_var], common_sents, first_var, second_var
            sent = -1
            for first_sent in sorted(common_sents):
                if first_sent in self.var_to_sent[second_var]:
                    sent_connections = self.get_connections(nodes=self.sent_to_var[first_sent])
                    sent_sub_graph = Graph(sent_connections,nodes=self.sent_to_var[first_sent])
                    print self.sent_to_var[first_sent]
                    print sent_sub_graph._graph
                    for graph in nx.connected_component_subgraphs(sent_sub_graph.undirected_nx_graph):
                        print graph.nodes()
                    if sent_sub_graph.all_pairs_shortest_paths == {}:
                       sent_sub_graph.all_pairs_shortest_paths = nx.all_pairs_shortest_path(sent_sub_graph.undirected_nx_graph) 

                    current_path = sent_sub_graph.all_pairs_shortest_paths[first_var][second_var]
                    if len(current_path) < 7:
                        sent = first_sent
                        break

            old_sent = sent

            try:    sent = min(common_sents)
            except: sent = -1

            global num_sent_mis_match
            if sent != old_sent:
                num_sent_mis_match += 1
            print 'num_sent_mis_match', num_sent_mis_match
            print 'old_sent', old_sent, 'sent', sent

            if sent != -1:
                pass
                # find path in this sentence
                sent_connections = self.get_connections(nodes=self.sent_to_var[sent])
                sent_sub_graph = Graph(sent_connections,nodes=self.sent_to_var[sent])
                if sent_sub_graph.all_pairs_shortest_paths == {}:
                   sent_sub_graph.all_pairs_shortest_paths = nx.all_pairs_shortest_path(sent_sub_graph.undirected_nx_graph) 
                current_path = sent_sub_graph.all_pairs_shortest_paths[first_var][second_var]
            # if sent not found, return the shortes_path
            else:
                current_path = self.all_pairs_shortest_paths[first_var][second_var]
            if len(current_path):# and len(current_path) < 9: 
                connections = self.get_connections(current_path)
                to_add_current_path = True

                if to_add_current_path:
                    paths_to_return.append([current_path,Graph(connections=connections,
                                                                nodes=set(current_path)),sent])
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

    # Node merging function
    def merge_nodes_in_graph(self,first_var='',second_var=''):
        # 1. check if not merging node with ancestor
        # 2. shift children of current node to new node
        # 3. Transfer incoming edges of the node to be replaced
        # 4. Update weights and 'var_to_sent' dictionaries
        # 5. Update text_index_to_var - it should take care of the 'alignments'
        
        # Step-1 Sanity checks
        # print first_var, second_var
        returned_value,first_var,second_var = \
                self.pre_merger_sanity_checks(first_var=first_var,second_var=second_var)

        if returned_value != -1:    return returned_value

        if second_var == first_var: return returned_value

        edges_to_merge = [':mod',':time',':location',':domain',':part',':unit',':quant',':degree',':source'] \
                         + [':op'+str(i) for i in range(20)] + [':ARG'+str(i) for i in range(20)]

        # Step-2 Shift children
        connections_to_remove = []
        for child in set(self._graph[second_var]):
            current_edge_set = self.edge_lables[second_var+' '+child]
            merge_child = False
            if child in self._graph[first_var]: merge_child = True
            for edge in current_edge_set:
                if edge in edges_to_merge:
                    merge_child = True

                    # add new connection with the 'first_var'
                    try:    self._graph[first_var].append(child)
                    except KeyError: self._graph[first_var] = [child]
                    self._graph[first_var] = list(set(self._graph[first_var]))

                    # add new edge lables in the new connection between 'first_var' and 'child'
                    try:    self.edge_lables[first_var+' '+child].extend(list(set(self.edge_lables[second_var+' '+child])))
                    except KeyError:    self.edge_lables[first_var+' '+child] = list(set(self.edge_lables[second_var+' '+child]))
                    self.edge_lables[first_var+' '+child] = list(set(self.edge_lables[first_var+' '+child]))

                    # update this new connection in 'reverse_graph'
                    self.reverse_graph[child].append(first_var)
                    self.reverse_graph[child].remove(second_var)
                    self.reverse_graph[child] = list(set(self.reverse_graph[child]))
                    break
            # update 'edge_lables'
            del self.edge_lables[second_var+' '+child]

            # Find the connections to delete
            if merge_child == False:
                connections_to_remove.append(child)

        for node in connections_to_remove:
            self.remove_connection(node1=second_var,node2=node)

        # Shift incoming nodes
        self.reverse_graph[second_var] = list(set(self.reverse_graph[second_var]))
        for parent in self.reverse_graph[second_var]:
            self._graph[parent].append(first_var)
            self._graph[parent].remove(second_var)

            try:    self.edge_lables[parent+' '+first_var].extend(list(set(self.edge_lables[parent+' '+second_var])))
            except: self.edge_lables[parent+' '+first_var] = list(set(self.edge_lables[parent+' '+second_var]))

            del self.edge_lables[parent+' '+second_var]
            self.edge_lables[parent+' '+first_var] = list(set(self.edge_lables[parent+' '+first_var]))

            self.reverse_graph[first_var].append(parent)

        self.reverse_graph[first_var]=list(set(self.reverse_graph[first_var]))

        # Step-4 Update weights and 'var_to_sent' dictionaries
        try:
            self.weights[first_var] += self.weights[second_var]
            del self.weights[second_var]
        except KeyError:    pass

        self.var_to_sent[first_var].extend(self.var_to_sent[second_var])

        # Step-5 Update text_index_to_var - it should take care of the 'alignments'
        for index in self.text_index_to_var:
            if second_var in self.text_index_to_var[index]:
                self.text_index_to_var[index].remove(second_var)
                self.text_index_to_var[index].append(first_var)
                self.text_index_to_var[index] = list(set(self.text_index_to_var[index]))
        self.remove(second_var)
        self.get_depth_dict_graph()

        # Remove the nodes that aren't reachable from the 'root' node
        nodes_to_remove = [node for node in self._graph.keys() if not node in self.shortest_root_paths.keys()]
        for node in nodes_to_remove:
            self.remove(node)

        # Update var_to_sent and sent_to_var
        self.sent_to_var = self.get_sent_to_var()
        for sent,sent_vars in self.sent_to_var.iteritems():
            sent_connections = self.get_connections(nodes=sent_vars)
            sent_sub_graph = Graph(sent_connections,nodes=sent_vars)
            connected_components = nx.connected_component_subgraphs(sent_sub_graph.undirected_nx_graph)
            if len(connected_components) == 1: continue
            # find the largest component
            largest_component = []

            for graph in connected_components:
                if len(graph.nodes()) > len(largest_component):
                    largest_component = graph.nodes()

            for node in sent_vars:
                if node not in largest_component:
                    self.var_to_sent[node].remove(sent)
        self.sent_to_var = self.get_sent_to_var()
 
        return returned_value
        # todo- handle the edge lable cases of '-in'

    def pre_merger_sanity_checks(self,first_var='',second_var='',debug=False):
        # Return values - 
        # 0 - Didn't merge
        # 1 - No merger needed
        # -1 - Passed 'pre_merger_sanity_tests'

        # 1. If vars are in the same sentence or if they are same, no merger needed
        # 2. Not derging dates
        # 3. Various checks on merging the named entities
        # 4. Not merging same vars if one in parent of other
        # 5. Not merging if they have common 'ARGs' for now'

        # Check-1
        common_sents = list(set(self.var_to_sent[first_var]).intersection(self.var_to_sent[second_var]))
        if len(common_sents)>0:
            if debug: 
                print 'No merging needed - same sentence'
                return 1, first_var, second_var

        if first_var == second_var:
            if debug: print 'No merging needed - same variable'
            return 1, first_var, second_var

        # Check-2
        if 'date-entity' in self.common_text[first_var] + self.common_text[second_var]:
            if debug: 
                print 'Can not merge - Not merging dates'
            return 0, first_var, second_var
        # Check-3
        op_list_first_node = []
        op_list_second_node= []

        # For every node, get the op_list if it has a child with edge ':name'
        for parent in self.reverse_graph[first_var]:
            if self.depth_dict[first_var] == self.depth_dict[parent] +1:
                for edge in self.edge_lables[parent+' '+first_var]:
                    if edge.startswith(':name'):
                        first_var = parent
                        break

        for parent in self.reverse_graph[second_var]:
            if self.depth_dict[second_var] == self.depth_dict[parent] +1:
                for edge in self.edge_lables[parent+' '+second_var]:
                    if edge.startswith(':name'):
                        second_var = parent
                        break

        op_list_second_node = self.get_op_list(var=first_var)
        op_list_first_node = self.get_op_list(var=second_var)

        if debug:
            print op_list_first_node, op_list_second_node
            print self.check_mutual_sublist(first_list=op_list_first_node,second_list=op_list_second_node)
            print self.check_initials(first_list=op_list_first_node,second_list=op_list_second_node,debug=True)

        # Special check for the case of merging two nodes that contains ':name'
        # Don't merge nodes with different names
        if not self.check_mutual_sublist(first_list=op_list_first_node,second_list=op_list_second_node):
            # don't merge if one isn't a sublist of other except when one is in the form of initials
            if self.check_initials(first_list=op_list_first_node,second_list=op_list_second_node):
                pass
            else:
                if debug: print 'Can not merge - Different names', op_list_first_node, op_list_second_node
                return 0, first_var, second_var

        # Check-4
        if second_var in self.get_sub_tree(node=first_var)._graph.keys():   return 1, first_var, second_var

        if first_var in self.get_sub_tree(node=second_var)._graph.keys():   return 1, first_var, second_var

        # Check-5
        edge_set_first_node = []
        edge_set_second_node = []
        for child in self._graph[first_var]:
            edge_set_first_node.extend(self.edge_lables[first_var+' '+child])

        for child in self._graph[second_var]:
            edge_set_second_node.extend(self.edge_lables[second_var+' '+child])

        for edge in edge_set_first_node:
            if '-of' not in edge and edge.startswith(':ARG') and edge in edge_set_second_node:
                if debug: print 'Can not merge - Maybe common args'
                return 0, first_var, second_var

        return -1, first_var, second_var

    def merge_same_children(self,):
        while True:
            found_nodes_to_merge = True
            for node in self._graph:
                children_common_text = [self.common_text[var] for var in self._graph[node]]
                if len(children_common_text) == len(set(children_common_text)):
                    continue
                duplicates = [item for item, count in collections.Counter(children_common_text).items() if count > 1]
                for duplicate_text in duplicates:
                    vars_to_merge = [var for var in self._graph[node] if self.common_text[var]==duplicate_text]
                    # check that subtree is same as well
                    children_subtrees = [self.get_subtree(node=var) for var in vars_to_merge]
                    if len(children_subtrees) == len(set(   )):
                        continue
                    duplicate_subtree = \
                        [item for item, count in collections.Counter(children_subtrees).items() if count > 1]

                    indices = [i for i, x in enumerate(children_subtrees) if x == duplicate_subtree[0]]
                    first_var = vars_to_merge[indices[0]]
                    second_var = vars_to_merge[indices[1]]
                    self.merge_nodes_in_graph(first_var=first_var,second_var=second_var)

                    found_nodes_to_merge = True
                    if found_nodes_to_merge:    break    

                if found_nodes_to_merge:    break    

            if not found_nodes_to_merge:    break    

    def get_op_list(self,var=''):
        # Returns if the node has any children with edge ':name'
        # Example - Input - :name (var2 / name :op1 "ABS-CBN" :op2 "News")))
        #           Output - ['ABS-CBN', 'News']
        text = ''
        for child_var in self._graph[var]:
            for edge in self.edge_lables[var+' '+child_var]:
                if edge.startswith(':name'):
                    text = self.common_text[child_var]

        if text == '':  return []

        text = text.strip(')')
        text = text.split('/')[1]
        text = text.split()
        op_list = []
        for index_word, word in enumerate(text):
            if word.startswith(':op'): op_list.append(text[index_word+1].lower())
        op_list = [word for word in op_list if word not in ['','""']]
        return op_list

    def check_initials(self,first_list=[],second_list=[],debug=False):
        # return True if and only if one is initials of other
        if not (len(first_list) == 1 or len(second_list) == 1): return False

        first_list = [x.strip('"') for x in first_list]
        second_list = [x.strip('"') for x in second_list]

        if debug:   print first_list,second_list

        if len(first_list) == 1:
            if first_list[0] == ''.join([x[0] for x in second_list]):   return True
        if len(second_list) == 1:
            if second_list[0] == ''.join([x[0] for x in first_list]):   return True
        return False

    def check_mutual_sublist(self,first_list=[],second_list=[]):
        first_sub_list = True
        second_sub_list = True
        for word in first_list:
            if word not in second_list:
                first_sub_list = False
                break
        for word in second_list:
            if word not in first_list:
                second_sub_list = False
                break
        if first_sub_list or second_sub_list:   return True
        else: return False

    # 'AMR-text' generation functions
    def generate_text_amr(self,):
        # Function to generate textual representation of AMR from the directed graph
        list_of_variables,depth_list = self.get_var_list_from_directed_graph()

        text_list_sub_graph = self.get_text_list(list_of_variables,depth_list)

        return text_list_sub_graph

    def get_var_list_from_directed_graph(self):
        # Get the list of vars for the 'text-AMR' representation 
        depth_dict = self.depth_dict
        def dfs(root,depth,depth_list=[],ordered_list=[],consturcted_list=[]):
            already_visited = False
            if root in set(ordered_list+consturcted_list):  already_visited = True

            # Preserving the order of children
            ordered_list.append(root)
            depth_list.append(depth)
            if already_visited :    return ordered_list

            # order children in ':name', ':ARGx', 'op', ':mod', ':time', others ,'ARGx-of'
            children_list =  self._graph[root]
            children_list = self.get_children_order(node=root,child_list=list(children_list))
            for child in children_list:
                ordered_list = dfs(child,depth+1,depth_list,ordered_list,consturcted_list)
            return ordered_list

        ordered_list = []
        depth_list = []
        # find a root node
        new_root = None
        for node in self.get_depth_order(self._graph.keys(),depth_dict=depth_dict):
            if len(self.reverse_graph[node]) == 0:
                new_root = node
                break
        if new_root == None:    return self._graph.keys(),[0]*len(self._graph.keys())

        # traverse and include the nodes in the new 
        depth = 0
        ordered_list = dfs(new_root,depth,depth_list,ordered_list=[],consturcted_list=ordered_list)

        while len(set(ordered_list)) != len(self._graph.keys()):
            # find a node connected to the graph consturcted so far
            new_node_found = False
            new_root = None
            # todo - add an order for node selection
            for node in self.get_depth_order(set(self._graph.keys())-set(ordered_list),depth_dict=depth_dict):
                for child_node in self._graph[node]:
                    if child_node in ordered_list:
                        new_root = node
                        index_to_append_at = ordered_list.index(child_node)
                        # find the location of the definition of the 'var'
                        for index,var in enumerate(ordered_list):
                            if index == len(ordered_list) -1:   continue
                            if var == child_node:
                                if depth_list[index] <  depth_list[index+1]:
                                    index_to_append_at = index 
                                    break

                        depth = depth_list[index_to_append_at]

                        temp_depth_list = list(depth_list)
                        temp_ordered_list = list(ordered_list)

                        try:
                            index_to_append_at +=next(x[0] for x in enumerate(temp_depth_list[index_to_append_at+1:])\
                                                            if x[1] <= temp_depth_list[index_to_append_at])
                        except:
                            index_to_append_at = len(ordered_list) -1


                        new_node_found = True

                        break
                if new_node_found:  break
            new_depth_list = []
            temp_list = dfs(new_root,depth+1,new_depth_list,
                                ordered_list=[],consturcted_list=ordered_list)
            try: 
                if child_node in temp_list:
                    child_index = temp_list.index(child_node)
                temp_list.pop(child_index)
                new_depth_list.pop(child_index)
            except: pass

            ordered_list[index_to_append_at+1 : index_to_append_at+1] = temp_list
            depth_list[index_to_append_at+1 : index_to_append_at+1] = new_depth_list

        return ordered_list,depth_list

    def get_text_list(self,list_of_variables,depth_list):
        # get 'AMR-text' from the 'var_list'
        # adding attributes just to take ease the process of text list formation
        amr_node_list = []
        text_list = []
        previous_higher_depth_index = 0
        num_closing_brackets_to_add = 0

        for index_variable,variable in enumerate(list_of_variables):
            new_node_dict = {}
            new_node_dict['depth'] = depth_list[index_variable]
            new_node_dict['variable'] = variable
            new_node_dict['common_text'] = self.common_text[variable]

            # if var has been defined, its common_text will be ''
            for index,var in enumerate(list_of_variables[:index_variable]):
                if var != variable: continue
                else:
                    if amr_node_list[index]['common_text'] != '':   new_node_dict['common_text'] = ''


            # check if there is another location where the var has been defined
            for forward_index,forward_var in enumerate(list_of_variables):
                if forward_index <= index_variable: continue
                if forward_index == len(list_of_variables) -1: continue 
                if forward_var != variable: continue
                else:
                    if depth_list[forward_index+1] <= depth_list[forward_index]:    continue
                    else: new_node_dict['common_text'] = ''

            temp_depth_list = depth_list[:index_variable]
            temp_depth_list.reverse()
            if index_variable +1 < len(list_of_variables):
                if depth_list[index_variable] >= depth_list[index_variable+1]:
                    num_closing_brackets_to_add = 1 + depth_list[index_variable] - depth_list[index_variable+1]
                else:
                    num_closing_brackets_to_add = 0
            else:
                num_closing_brackets_to_add = 1 + depth_list[index_variable]

            if new_node_dict['depth'] == 0:
                parent_index_new_amr = -1
                # if root text need to be changed as well
                new_node_dict['text'] = '('+ new_node_dict['variable'] + ' ' + new_node_dict['common_text']\
                                        + ')'*num_closing_brackets_to_add
            else:
                parent_index_new_amr=(len(temp_depth_list)-1) - temp_depth_list.index(new_node_dict['depth']-1)
                amr_node_list[parent_index_new_amr]['children_list'].append(index_variable)

                try:
                    edge = self.edge_lables[amr_node_list[parent_index_new_amr]['variable']+' '+new_node_dict['variable']][0]
                except KeyError:
                    edge = self.edge_lables[new_node_dict['variable']+' '+amr_node_list[parent_index_new_amr]['variable']][0]
                    if '-of' not in edge:   edge = edge + '-of'
                    else:   edge = edge[:-3]

                new_node_dict['text'] = edge + ' (' + new_node_dict['variable'] + ' ' \
                                    + new_node_dict['common_text'] + ')'*num_closing_brackets_to_add

                if new_node_dict['common_text'] == '':
                    new_node_dict['text'] = edge + ' ' + new_node_dict['variable'] + ')'*(num_closing_brackets_to_add-1)

            new_node_dict['children_list'] = []
            amr_node_list.append(new_node_dict)
            text_list.append(' '*6*new_node_dict['depth']+new_node_dict['text'])
        return text_list

    def get_depth_order(self,nodes=[],depth_dict={}):
        # Function to return the nodes in order dictated by their depth
        relevant_tuples = []
        for key in depth_dict:
            if key in nodes:
                relevant_tuples.append((key, depth_dict[key]))

        relevant_tuples = sorted(relevant_tuples, key=lambda x: x[1])
        # just return the list of vars s.t. first var has the least depth 
        return [x[0] for x in relevant_tuples]

    def get_children_order(self,node='',child_list=[]):
        # From the node, get the order of children according to edge lables
        ordered_children_list = []
        parent_var = node
        # order children in ':name', ':ARGx', 'op', ':mod', ':time', others ,'ARGx-of'
        order_children = [':name'] + [':ARG'+str(x) for x in range(20)] + [':op'+str(x) for x in range(20)] + \
                            [':mod',':time',':location',':part'] + []
        relevant_var_edge_dict = {}
        relevant_var_order_dict = {}
        # print node, self._graph[node]
        for var in self._graph[node]:
            if var in child_list:
                for current_edge in self.edge_lables[parent_var+' '+var]:
                    relevant_var_edge_dict[var] = current_edge

                    if current_edge in order_children:  relevant_var_order_dict[var] = order_children.index(current_edge)
                    elif '-' in current_edge:   relevant_var_order_dict[var] = 1000
                    else:   relevant_var_order_dict[var] = 100

        sorted_relevant_var_order_dict = sorted(relevant_var_order_dict.items(), key=operator.itemgetter(1))
        ordered_children_list = [x[0] for x in sorted_relevant_var_order_dict]
        if set(child_list) != set(ordered_children_list):
            ordered_children_list.extend(list(set([x for x in child_list if x not in ordered_children_list])))
        return ordered_children_list

    def get_depth_dict_graph(self,):
        # Get Depth dictionary accoding to the root node
        self.depth_dict = {}
        self.pseudo_nx_graph = nx.Graph()
        if len(self._graph.keys()) == 0:    return
        for key in self._graph.keys():
            self.pseudo_nx_graph.add_node(key)
            for var in self._graph[key]: 
                self.pseudo_nx_graph.add_edge(key,var)

        if self.root == '':
            # search for nodes
            for node in self.reverse_graph.keys():
                # print node, self.reverse_graph[node]
                if len(self.reverse_graph[node]) == 0:
                    self.root = node 
                    break
        if self.root == '': self.root = self._graph.keys()[0]

        self.shortest_root_paths = nx.single_source_shortest_path(self.pseudo_nx_graph,source=self.root)

        for key in self._graph.keys():
            try:    self.depth_dict[key] = len(self.shortest_root_paths[key])
            except KeyError: self.depth_dict[key] = float("inf")

    # Auxillary Functions (Independent) 
    def connect_unconnected_components(self,nodes):
        if self.all_pairs_shortest_paths == {}:
            self.all_pairs_shortest_paths = nx.all_pairs_shortest_path(self.undirected_nx_graph)
        # given a list of nodes, find other nodes such that the final set of nodes is connected
        # step-1 : get connected components
        for node in nodes:
            if node not in self._graph.keys():
                print 'some issue'
                0/0

        connections = self.get_connections(nodes=nodes)
        sub_graph = Graph(nodes=nodes,connections=connections)
        graphs = list(nx.connected_component_subgraphs(sub_graph.undirected_nx_graph))
        connected_components = [graph.nodes() for graph in graphs]
        while len(connected_components) > 1:
            # Find the shortest path between all the non-connected components to get a pair of components to connect
            # just a very long path
            shortest_path = [i for i in range(100)]
            for index_component_1, component_1 in enumerate(connected_components):
                for index_component_2, component_2 in enumerate(connected_components[index_component_1+1:]):
                    for node_1 in component_1:
                        for node_2 in component_2:
                            if len(self.all_pairs_shortest_paths[node_1][node_2]) < len(shortest_path):
                                shortest_path = self.all_pairs_shortest_paths[node_1][node_2]

            nodes = list(set(nodes + shortest_path))
            connections = self.get_connections(nodes=nodes)
            sub_graph = Graph(nodes=nodes,connections=connections)
            graphs = list(nx.connected_component_subgraphs(sub_graph.undirected_nx_graph))
            connected_components = [graph.nodes() for graph in graphs]

        return nodes

    def get_edge_lable(self,node,child):
        # Returns 'edge_lable' between node -> child
        if node+child in self.edge_lables.keys(): return self.edge_lables[node+' '+child]
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
                for lable in self.edge_lables[node+' '+child]:
                    if lable.startswith(':name'):
                        selected_nodes.add(child)
        connections = self.get_connections(selected_nodes)

        selected_edge_lables = {}
        for pair in self.edge_lables:
            parent, child = pair.split(' ')
            if parent in selected_nodes and child in selected_nodes:
                selected_edge_lables[pair] = self.edge_lables[pair][0]

        selected_weights = {}
        for key in self.weights:
            if key in selected_nodes:
                selected_weights[key] = self.weights[key]

        selected_var_to_sent = {}
        for key in self.var_to_sent:
            if key in selected_nodes:
                selected_var_to_sent[key] = self.var_to_sent[key]

        selected_common_text = {}
        for key in self.common_text:
            if key in selected_nodes:
                selected_common_text[key] = self.common_text[key]

        return selected_nodes, Graph(connections=connections,nodes=set(selected_nodes),
                                    edge_lables=selected_edge_lables,weights=selected_weights,
                                    var_to_sent=self.var_to_sent,common_text=selected_common_text)

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
        # write a code to dfs traverse the graph and find any topological ordering
        # add temp node
        temp_in_degree = self.in_degree.copy()
        queue = []
        for node in temp_in_degree.keys():
            if temp_in_degree[node] == 0:
                queue.append(node)
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
        fixed_edge_lables = {}
        for key in edge_lables:
            parent, child = key.split(' ')
            lable = edge_lables[key]
            if '-' in lable:
                child, parent = parent, child
                lable = lable[ : lable.index('-')]
                try:    fixed_edge_lables[parent+' '+child].append(lable)
                except: fixed_edge_lables[parent+' '+child] = [lable]
            else:
                try:    fixed_edge_lables[parent+' '+child].append(lable)
                except: fixed_edge_lables[parent+' '+child] = [lable]

        for key in fixed_edge_lables:
            for lable in fixed_edge_lables[key]:
                if '-' in fixed_edge_lables[key]:
                    print 'some error'
                    print self._graph
                    print fixed_edge_lables
                    print key, fixed_edge_lables[key]
                    0/0
        return fixed_edge_lables

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
            if key == "Rinfinite":  continue
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
                if var == "Rinfinite":  continue
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

    def remove_connection(self,node1,node2):
        self._graph[node1].remove(node2)
        self.reverse_graph[node2].remove(node1)
        try:
            del self.edge_lables[node1+' '+node2]
            del self.edge_lables[node2+' '+node1]
        except KeyError:    pass

    def remove(self, node,reconstruct_nx=False):
        """ Remove all references to node """
        for n, cxns in self._graph.iteritems():
            while node in cxns:
                cxns.remove(node)
        for n, cxns in self.reverse_graph.iteritems():
            while node in cxns:
                cxns.remove(node)
        for n, cxns in self.text_index_to_var.iteritems():
            while node in cxns:
                cxns.remove(node)

        try:    del self._graph[node]
        except KeyError:    pass
        try:    del self.reverse_graph[node]
        except KeyError:    pass
        try:    del self.common_text[node]
        except KeyError:    pass
        try:    del self.var_to_sent[node]
        except KeyError:    pass

        if reconstruct_nx:  self.generate_nx_graph()

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
