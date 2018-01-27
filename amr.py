from directed_graph import Graph
from concept_relation_list import concept_relation_list
import copy
import sys
import operator

# It's the AMR class - saves the text representation of a single AMR

# Global variables to measure effeciency of mergers
count = 0
total_count=0
interest  = 0

class AMR(object):
	""" 
		Graph data structure, directed by default.
	    Initializatin expects input of type - [['A', 'B'], ['B', 'C'], ['B', 'D']]
		The attributes associated with AMR are - 'parent_index','children_list','depth','no_of_children',
		'child_number','text','variable','variable_start_index','variable_end_index','common_text'
	"""

	def __init__(self, text_list=[],amr_with_attributes=False,text='',alignments=[],var_to_sent={},
						sent_index=0):
		# If the 'amr' that we get doesn't has attributes, it is just as text,
		# 	i.e. each element of the list is just a line of the text
		# else the amr that has all the attributes, and it is in the required form
		self.text_list = text_list
		self.amr = self.text_list
		# mapping from 'variables' to indices in self.amr
		self.var_to_index = {}
		if amr_with_attributes == False:
			# add attributes
			self.add_attributes()
			# add other attributes like 'variable_start_index'
			self.add_variable_info()
		# contains the edge lable for every class
		self.edges = {}
		self.connections = self.get_edge_info()

		self.get_var_to_index_mapping()

		# Contains all the 'variables' in the list
		self.nodes = self.get_node_info()

		# get 'var_to_sent'
		if var_to_sent == {}:
			for key in self.var_to_index.keys():	var_to_sent[key] = [sent_index]
		self.var_to_sent = var_to_sent

		self.directed_graph = Graph(connections=self.connections,nodes=self.nodes,
										edge_lables=self.edges,var_to_sent=self.var_to_sent)

		self.topological_order = self.directed_graph.topological_order
		# self.text is a list of sentences in case of a document AMR
		self.text = text
		self.split_text = (' '.join(self.text)).split()
		self.alignments = None
		self.get_alignments(alignments)
		# Not updated while mering any 2 nodes
		self.get_sentence_boundaries_amr()

		self.get_text_index_to_var()

		# get detph_list
		self.depth_dict = {}
		self.get_depth_dict()


	# Sentence Graph functions
	def get_sentence_reference_graph(self,):
		# a graph containing sentence to sentence links
		self.get_sentence_boundaries_amr()
		sentence_connections = []
		weights = {}
		for index_node, node in enumerate(self.amr):
			current_sent_index = self.node_index_to_sent_index(index_node)
			if current_sent_index == -1:	continue
			current_var = node['variable']
			for location in self.var_to_index[current_var]:
				location_sent_index = self.node_index_to_sent_index(location)
				if location_sent_index != current_sent_index:
					if (current_sent_index, location_sent_index) not in sentence_connections:
						sentence_connections.append((current_sent_index, location_sent_index))
						sentence_connections.append((location_sent_index, current_sent_index))
						weights[str(current_sent_index)+' '+str(location_sent_index)] = 1
						weights[str(location_sent_index)+' '+str(current_sent_index)] = 1
					else:
						weights[str(current_sent_index)+' '+str(location_sent_index)] += 1
						weights[str(location_sent_index)+' '+str(current_sent_index)] += 1

		self.sentence_reference_graph = Graph(connections=sentence_connections,
													nodes=range(0,len(self.sentence_boundries)),
													weights=weights)

	# Merging - Core Functions
	def merge_named_entities(self,):
		# Desined specifically to run initially, may not work if run after some other mergers
		# name list 
		existing_names = []
		node_merged = False
		for index_node,node in enumerate(self.amr):
			parent_index_current_node = self.amr[index_node]['parent_index']
			if ':name ' in node['text']:
				node_merged = False
				for index_existing_node in existing_names:
					if self.amr[index_existing_node]['common_text'].strip() == node['common_text'].strip():
						parent_index_existing_node = self.amr[index_existing_node]['parent_index']
						if self.amr[parent_index_existing_node]['common_text'] == \
							self.amr[parent_index_current_node]['common_text']:
							# If successfull merger, restart merging
							successfull_merge = self.merge_nodes(first_node_index=parent_index_existing_node\
								,second_node_index=parent_index_current_node)
							if successfull_merge == 2:
								return 1
				if not node_merged:
					existing_names.append(index_node)
		return 0

	def merge_date_entites(self,):
		existing_dates = []
		for index_node,node in enumerate(self.amr):
			node_merged = False
			if 'date-entity ' in node['text']:
				for index_existing_node in existing_dates:
					if self.amr[index_existing_node]['common_text'].strip() == node['common_text'].strip():
						self.merge_nodes(first_node_index=index_existing_node,second_node_index=index_node)
						return 1
				if not node_merged:
					existing_dates.append(index_node)
		return 0

	def merge_nodes(self,first_alignment=[],second_alignment=[],first_node_index=None,second_node_index=None,
		debug=False):
		# steps in the procedure - 
		# 1. sanity checks
		# 2. Merging subtrees
		# 3. Reconstruct AMR
		# move subtree of the second node to first node
		# Return values - 
		# 0 - Didn't merge
		# 1 - No merger needed
		# 2 - Successfull merge
		returned_value, first_node_index, second_node_index, common_variable_name, removed_variable = \
				self.pre_merging_sanity_tests(first_alignment=first_alignment,
												second_alignment=second_alignment,
												first_node_index=first_node_index,
												second_node_index=second_node_index,
												debug=debug)

		if returned_value != -1:	return returned_value

		text_list = self.move_subtree(first_node_index,second_node_index,common_variable_name)

		if text_list == []:
			if debug: print 'Can not merge - Trying to merge node with ancestor' 
			return 0
		text_list =[line + '\n' for line in text_list]

		# update var_to_sent list
		self.var_to_sent[common_variable_name] = list(set(self.var_to_sent[common_variable_name]+\
															self.var_to_sent[removed_variable]))

		self.reconstruct_amr(text_list=text_list)
		self.merge_multiple_same_var_children()
		self.post_merging_sanity_tests()		

		return 2

	def pre_merging_sanity_tests(self,first_alignment=[],second_alignment=[],first_node_index=None
		,second_node_index=None,debug=False):
		# Return values - 
		# 0 - Didn't merge
		# 1 - No merger needed
		# -1 - Passed 'pre_merger_sanity_tests'
		# Check for valid alginments
		# print '1 ',first_alignment, second_alignment, first_node_index, second_node_index
		common_variable_name = ''
		removed_variable = ''

		if first_alignment != [] and second_alignment != []:
			if first_alignment[1] == second_alignment[1]:
				if debug: print 'No merging needed - same sentence'
				return 1, first_node_index, second_node_index, common_variable_name, removed_variable
			if first_alignment[-1] == 'r' or second_alignment[-1] == 'r':
				if debug: print 'Can not merge - Invalid alignment'
				return 0, first_node_index, second_node_index, common_variable_name, removed_variable

		# Get alignments if alignments not given
		if first_node_index == None:
			first_node_index = self.alignment_to_node_index(first_alignment)
		if second_node_index == None:
			second_node_index = self.alignment_to_node_index(second_alignment)

		# Not merging dates for now
		if 'date-entity' in self.amr[first_node_index]['text']:
			if debug: print 'Can not merge - Not merging dates'
			return 0, first_node_index, second_node_index, common_variable_name, removed_variable

		if 'date-entity' in self.amr[second_node_index]['text']:
			if debug: print 'Can not merge - Not merging dates'
			return 0, first_node_index, second_node_index, common_variable_name, removed_variable

		op_list_first_node = []
		op_list_second_node= []
		first_node_index = self.var_to_index[self.amr[first_node_index]['variable']][0]
		second_node_index = self.var_to_index[self.amr[second_node_index]['variable']][0]

		# For every node, get the op_list if it has a child with edge ':name'
		if self.amr[first_node_index]['text'].startswith(':name'):
			if self.amr[first_node_index]['parent_index'] != 0:
				first_node_index = self.amr[first_node_index]['parent_index']

		if self.amr[second_node_index]['text'].startswith(':name'):
			if self.amr[second_node_index]['parent_index'] != 0:
				second_node_index = self.amr[second_node_index]['parent_index']

		op_list_second_node = self.get_op_list(index=second_node_index)
		op_list_first_node = self.get_op_list(index=first_node_index)

		# Special check for the case of merging two nodes that contains ':name'
		# Don't merge nodes with different names
		if not self.check_mutual_sublist(first_list=op_list_first_node,second_list=op_list_second_node) and \
			self.check_initials(first_list=op_list_first_node,second_list=op_list_second_node):
			if debug: print 'Can not merge - Different names', op_list_first_node, op_list_second_node
			return 0, first_node_index, second_node_index, common_variable_name, removed_variable

		common_variable_name = self.amr[first_node_index]['variable']
		removed_variable = self.amr[second_node_index]['variable']
		# Check if merging is needed or not
		if common_variable_name == removed_variable:
			if debug: print 'No merging needed - same variable'
			return 1, first_node_index, second_node_index, common_variable_name, removed_variable
		if removed_variable == self.amr[self.amr[first_node_index]['parent_index']]['variable']:
			if debug: print 'No merging needed - same parent variable'
			return 1, first_node_index, second_node_index, common_variable_name, removed_variable
		if common_variable_name == self.amr[self.amr[second_node_index]['parent_index']]['variable']:
			if debug: print 'No merging needed - same parent variable'
			return 1, first_node_index, second_node_index, common_variable_name, removed_variable

		# Don't merge if they have common 'ARGs' as children, ex. if both have 'ARG0' as a child
		children_edges_first_node = self.get_edges_children(first_node_index)
		children_edges_second_node = self.get_edges_children(second_node_index)
		for edge in children_edges_first_node:
			if '-of' not in edge and edge.startswith(':ARG') and edge in children_edges_second_node:
				if debug: print 'Can not merge - Maybe common args'
				return 0, first_node_index, second_node_index, common_variable_name, removed_variable

		# Sanity check passed - return -1
		return -1, first_node_index, second_node_index, common_variable_name, removed_variable

	def move_subtree(self,first_node_index,second_node_index,new_name=''):
		# make a new copy of the text list, at the location where,the var has some children(saved in last step)
		# traverse in reverse order (accordingly do) -
		# 1. change var name
		# 2. remove children
		# 3. add children, also fix the depth
		# 4. Sanity check so that node doesn't become its own ancestor
		# 5. Fix number of brackets
		# 6. Update alginments, wherever they are needed

		text_list = []
		depth_list = []
		global count
		# For the first variable find the location where the variable is difined
		for index in self.var_to_index[self.amr[first_node_index]['variable']]:
			if self.amr[index]['children_list'] != []:
				first_node_index = index
				break
		# Creat initial copies of depth and text lists
		for node in self.amr:
			text_list.append(node['text'])
			depth_list.append(node['depth'])

		first_node_depth = self.amr[first_node_index]['depth']
		second_node_depth = self.amr[second_node_index]['depth']
		new_name = self.amr[first_node_index]['variable']
		previous_name = self.amr[second_node_index]['variable']

		# get a list of indices to traverse:
		# 	1. Index corresponding to the definition of first variable
		# 	2. Index of the second variable, and all its occurances
		indices_to_traverse = []
		indices_to_traverse = self.var_to_index[self.amr[second_node_index]['variable']] + [first_node_index]
		indices_to_traverse.sort()
		indices_to_traverse.reverse()

		collected_children = []
		num_closing_brackets_to_add = 0
		index_to_insert_at = first_node_index

		# Steps 1,2
		# Traverse the list in reverse order, updating var_names, get 'collected_children' that have to be moved
		for index in indices_to_traverse:
			# Upon reaching the first node, update that node in the 'text_list'
			if index == first_node_index:
				if collected_children != []:
					# if opening bracket missing, add it
					variable_start_index = self.amr[index]['variable_start_index']
					if '(' not in text_list[index]: 
						text_list[index] = text_list[index][:variable_start_index] + '(' + \
											text_list[index][variable_start_index:]
					text_list[index] = text_list[index].strip(')')
			else:
				# If at the difinition of second variable
				if len(self.amr[index]['children_list']) != 0:
					# Update info for the locaiton of second variable difinition
					second_node_index = index
					second_node_depth = depth_list[index]

					if index_to_insert_at+1 < len(self.amr):next_depth = self.amr[index_to_insert_at+1]['depth']
					else:	next_depth = 0
					# update first_node_index if it occurs before second_node_index in reverse order traversal
					if index < first_node_index:
						variable_start_index = self.amr[first_node_index]['variable_start_index']
						if '(' not in text_list[first_node_index]: 
								text_list[first_node_index] = text_list[first_node_index][:variable_start_index] + '(' + \
												text_list[first_node_index][variable_start_index:]
						text_list[first_node_index] = text_list[first_node_index].strip(')')

					# Collect children and remove redundant elements from text and depth lists
					for temp_index in range(index+1, index+self.get_size_linear_subtree(index)+1):
						collected_children.append([self.amr[temp_index]['text'],self.amr[temp_index]['depth']])
						text_list.pop(index+1)
						depth_list.pop(index+1)
					# Update index to insert at if location of fist node is below this one

					# Prepare the last node of collected children
					if index < first_node_index:
						index_to_insert_at -= (len(self.amr) - len(text_list))

					collected_children[-1][0] = collected_children[-1][0]#.strip(')')

				# change var name in every case
				text_list[index] = self.replace_variable_in_one_text_line(index,new_name)

		# Add closing brackets in the text to be merged
		# Handling the cases where one is child of other
		if first_node_index in range(second_node_index,second_node_index\
										+self.get_size_linear_subtree(second_node_index)+1): return []
		if second_node_index in range(first_node_index,first_node_index\
										+self.get_size_linear_subtree(first_node_index)+1): return []

		children_inserted = 0
 		if collected_children != []:
 			# Step - 3
			# properly merge nodes by removing the common nodes
			children_edges_first_node = self.get_edges_children(first_node_index)
			children_edges_second_node = self.get_edges_children(second_node_index)

			initial_depth = collected_children[0][1]

			collected_children_vars = []
			edges_to_merge = [':mod',':time',':location'] + [':op'+str(i) for i in range(20)] + [':ARG'+str(i) for i in range(20)]
			while True:
				finished = True
				for index_child,child in enumerate(collected_children):
					if child[0][5:8] == '-of' and child[0].startswith(':ARG'):
						children_inserted += 1
						continue
					var, _, _ = self.get_var_info_in_one_text_line(child[0])
					collected_children_vars.append(var)
					if child[1] == initial_depth:
						for edge in children_edges_first_node:
							if child[0].startswith(edge) and edge in edges_to_merge: children_inserted += 1
							# Remove full subtree hanging from that node(child) if its not in edges to merge
							if child[0].startswith(edge) and edge not in edges_to_merge:
								if len(collected_children[index_child+1:]) == 0: temp_index = -1
								for temp_index, temp_child in enumerate(collected_children[index_child+1:]):
									if temp_child[1] == initial_depth: break
								if temp_index == -1:	temp_index += 1
								if temp_index == len(collected_children[index_child+1:])-1:	temp_index += 1
								next_index_initial_depth = temp_index+1
								if len(collected_children) == 1: collected_children = []
								collected_children[index_child:index_child+next_index_initial_depth] = []
								finished = False
								break
					if not finished: break
					if len(collected_children)==0: 
						finished = True
						break
				if finished:	break

			# Step - 4
			# Not merging if any node becomes an ancestor of itself
			if new_name in collected_children_vars:	return []
			new_name_ancestor_list = self.directed_graph.get_ancestor_list(new_name)
			previous_name_ancestor_list = self.directed_graph.get_ancestor_list(previous_name)

			new_name_var_list_linear_subtree = self.get_size_linear_subtree(self.var_to_index[new_name][0],
																				return_vars=True)[1]
			previous_name_var_list_linear_subtree = self.get_size_linear_subtree(\
														self.var_to_index[previous_name][0],return_vars=True)[1]

			if len(list(set(new_name_ancestor_list).intersection(previous_name_var_list_linear_subtree))) != 0:
				return []
			if len(list(set(previous_name_ancestor_list).intersection(new_name_var_list_linear_subtree))) != 0:
				return []

		# Step - 5
 		if collected_children != []:
			collected_children[-1][0] = collected_children[-1][0].strip(')')
			num_closing_brackets_to_add = collected_children[-1][1] - (second_node_depth- first_node_depth)
			if (index_to_insert_at+1)<len(self.amr):
				num_closing_brackets_to_add -= next_depth
			if '(' in collected_children[-1][0]: num_closing_brackets_to_add += 1
			collected_children[-1][0] += ')'*(num_closing_brackets_to_add)

		# Add the new info in the text and depth list
		if (index_to_insert_at+1)<len(self.amr):
			text_list[index_to_insert_at+1:index_to_insert_at+1] = [x[0] for x in collected_children ]
			depth_list[index_to_insert_at+1:index_to_insert_at+1] = [x[1]-(second_node_depth- first_node_depth)\
											 for x in collected_children]
		else:
			text_list += [x[0] for x in collected_children ]
			depth_list += [x[1] - (second_node_depth- first_node_depth) for x in collected_children ]

		final_text_list = []
		for index,text in enumerate(text_list):
			final_text_list.append(' '*6*depth_list[index] + text_list[index])
			if depth_list[index] == 0 and index>0:	print 'error'

		# Step-6
		# Update alignments of the subtree of second variable definition
		path_first_node = self.node_index_to_alignment(first_node_index)
		path_second_node = self.node_index_to_alignment(second_node_index)

		# Update alignments for the children that have been shifted in the first node
		for key in self.alignments.keys():
			path = path_first_node
			for index_alignment, alignment in enumerate(self.alignments[key]):
				if alignment[:len(path)] == path and len(path) < len(alignment) and alignment[len(path)]!='r':
					self.alignments[key][index_alignment][len(path)] = str(children_inserted+
							int(self.alignments[key][index_alignment][len(path)]))

		for key in self.alignments.keys():
			path = path_second_node
			for index_alignment, alignment in enumerate(self.alignments[key]):
				if alignment[:len(path)] == path and len(path) <= len(alignment):
					current_index = self.alignment_to_node_index(alignment)
					current_var = self.amr[current_index]['variable']
					parent_index = self.amr[current_index]['parent_index']
					parent_var = self.amr[parent_index]['variable']
					# special case when node has been dropped
					if self.edges[parent_var+current_var].startswith(':name'):
						self.alignments[key][index_alignment] = path_first_node
						continue
					self.alignments[key][index_alignment][0:len(path)] = path_first_node
		return final_text_list

	def reconstruct_amr(self,text_list=[]):
		# Reconstruct the AMR after merging two nodes
		del self.text_list
		del self.amr
		del self.var_to_index
		del self.nodes
		del self.edges
		del self.directed_graph
		del self.topological_order
		del self.text_index_to_var
		del	self.depth_dict

		# self.text is a list of sentences in case of a document AMR
		self.text_list = text_list
		self.amr = self.text_list
		# mapping from 'variables' to indices in self.amr
		self.var_to_index = {}
		# add attributes
		self.add_attributes()
		# add other attributes like 'variable_start_index'
		self.add_variable_info()
		# contains the edge lable for every class
		self.edges = {}
		self.connections = self.get_edge_info()
		self.get_var_to_index_mapping()
		# Contains all the 'variables' in the list
		self.nodes = self.get_node_info()

		# remove dropped vars from var_to_sent
		var_to_sent = {}
		# print self.nodes
		for key in self.var_to_sent:
			if key in self.nodes :	var_to_sent[key] = self.var_to_sent[key]

		del self.var_to_sent
		self.var_to_sent = var_to_sent

		self.directed_graph = Graph(connections=self.connections,nodes=self.nodes,
									edge_lables=self.edges,var_to_sent=self.var_to_sent)

		self.topological_order = self.directed_graph.topological_order
		# update alignments
		alignments = []
		for key in self.alignments.keys():
			for alignment in self.alignments[key]:
				alignments.append(key+'-'+'.'.join(alignment))

		self.alignments = None
		self.get_alignments(alignments)
		self.get_text_index_to_var()
		self.get_depth_dict()

	def merge_multiple_same_var_children(self,):
		def dfs_amr(index=0,text_list=[]):
			node = self.amr[index]
			text_list.append(node['depth']*6*' '+node['text'])
			var_list = []

			for child_number, child_index in enumerate(node['children_list']):
				dfs_amr(index=child_index,text_list=text_list)
				node_removed = None
				# get a list of vars that have been seen so far
				current_var = self.amr[child_index]['variable']
				if current_var in var_list:
					# If this node is the definition remove the previous instances of that variable 
					# No bracket balancing needed
					if len(self.amr[child_index]['common_text']) > 0:
						for temp_child_number, temp_index in enumerate(node['children_list']):
							#Traverse children of current node till this children
							if temp_index == child_index: break
							if current_var == self.amr[temp_index]['variable']:
								text_list[temp_index] = ''
								node_removed = temp_index
					# Else if it is not definition, remove this node, add current brackets to the previous line
					else:
						# Find a non empty line in the 'text_list' to add ')'
						temp_index = -2
						while text_list[temp_index] == '':	temp_index -= 1
						text_list[temp_index] = text_list[temp_index] + text_list[-1].count(')')*')'
						text_list[-1] = ''
						node_removed = child_index

				# Fixing alignments if node removed
				if node_removed:
					new_location = self.var_to_index[current_var][0]
					path_removed_node = self.node_index_to_alignment(node_removed)
					path_new_location = self.node_index_to_alignment(new_location)
					for key in self.alignments.keys():
						for index_alignment, alignment in enumerate(self.alignments[key]):
							if alignment == path_removed_node:
								self.alignments[key][index_alignment] = path_new_location

				var_list.append(current_var)

		text_list = []
		dfs_amr(index=0,text_list=text_list)
		# pop all empty lines from the text_list
		new_text_list = [line for line in text_list if line != '']
		self.reconstruct_amr(text_list=new_text_list)

	def post_merging_sanity_tests(self,):
		# Check if any node is children of itself
		# Think of other possible tests
		# No repreated edges, etc.
		# No empty lines, every line should have a variable

		# Check if brackets are balanced every where
		num_opening_brackets = 0
		num_closing_brackets = 0
		for index,line in enumerate(self.text_list):
			num_opening_brackets += line.count('(')
			num_closing_brackets += line.count(')')
			if num_closing_brackets > num_opening_brackets:
				print "Merging Failed terminating ..."
				sys.exit()
			if num_opening_brackets == num_closing_brackets:
				if index != len(self.text_list)-1:
					print "Merging Failed terminating ..."
					sys.exit()

		if num_opening_brackets != num_closing_brackets:
			print "Merging Failed terminating ..."
			sys.exit()

		return

	# Merging - Helper functions
	def get_op_list(self,index=-1):
		# Returns if the node has any children with edge ':name'
		# Example -	Input - :name (var2 / name :op1 "ABS-CBN" :op2 "News")))
		# 			Output - ['ABS-CBN', 'News']
		text = ''
		current_var = self.amr[index]['variable']
		for child_index in self.amr[index]['children_list']:
			child_var = self.amr[child_index]['variable']
			if self.edges[current_var+child_var].startswith(':name'):
				text = self.amr[child_index]['text']
		if text == '':	return []
		text = text.strip(')')
		text = text.split('/')[1]
		text = text.split()
		op_list = []
		for index_word, word in enumerate(text):
			if word.startswith(':op'): op_list.append(text[index_word+1].lower())
		return op_list

	def get_edges_children(self,node_index):
		children_edges = []
		for child_index in self.amr[node_index]['children_list']:
			edge = self.edges[self.amr[node_index]['variable']+self.amr[child_index]['variable']]
			children_edges.append(edge)
		return children_edges

	def check_initials(self,first_list=[],second_list=[]):
		if not (len(first_list) == 1 or len(second_list) == 1): return False
		first_list = [x.strip('"') for x in first_list]
		second_list = [x.strip('"') for x in second_list]
		if len(first_list) == 1:
			if first_list[0] == ''.join([x[0] for x in second_list]):	return True
		if len(second_list) == 1:
			if second_list[0] == ''.join([x[0] for x in second_list]):	return True

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
		if first_sub_list or second_sub_list:	return True
		else: return False

	def replace_variable_in_one_text_line(self,node_index,new_name=''):
		# Removes the existing variable and add new variable, doesn't change in the eixising AMR subtree

		# Update variable name in the text
		text = self.amr[node_index]['text']
		previous_name = self.amr[node_index]['variable']
		variable_start_index = self.amr[node_index]['variable_start_index']
		variable_end_index = self.amr[node_index]['variable_end_index'] 
		variable_end_index += len(new_name)-len(previous_name)
		text = text[ : variable_start_index] + new_name + text[variable_end_index+1 : ]

		# Prepare text
		if '(' in text: text = text[ : variable_start_index-1] + new_name
		else:	text = text[ : variable_start_index] + new_name

		# Add closing brackets, assumig children will be removed
		num_closing_brackets_to_add = self.amr[node_index]['depth']
		if (node_index+self.get_size_linear_subtree(node_index)+1)<len(self.amr):
			num_closing_brackets_to_add-=self.amr[node_index+self.get_size_linear_subtree(node_index)+1]['depth']
		text = text.strip(')')+')'*num_closing_brackets_to_add

		# Get other info corresponding to the variable
		variable, variable_start_index, variable_end_index = self.get_var_info_in_one_text_line(text)
		return text

	# Translation functions - provides traslations between - (word,alignment); (alignment, node_index); 
	# 															(node_index, alignment); (node_index, sent_index)
	def word_to_alignment(self,word='',sentence='',location_of_word=0):
		# assuming - tokenization of words in gold-standard and coreference resolver is same
		if str(location_of_word) in self.alignments.keys():
			# for now, not choosing the node, if there are multiple possible alignments
			if len(self.alignments[str(location_of_word)]) == 1:
				return self.alignments[str(location_of_word)][0]
			if len(self.alignments[str(location_of_word)]) >= 1:
				non_edge_alignments = []
				for alignment in self.alignments[str(location_of_word)]:
					if alignment[-1] != ['r']:	non_edge_alignments.append(alignment)
				if len(non_edge_alignments) == 0:	return None
				min_index = 0
				for temp_index, alignment in enumerate(non_edge_alignments):
					if len(alignment) < len(non_edge_alignments[min_index]): min_index = temp_index
				return non_edge_alignments[min_index]
		else: return None

	def alignment_to_node_index(self,alignment):
		global total_count
		global interest 
		total_count += 1
		# first element in alignment list is always useless - need it just because it is the original AMRs
		index = 0
		# how to handle the fucking internal nodes
		# for now -
		# merging only the nodes, that are in the written AMR representaion, not the once that are in graphical
		for index_in_alignment, branch_to_take in enumerate(alignment[1:]):
			branch_to_take = int(branch_to_take) - 1
			if index != 0:
				branch_to_take = branch_to_take - (self.amr[index]['text'].count(':')-1)
			else:
				# because text at first point doesn't start with a ':'
				branch_to_take = branch_to_take - self.amr[index]['text'].count(':')
			if branch_to_take < 0:
				break
				print branch_to_take, self.amr[index]['text']
			if int(branch_to_take) >= len(self.amr[index]['children_list']):
				# just a hack, works for now, seems better now, used only around 10 times
				# Example case - merging a node like - :name(/ :op1 dlfkasj :op2 dfklsaj , op2 can be aligned 
				# if index_in_alignment < len(alignment) -2: 
				interest += 1
				break
				# print alignment, self.amr[index]['children_list'] 
				# print 'gonna get error'
				# self.print_amr()
			index = self.amr[index]['children_list'][int(branch_to_take)]
		return index

	def node_index_to_alignment(self,node_index):
		# Given the node_index return the alignment
		path = []
		new_parent_index = node_index
		while new_parent_index != 0:
			try:	path[0] = str(int(path[0]) + self.amr[new_parent_index]['text'].count(':')-1)
			except:	pass
			path.insert(0,str(self.amr[new_parent_index]['child_number']+1))
			new_parent_index = self.amr[new_parent_index]['parent_index']
		return ['1'] + path

	def node_index_to_sent_index(self,index_node):
		# returns the sentence index given the node_index
		for index_sent, sent_range in enumerate(self.sentence_boundries):
			if index_node in range(sent_range[0],sent_range[1]+1):	return index_sent
		return -1

	def amr_to_text_based_on_alignments(self,var_list=[]):
		text = ''
		selected_keys_list = []
		for key in self.alignments.keys():
			for alignment in self.alignments[key]:
				index = self.alignment_to_node_index(alignment)
				var = self.amr[index]['variable']
				if var in var_list:
					selected_keys_list.append(int(key))
		word_list = []
		for key in set(selected_keys_list):
			word_list.append(self.split_text[key])
			text = text + self.split_text[key] + ' '
		# print text
		# return text
		return ' '.join(list(set(word_list)) )

	# Convert AMR-Graph -> AMR-text
	def get_AMR_from_directed_graph(self,topological_order_sub_graph={},sub_graph={}):
		# Function to convert graph to text-AMR
		# 'topological_ordering' contains the list of unused variables

		# get list of variables from the directed graph
		list_of_variables,depth_list = self.get_list_from_directed_graph(sub_graph)

		text_list_sub_graph = self.get_text_subgraph(list_of_variables,depth_list)

		return AMR(text_list_sub_graph,amr_with_attributes=False)

	def get_list_from_directed_graph(self,sub_graph={}):

		def dfs(root,sub_graph,depth,depth_list=[],ordered_list=[],consturcted_list=[]):
			already_visited = False
			if root in set(ordered_list+consturcted_list):	already_visited = True
			# Preserving the order of children
			if already_visited :	return ordered_list
			ordered_list.append(root)
			depth_list.append(depth)
			# order children in ':name', ':ARGx', 'op', ':mod', ':time', others ,'ARGx-of'
			children_list =  sub_graph._graph[root]
			children_list = self.get_children_order(node=root,child_list=list(children_list))
			# print children_list
			for child in children_list:
				ordered_list = dfs(child,sub_graph,depth+1,depth_list,ordered_list,consturcted_list)
			return ordered_list

		ordered_list = []
		depth_list = []
		# find a root node
		new_root = None
		for node in self.get_depth_order(sub_graph._graph.keys()):
			if len(sub_graph.reverse_graph[node]) == 0:
				new_root = node
				break
		if new_root == None:	return sub_graph._graph.keys(),[0]*len(sub_graph._graph.keys())

		# traverse and include the nodes in the new 
		depth = 0
		ordered_list = dfs(new_root,sub_graph,depth,depth_list,ordered_list=[],consturcted_list=ordered_list)

		while len(set(ordered_list)) != len(sub_graph._graph.keys()):
			# find a node connected to the graph consturcted so far
			new_node_found = False
			new_root = None
			# todo - add an order for node selection
			for node in self.get_depth_order(set(sub_graph._graph.keys())-set(ordered_list)):
				for child_node in sub_graph._graph[node]:
					if child_node in ordered_list:
						new_root = node
						index_to_append_at = ordered_list.index(child_node)
						depth = depth_list[index_to_append_at]

						temp_depth_list = list(depth_list)
						temp_ordered_list = list(ordered_list)
						# print index_to_append_at, depth_list, depth_list[index_to_append_at]

						try:
							index_to_append_at +=next(x[0] for x in enumerate(temp_depth_list[index_to_append_at+1:])\
															if x[1] <= temp_depth_list[index_to_append_at])
						except:
							index_to_append_at = len(ordered_list) -1


						new_node_found = True

						# print index_to_append_at, depth_list, depth
						# print ordered_list

						break
				if new_node_found:	break
			new_depth_list = []
			temp_list = dfs(new_root,sub_graph,depth+1,new_depth_list,
								ordered_list=[],consturcted_list=ordered_list)

			# add temp_list in the ordered_list, update the depth list
			ordered_list[index_to_append_at+1 : index_to_append_at+1] = temp_list
			depth_list[index_to_append_at+1 : index_to_append_at+1] = new_depth_list

		# 0/0
		return ordered_list,depth_list

	def get_text_subgraph(self,list_of_variables,depth_list):
		# add attributes and text corresponding to the variable list
		# todo - handle variable repetition
		# adding attributes just to take ease the process of text list formation
		amr_node_list = []
		text_list = []
		previous_higher_depth_index = 0
		num_closing_brackets_to_add = 0
		for index_variable,variable in enumerate(list_of_variables):
			new_node_dict = {}
			new_node_dict['depth'] = depth_list[index_variable]
			new_node_dict['variable'] = variable
			new_node_dict['variable_start_index']=self.amr[self.var_to_index[variable][0]]['variable_start_index']
			new_node_dict['variable_end_index'] = self.amr[self.var_to_index[variable][0]]['variable_end_index']
			new_node_dict['common_text'] = self.amr[self.var_to_index[variable][0]]['common_text']

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
				new_node_dict['child_number'] = 0
				# if root text need to be changed as well
				new_node_dict['text'] = '('+ new_node_dict['variable'] + ' ' + new_node_dict['common_text']\
										+ ')'*num_closing_brackets_to_add
			else:
				parent_index_new_amr=(len(temp_depth_list)-1) - temp_depth_list.index(new_node_dict['depth']-1)
				amr_node_list[parent_index_new_amr]['children_list'].append(index_variable)
				new_node_dict['child_number'] = len(amr_node_list[parent_index_new_amr]['children_list'])
				amr_node_list[parent_index_new_amr]['no_of_children'] += 1

				# text = edge + '(' + variable + ' ' + common_text + ')'*num_closing_brackets
				try:
					edge = self.edges[amr_node_list[parent_index_new_amr]['variable']+new_node_dict['variable']]
				except KeyError:
					edge = self.edges[new_node_dict['variable']+amr_node_list[parent_index_new_amr]['variable']]
					if '-of' not in edge:	edge = edge + '-of'
					else:	edge = edge[:-3]
						# print edge, new_node_dict['variable'],amr_node_list[parent_index_new_amr]['variable']
						# return 'some_error'

				new_node_dict['text'] = edge + ' (' + new_node_dict['variable'] + ' ' \
									+ new_node_dict['common_text'] + ')'*num_closing_brackets_to_add

			new_node_dict['no_of_children'] = 0
			new_node_dict['children_list'] = []
			amr_node_list.append(new_node_dict)
			text_list.append(' '*6*new_node_dict['depth']+new_node_dict['text'])
		return text_list

	def get_depth_order(self,nodes=[]):
		relevant_tuples = []
		for key in self.depth_dict:
			if key in nodes:
				relevant_tuples.append((key, self.depth_dict[key]))

		relevant_tuples = sorted(relevant_tuples, key=lambda x: x[1])
		# just return the list of vars s.t. first var has the least depth 
		return [x[0] for x in relevant_tuples]

	def get_children_order(self,node='',child_list=[]):
		ordered_children_list = []
		node_index = self.var_to_index[node][0]
		parent_var = node
		# order children in ':name', ':ARGx', 'op', ':mod', ':time', others ,'ARGx-of'
		order_children = [':name'] + [':ARG'+str(x) for x in range(20)] + [':op'+str(x) for x in range(20)] + \
							[':mod',':time'] + []
		relevant_var_edge_dict = {}
		relevant_var_order_dict = {}
		# print node, child_list
		for child_index in self.amr[node_index]['children_list']:
			var = self.amr[child_index]['variable']
			# print var
			if var in child_list:
				try:
					current_edge = self.edges[parent_var+var]
				except:
					current_edge = self.edges[var+parent_var]
				relevant_var_edge_dict[var] = current_edge

				if current_edge in order_children:	relevant_var_order_dict[var] = order_children.index(current_edge)
				elif '-' in current_edge:	relevant_var_order_dict[var] = 1000
				else:	relevant_var_order_dict[var] = 100

		# print node, relevant_var_edge_dict
		# print node, relevant_var_order_dict
		sorted_relevant_var_order_dict = sorted(relevant_var_order_dict.items(), key=operator.itemgetter(1))
		# print node, sorted_relevant_var_order_dict
		ordered_children_list = [x[0] for x in sorted_relevant_var_order_dict]
		# print node, ordered_children_list
		if set(child_list) != set(ordered_children_list):
			ordered_children_list.extend(list(set([x for x in child_list if x not in ordered_children_list])))
			# print 'something horrible happened'
		# 	print child_list, ordered_children_list, self.amr[node_index]['text'], self.amr[node_index]['children_list']
		return ordered_children_list

	# Helper functions
	def print_amr(self,file='',print_indices=True,write_in_file=False,one_line_output=False,return_str=False):		
		printed = ''
		if write_in_file:
			for index_node,node in enumerate(self.amr):
				if print_indices: 
					if return_str:	printed += index_node
					else:			file.write(index_node)
				if one_line_output:
					if return_str: 	printed += node['text']+' '
					else:			file.write(node['text']+' ')
				else:
					if return_str:	file.write(node['text']+' '+ '\n')
					else:			printed += node['text']+' '+ '\n'
		else:
			for index_node,node in enumerate(self.amr):
				if print_indices: 
					print index_node,

				if one_line_output:
					print node['depth']*'	' + node['text'],
				else:
					print node['depth']*'	' + node['text']
		return printed

	def get_topological_order_sub_graph(self,nodes):
		# returns the topological order in the sub graph
		return self.directed_graph.get_topological_order_sub_graph()

	def get_size_linear_subtree(self,node_index,return_vars=False):
		initial_index = node_index
		initial_depth = self.amr[node_index]['depth']
		var_list_linear_subtree = []
		while node_index < len(self.amr):
			if self.amr[node_index]['depth'] <= initial_depth and node_index!=initial_index:
				break
			var_list_linear_subtree.append(self.get_var_info_in_one_text_line(self.amr[node_index]['text'])[0])
			node_index += 1
		if return_vars:
			return (node_index-1)-initial_index, var_list_linear_subtree

		return (node_index-1)-initial_index

	def break_path_by_sentences(self,path):
		# path - a list of connected vars
		# return - a dict (sent -> var sets)
		current_sent = 0
		var_sent_dict = {}
		possible_current_sents = []
		current_var_set = []
		for var in path:
			current_var_sents = self.var_to_sent[var]
			if possible_current_sents != []:
				# if current_var can be in one of the possible current_sents - add it
				if len(list(set(current_var_sents).intersection(possible_current_sents))) != 0:
					possible_current_sents = list(set(current_var_sents).intersection(possible_current_sents))
					current_var_set.append(var)
				# else, add current var set and start with new possibility of sentences
				else:
					# to-copy
					var_sent_dict[possible_current_sents[0]] = list(current_var_set)
					del current_var_set
					possible_current_sents = current_var_sents
					current_var_set = [var]
			else:
				possible_current_sents = current_var_sents
				current_var_set = [var]

		var_sent_dict[possible_current_sents[0]] = list(current_var_set)
		del current_var_set

		# second iteration to find sentences for vars occuring in multiple sents
		possible_current_sents = var_sent_dict.keys()
		for var in path:
			current_var_sents = self.var_to_sent[var]
			for sent_index in set(current_var_sents).intersection(possible_current_sents):
				temp_var_list = list(set(var_sent_dict[sent_index] + [var]))
				var_sent_dict[sent_index] = list(temp_var_list)

		return var_sent_dict

	def get_concept_relation_list(self,story_index=0,debug=False):
		# get concept relation list
		try:	del self.concept_relation_list
		except:	pass
		self.concept_relation_list = concept_relation_list(index_to_var=self.text_index_to_var,
															story_index=story_index,
															var_list=list(self.var_to_index.keys()),
															aligned_vars=self.aligned_vars,
															graph=self.directed_graph,
															text=self.text)
		if debug:	self.concept_relation_list.print_tuples()

	def get_sent_amr(self,sent_index=0):
		var_list = []
		for key in self.var_to_sent:
			if sent_index in self.var_to_sent[key]:
				var_list.append(key)
		return list(set(var_list))

	# AMR-class construction helper functions
	def get_depth_dict(self,):
		self.depth_dict = {}
		for node in self.amr:
			var = node['variable']
			try: self.depth_dict[var] = min(node['depth'],self.depth_dict[var])
			except: self.depth_dict[var] = node['depth']

	def get_edge_info(self,):
		# gives the edge lable and all the connections
		connections = []
		for index_node, node in enumerate(self.amr):
			if 'children_list' not in node.keys():
				# generally arise because of issues with depth
				self.print_amr()
				print node, index_node
			for child in node['children_list']:
				self.edges[node['variable']+self.amr[child]['variable']] \
					= self.amr[child]['text'][0:self.amr[child]['text'].index(' ')]
				# Examples for '-' cases are '-of', '-to' 
				if '-' in self.edges[node['variable']+self.amr[child]['variable']]:
					connections.append([self.amr[child]['variable'], node['variable']])
				else:
					connections.append([node['variable'], self.amr[child]['variable']])
				# remove the imaginary edges from the graphical structure
				if node['variable'] == 'Rinfinite': connections.pop()
		return connections

	def get_node_info(self,):
		# gives the list of all the 'variables' in the AMR
		nodes = []
		for node in self.amr:
			nodes.append(node['variable'])
			if node['variable'] == 'Rinfinite': nodes.pop()
		return nodes

	def get_alignments(self,alignments=[]):
		# self.alignments is a list of alignment
		# alignment is a list of branch to take at each step in AMR
		new_format_alignment = {}
		for alignment in alignments:
			if alignment.split('-')[0] in new_format_alignment.keys():
				new_format_alignment[alignment.split('-')[0]].append(alignment.split('-')[1].split('.'))
			else:
				new_format_alignment[alignment.split('-')[0]] = [alignment.split('-')[1].split('.')]
		self.alignments = new_format_alignment
		# print 'num_alignments -', len(self.alignments)

	def get_text_index_to_var(self,):
		# creates the text-index to var map
		self.aligned_vars = []
		self.text_index_to_var = {}
		for key in self.alignments:
			temp_var_set = []
			for alignment in self.alignments[key]:
				# alignment = self.alignments[key][0]
				# hacks - todo: fix, it's wrong
				# if alignment[-1] == 'r':	alignment[-1] = '0'
				if alignment[-1] == 'r':	alignment.pop()
				if alignment[-1] == '':		alignment.pop()

				index = self.alignment_to_node_index(alignment)
				temp_var_set.append(self.amr[index]['variable'])
			self.aligned_vars.extend(temp_var_set)
			self.text_index_to_var[key] = temp_var_set

		# print 'num_aligned_vars -', len(set(self.aligned_vars))
		# print 'num_vars -', len(self.var_to_index)

	def get_var_to_index_mapping(self,):
		# at one of the indeices mapped with the variable,
		# we will have the text information accociated with the variable
		for index, node in enumerate(self.amr):
			if node['variable'] not in self.var_to_index.keys():	self.var_to_index[node['variable']] = []
			# Not fully sure if this 'common_text' based method works for finding out if this is where the
			# variable is defined
			if len(self.amr[index]['common_text']) > 0:	self.var_to_index[node['variable']].insert(0,index)
			else: self.var_to_index[node['variable']].append(index)

	def get_var_info_in_one_text_line(self,text):
		# return variable,variable_start_index,variable_end_index, for any piece of text in AMR format
		if '(' not in text:
			# for cases where 'text' is of the form ':ARG0 o'
			variable = text[text.strip().rfind(' ')+1 :	].strip(')')
			variable_start_index = text.strip().rfind(' ')+1
			variable_end_index = variable_start_index + len(variable)-1
		else:
			variable_start_index = text.index('(')
			if ' ' not in text[variable_start_index:]:
				self.print_amr()
				print text
			variable = text[variable_start_index + 1 : variable_start_index +\
						text[variable_start_index:].index(' ')]

			variable_start_index = variable_start_index + 1
			variable_end_index = variable_start_index + len(variable)-1

		if '~' in variable:
			variable = variable[ : variable.index('~')]
		return variable, variable_start_index, variable_end_index

	def get_sentence_boundaries_amr(self,):
		self.sentence_boundries = []
		previous_depth_1_index = 0
		for index_node, node in enumerate(self.amr):
			if node['depth'] == 1:
				if index_node > 1:
					self.sentence_boundries.append([previous_depth_1_index,index_node-1])
				previous_depth_1_index = index_node
		self.sentence_boundries.append([previous_depth_1_index,index_node])

	def add_variable_info(self,):
		# adding variable, it's index and all other sutff
		for index,node in enumerate(self.amr):
			variable,variable_start_index,variable_end_index =  self.get_var_info_in_one_text_line(node['text'])
			node['variable'] = variable
			node['variable_start_index'] = variable_start_index
			node['variable_end_index'] = variable_end_index
			node['common_text'] = node['text'][variable_end_index+1:].strip().strip(')')

	def add_attributes(self,):
		# Takes the AMR as input in the form of 'text'. 'text' is simply a list of lines from the file
		# Returns the AMR in the form of dictionary, with some added attributes like,'parent_index','depth' etc.
		# 'depth_amr' the list of nodes
		amr = self.text_list
		depth_amr = []
		for line in amr:
			# Calculate depth, as (leading_spaces % 6)
			if type(line) == type('string'):
				depth = (len(line) - len(line.lstrip(' '))) / 6
				line = line.strip()	
				depth_amr.append({'text':line,'depth':depth})
		amr = depth_amr
		depth_amr = []
		# add no_of_children field
		amr[0]['parent_index'] = -1
		# for line in self.text_list:
		# 	print line,
		amr[0]['children_list'] = []
		for index, line in enumerate(amr):
			no_of_children = 0
			depth = line['depth']
			temp_depth = depth+1
			temp_index = index
			while temp_depth > depth:
				temp_index = temp_index + 1
				if temp_index >= len(amr):
					break
				temp_depth = amr[temp_index]['depth']
				if temp_depth == depth + 1:
					no_of_children = no_of_children + 1
					# append in parents children list
					amr[index]['children_list'].append(temp_index)
					# adding parent_index and empty children list
					amr[temp_index]['parent_index'] =  index
					amr[temp_index]['children_list'] = []
			amr[index]['no_of_children'] = no_of_children
		# add_child_number field
		def add_child_number(amr,line_no):
			child_number = 0
			for index, line in enumerate(amr[line_no+1:]):
				if line['depth'] <= amr[line_no]['depth']:
					break
				if line['depth'] == amr[line_no]['depth'] + 1:
					amr[line_no+index+1]['child_number'] = child_number
					child_number = child_number + 1
					add_child_number(amr,line_no+index+1)
		amr[0]['child_number'] = 0
		add_child_number(amr,0)
		self.amr = amr
