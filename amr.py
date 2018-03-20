from directed_graph import Graph
#from concept_relation_list import concept_relation_list
import copy
import sys
import operator

# It's the AMR class - saves the text representation of a single AMR

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
		self.common_text = self.get_common_text_var_mapping()

		# get 'var_to_sent'
		if var_to_sent == {}:
			for key in self.var_to_index.keys():	var_to_sent[key] = [sent_index]
		self.var_to_sent = var_to_sent

		self.alignments = None
		self.get_alignments(alignments)
		# Not updated while mering any 2 nodes
		self.get_sentence_boundaries_amr()

		self.get_text_index_to_var()

		self.directed_graph = Graph(connections=self.connections,nodes=self.nodes,
										edge_lables=self.edges,var_to_sent=self.var_to_sent,
										common_text=self.common_text,
										text_index_to_var=self.text_index_to_var,
										root=self.amr[0]['variable'])

		self.topological_order = self.directed_graph.topological_order
		# self.text is a list of sentences in case of a document AMR
		self.text = text
		self.split_text = (' '.join(self.text)).split()

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

	def merge_named_entities_graph(self,):
		# Desined specifically to run initially, may not work if run after some other mergers
		# name list 
		existing_names = []
		node_merged = False
		for var in self.directed_graph._graph.keys():
			parent_var = ''
			for node in self.directed_graph.reverse_graph[var]:
				if self.directed_graph.depth_dict[node]+1 == self.directed_graph.depth_dict[var]:
					self.directed_graph.edge_lables[node+' '+var]
					parent_var = node 
					break

			if parent_var!= '' and ':name' in self.directed_graph.edge_lables[parent_var+' '+var]:
				node_merged = False
				for existing_var in existing_names:
					can_merge = False

					for node in self.directed_graph.reverse_graph[existing_var]:
						if self.directed_graph.depth_dict[node]+1 == self.directed_graph.depth_dict[existing_var]:
							parent_existing_var = node

					op_list_second_node = self.directed_graph.get_op_list(var=parent_existing_var)
					op_list_first_node = self.directed_graph.get_op_list(var=parent_var)
		
					if not self.check_mutual_sublist(first_list=op_list_first_node,second_list=op_list_second_node):
						# don't merge if one isn't a sublist of other except when one is in the form of initials
						if self.check_initials(first_list=op_list_first_node,second_list=op_list_second_node):
							can_merge = True
						else:	can_merge = False
					else:	can_merge = True

					if self.directed_graph.common_text[existing_var].strip() == \
								 self.directed_graph.common_text[var].strip():
						can_merge = True

					if can_merge:
						if self.directed_graph.common_text[parent_existing_var] == \
							self.directed_graph.common_text[parent_var]:
							# If successfull merger, restart merging
							successfull_merge = self.merge_nodes(first_var=existing_var,second_var=var,debug=False)
	
							if successfull_merge == 2:
								# self.reconstruct_amr()
								# print successfull_merge
								return 1
				if not node_merged:
					existing_names.append(var)
		return 0

	def merge_date_entites(self,):
		existing_dates = []
		for index_node,node in enumerate(self.amr):
			node_merged = False
			if 'date-entity ' in node['text']:
				for index_existing_node in existing_dates:
					if self.amr[index_existing_node]['common_text'].strip() == node['common_text'].strip():
						self.merge_nodes(first_node_index=index_existing_node,second_node_index=index_node)
						self.reconstruct_amr()
						return 1
				if not node_merged:
					existing_dates.append(index_node)
		return 0

	def merge_nodes(self,first_alignment=[],second_alignment=[],
		first_node_index=None,second_node_index=None,debug=False,
		first_var='',second_var=''):
		# steps in the procedure - 
		# 1. sanity checks
		# 2. Merging subtrees
		# 3. Reconstruct AMR
		# move subtree of the second node to first node
		# Return values - 
		# 0 - Didn't merge
		# 1 - No merger needed
		# 2 - Successfull merge

		if first_var == '':	first_var = self.amr[first_node_index]['variable']
		if second_var == '': second_var = self.amr[second_node_index]['variable']

		returned_value = self.move_subtree_via_directed_graph(first_var=first_var,second_var=second_var)

		if debug:	print returned_value
		if returned_value != -1:	return returned_value

		return 2

	def move_subtree_via_directed_graph(self,first_var='',second_var=''):
		# get AMR 'text_list' by merging and generation using the directed graph

		returned_value = self.directed_graph.merge_nodes_in_graph(
							first_var=first_var,second_var=second_var)

		return returned_value

	def reconstruct_amr(self):
		text_list=self.directed_graph.generate_text_amr()
		text_list =[line + '\n' for line in text_list]
	
		text_index_to_var = self.directed_graph.text_index_to_var
		var_to_sent = self.directed_graph.var_to_sent

		# Reconstruct the AMR after merging two nodes
		del self.text_list
		del self.amr
		del self.var_to_index
		del self.nodes
		del self.edges
		del self.directed_graph
		del self.topological_order
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

		del self.var_to_sent
		self.var_to_sent = {}
		for var in var_to_sent.keys():
			if var in self.nodes:
				self.var_to_sent[var] = var_to_sent[var]

		self.common_text = self.get_common_text_var_mapping()

		temp = set(self.alignments.keys())
		del self.alignments
		self.alignments = {}
		for text_index in text_index_to_var.keys():
			# alignment in case of KeyError is mostly useless (but not always)
			self.alignments[text_index] = []
			for var in text_index_to_var[text_index]:
				try:	node_index = self.var_to_index[var][0]
				except KeyError:	break
				var_path = self.node_index_to_alignment(node_index)
				self.alignments[text_index].append(var_path)

		alignments = []
		for key in self.alignments.keys():
			for alignment in self.alignments[key]:
				alignments.append(key+'-'+'.'.join(alignment))

		self.alignments = None
		self.get_alignments(alignments)
		self.get_text_index_to_var()

		var_set = []
		for key in self.text_index_to_var.keys():
			var_set.extend(self.text_index_to_var[key])

		var_set = list(set(var_set))
		for var in var_set:
			if var not in self.nodes:
				print 'some bug'
				0/0
	
		self.directed_graph = Graph(connections=self.connections,nodes=self.nodes,
									edge_lables=self.edges,var_to_sent=self.var_to_sent,
									common_text=self.common_text,
									text_index_to_var=self.text_index_to_var,
									root=self.amr[0]['variable'])


		self.topological_order = self.directed_graph.topological_order
		self.get_depth_dict()

	def post_merging_sanity_tests(self,):
		# Check if any node is children of itself
		# No repreated edges, etc.
		# No empty lines, every line should have a variable

		# Check if brackets are balanced every where
		num_opening_brackets = 0
		num_closing_brackets = 0
		for index,line in enumerate(self.text_list):
			num_opening_brackets += line.count('(')
			num_closing_brackets += line.count(')')
			if num_closing_brackets > num_opening_brackets:
				self.print_amr(print_indices=False)
				print "Merging Failed terminating ..."
				sys.exit()
			if num_opening_brackets == num_closing_brackets:
				if index != len(self.text_list)-1:
					self.print_amr(print_indices=False)
					print "Merging Failed terminating ..."
					sys.exit()

		if num_opening_brackets != num_closing_brackets:
			self.print_amr(print_indices=False)
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
		# print 'current_var ', current_var
		for child_index in self.amr[index]['children_list']:
			child_var = self.amr[child_index]['variable']
			if self.edges[current_var+' '+child_var].startswith(':name'):
				text = self.amr[child_index]['text']

		if text == '':	return []

		text = text.strip(')')
		text = text.split('/')[1]
		text = text.split()
		op_list = []
		for index_word, word in enumerate(text):
			if word.startswith(':op'): op_list.append(text[index_word+1].lower())
		op_list = [word for word in op_list if word!='']
		return op_list

	def get_edges_children(self,node_index):
		children_edges = []
		for child_index in self.amr[node_index]['children_list']:
			edge = self.edges[self.amr[node_index]['variable']+' '+self.amr[child_index]['variable']]
			children_edges.append(edge)
		return children_edges

	def check_initials(self,first_list=[],second_list=[],debug=False):
		# return True if and only if one is initials of other
		if not (len(first_list) == 1 or len(second_list) == 1): return False

		first_list = [x.strip('"') for x in first_list]
		second_list = [x.strip('"') for x in second_list]

		if debug:	print first_list,second_list

		if len(first_list) == 1:
			if first_list[0] == ''.join([x[0] for x in second_list]):	return True
		if len(second_list) == 1:
			if second_list[0] == ''.join([x[0] for x in first_list]):	return True
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

	# Translation functions - provides traslations between - 
	# (word,alignment); (alignment, node_index); 
	# (node_index, alignment); (node_index, sent_index)
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
		return ' '.join(list(set(word_list)) )

	# Convert AMR-Graph -> AMR-text
	def get_AMR_from_directed_graph(self,topological_order_sub_graph={},sub_graph={}):
		# Function to convert graph to text-AMR
		# get list of variables from the directed graph

		list_of_variables,depth_list = sub_graph.get_var_list_from_directed_graph()

		text_list_sub_graph = sub_graph.get_text_list(list_of_variables,depth_list)

		return AMR(text_list=text_list_sub_graph,text=self.text,amr_with_attributes=False)#,

	# Helper functions
	def print_amr(self,file='',print_indices=True,write_in_file=False,
		one_line_output=False,return_str=False,to_print=True):
		printed = ''
		if write_in_file:
			for index_node,node in enumerate(self.amr):
				if one_line_output:
					file.write(node['text']+' ')
				else:
					if print_indices: 
						file.write(str(index_node) + ' ')
					file.write(node['depth']*'	' + node['text']+ '\n')
			file.write('\n')
		if to_print:
			# print only if not writing in file
			for index_node,node in enumerate(self.amr):
				if one_line_output:
					print ' ' + node['text'],
				else:
					if print_indices:
						print str(index_node) + ' ',
					print node['depth']*'	' + node['text']

		if return_str:
			for index_node,node in enumerate(self.amr):
				if one_line_output:
					printed += ' ' + node['text']
				else:
					if print_indices:
						printed += str(index_node) + ' '
					printed += node['depth']*'	' + node['text'] + '\n'
		return printed

	def get_nodes(self,):
		node_list = []
		for index_node,node in enumerate(self.amr):
			node_list.append(node['common_text'])
		node_list = [x for x in node_list if x != '']
		node_list = [x if ':' not in x else x[: x.index(':')-1] for x in node_list]
		for node in node_list:
			if not node.startswith('/ '):
				# print node
				0/0
		node_list = [node[1:] for node in node_list]
		return node_list

	def get_edge_tuples(self,):
		edge_tuple_list = []

		for parent_child_pair in self.directed_graph.edge_lables:
			parent, child = parent_child_pair.split(' ')
			lable = self.directed_graph.edge_lables[parent_child_pair][0].strip()

			parent_index = self.var_to_index[parent][0]
			child_index = self.var_to_index[child][0]

			parent_common_text = self.amr[parent_index]['common_text']
			child_common_text = self.amr[child_index]['common_text']

			if ':' in parent_common_text:
				parent_common_text = parent_common_text[: parent_common_text.index(':')-1]
			if ':' in child_common_text:
				child_common_text = child_common_text[: child_common_text.index(':')-1]

			parent_common_text = parent_common_text[1:].strip()
			child_common_text = child_common_text[1:].strip()
	
			edge_tuple_list.append(parent_common_text+'_'+lable+'_'+child_common_text)

		return edge_tuple_list

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
	def get_common_text_var_mapping(self,):
		common_text = {}
		for var in self.nodes:
			index_var = self.var_to_index[var][0]
			common_text[var] = self.amr[index_var]['common_text']
		return common_text

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
				self.edges[node['variable']+' '+self.amr[child]['variable']] \
					= self.amr[child]['text'][0:self.amr[child]['text'].index(' ')]
				# Examples for '-' cases are '-of', '-to' 
				if '-' in self.edges[node['variable']+' '+self.amr[child]['variable']]:
					connections.append([self.amr[child]['variable'], node['variable']])
				else:
					connections.append([node['variable'], self.amr[child]['variable']])
				# remove the imaginary edges from the graphical structure
		return connections

	def get_node_info(self,):
		# gives the list of all the 'variables' in the AMR
		nodes = []
		for node in self.amr:
			nodes.append(node['variable'])
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

	def get_text_index_to_var(self,):
		# creates the text-index to var map
		self.aligned_vars = []
		self.text_index_to_var = {}
		for key in self.alignments:
			temp_var_set = []
			for alignment in self.alignments[key]:
				if alignment[-1] == 'r':	alignment.pop()
				if alignment[-1] == '':		alignment.pop()

				index = self.alignment_to_node_index(alignment)
				temp_var_set.append(self.amr[index]['variable'])
			self.aligned_vars.extend(temp_var_set)
			self.text_index_to_var[key] = temp_var_set

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
