import codecs
import sys
import os
import pickle
import nltk
from nltk.corpus import stopwords
import collections
import operator
import sys
import networkx as nx
import matplotlib.pyplot as plt
from read_data import *
from resolve_coref import *
from directed_graph import Graph
from amr import AMR
#from concept_relation_list import concept_relation_list

# Input - list of sentence AMRs, the form of AMRs so far is the 'AMR' class
# Procedure - 
# 	1. add an extra root node and join roots of each sentences as the children of the main root node
# 	2. makes sure that variable names are not same across different sentences
# Output - merged document amr

# change the code to make changes to make the variable change names in original sentences as well
def merge_sentence_amrs(amr_list,debug=False):
	# 'doc_amr' will contain a list of nodes
	doc_amr = []
	# add root node, this is just for AMR representation using AMRICA
	# this node should never be used in the actual representaion of AMR
	doc_amr.append({'child_number': 0,
					'children_list': [],
					'text': '(Rinfinite / root',
					'common_text': '/ root',
					'depth': 0,
					'parent_index': -1,
					'num_children': 0,
					'variable': 'Rinfinite'
					})
	variable_list = []
	new_var_index = 0
	document_text = []
	document_alignments = []
	var_to_sent = {}
	for sent_index_in_doc, sent_amr in enumerate(amr_list):
		# To Replace the variable names that have been previously used in a different sentence
		current_sent_variable_list = []
		replaced_var_current_sent = []
		# dictionary mapping from old to replaced variables
		replaced_dict_current_sent = {}
		doc_amr[0]['children_list'].append(len(doc_amr))
		cycles = list(nx.simple_cycles(sent_amr.directed_graph.nx_graph))
		# todo - remove only those edges that removes all the cycles from the graph
		if len(cycles) != 0:
			print sent_amr.directed_graph._graph
			for cycle in cycles:
				print cycle
				for node in cycle:
					sent_amr.directed_graph.remove(node=node,reconstruct_nx=True)

			connected_components = nx.connected_component_subgraphs(sent_amr.directed_graph.undirected_nx_graph)

			if len(connected_components) > 1:
				# find the largest component
				largest_component = []
				for graph in connected_components:
					print graph.nodes()
					if len(graph.nodes()) > len(largest_component):
						largest_component = graph.nodes()

				for node in sent_amr.directed_graph._graph.keys():
					if node not in largest_component:	sent_amr.directed_graph.remove(node=node)

			print sent_amr.directed_graph._graph

			sent_amr.reconstruct_amr()

		# Creating correct alignments
		for current_word_index in sorted(sent_amr.alignments.keys()):
			for alignment in sent_amr.alignments[current_word_index]:
				# print alignment
				# This is slightly different than the original alignment format
				# Example - ['1', '1', '1', '1', '1', '2', '1']
				# (Rinfinite / root 
				# 	:ARG0 (r / report-01~e.4 										- 1
				# 		:ARG0 (p2 / person    										- 1
				# 			:ARG0-of (h / have-org-role-91 							- 1
				# 				:ARG1 (p / police~e.2 								- 1
				# 					:mod (c / city :wiki "Kathmandu" 				- 1
				# 						:name (n / name :op1 "Kathmandu"~e.1))) 	- 2, then internally - 1
				#	 			:ARG2 (o / officer~e.3))))

				if len(alignment[1:-1]) > 0:
					new_alignment = str(len(' '.join(document_text).split()) + int(current_word_index))\
												+'-'+'1.'+str(sent_index_in_doc+1)+'.'\
												+'.'.join(str(int(x)) for x in alignment[1:-1])
				else:
					new_alignment = str(len(' '.join(document_text).split()) + int(current_word_index))\
												+'-'+'1.'+str(sent_index_in_doc+1)

				if alignment[-1] == 'r':
					new_alignment += '.r'
				else:	
					if len(alignment) > 1:	new_alignment += '.'+str(int(alignment[-1]))

				document_alignments.append(new_alignment)
		# ' <eos> ', has to be used whenever I want to save the text. and ' ' should be used to join sentences
		# whenever I want to find the location of any word in the story.
		document_text.append(sent_amr.text.strip())
		if debug:	print sent_amr.text.strip()

	
		for node_index_in_sent, node in enumerate(sent_amr.amr):
			if node_index_in_sent == 0:
				node['text'] = ':ARG' + str(sent_index_in_doc) + ' ' + node['text']
				node['child_number'] = sent_index_in_doc
				node['parent_index'] = 0
				node['variable_start_index'] += len(':ARG' + str(sent_index_in_doc) + ' ')
				node['variable_end_index'] += len(':ARG' + str(sent_index_in_doc) + ' ')
				current_sent_start_index = len(doc_amr)
			else:
				node['parent_index'] += current_sent_start_index

			if node['variable'] in variable_list:
				# Permitted repetition inside the same sentence
				previous_name = node['variable']
				if node['variable'] in replaced_var_current_sent:
					node['variable'] = replaced_dict_current_sent[node['variable']]
				else:
					replaced_var_current_sent.append(node['variable'])
					replaced_dict_current_sent[node['variable']] = 'var'+str(new_var_index)
					node['variable'] = 'var'+str(new_var_index)
					new_var_index += 1

				variable_start_index = node['variable_start_index']
				variable_end_index = node['variable_end_index']
				# replace variable in the text
				node['text'] = node['text'][ : variable_start_index] + node['variable'] + \
									node['text'][variable_end_index+1 : ]
				node['variable_end_index'] = variable_start_index + len(node['variable']) - 1

			# update other parameters
			variable_end_index = node['variable_end_index']
			node['children_list'] = [current_sent_start_index + x for x in node['children_list']]
			node['depth'] += 1
			node['common_text'] = node['text'][variable_end_index+1:].strip().strip(')')
			# need to differently handle the first node of each sentence
			
			current_sent_variable_list.append(node['variable'])
			var_to_sent[node['variable']] = [sent_index_in_doc]
			doc_amr.append(node)
		# append new sent variable list only when we finish processing this sentence
		variable_list = variable_list + current_sent_variable_list

	# add an extra 'closing paranthesis' in the end
	doc_amr[-1]['text'] = doc_amr[-1]['text'] + ')'
	return doc_amr, document_text, document_alignments, var_to_sent
