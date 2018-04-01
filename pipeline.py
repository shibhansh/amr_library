import codecs
import os
import pickle
import nltk
from nltk.corpus import stopwords
import collections
import operator
import sys
import copy
from matplotlib import pylab
import networkx as nx

from read_data import *
from resolve_coref import *
from generate_document_graph import *
from tok_std_format_conversion import *
from directed_graph import Graph
from amr import AMR
# from concept_relation_list import concept_relation_list

def save_stories(stories,path=''):
	if path == '':
		path = dataset+"/stories_"+dataset+".txt"
	os.system("touch "+path)
	f = codecs.open(path,'w')
	for i in range(0,len(stories)):
		f.write(stories[i])
		f.write('\n')
	f.close()

def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--input_file', help="Path of the file containing AMRs of each sentence", type=str, 
				default='/home/prerna/Documents/thesis_work/LDC2015E86_DEFT_Phase_2_AMR_Annotation_R1/' + \
				'data/amrs/split/test/deft-p2-amr-r1-amrs-test-alignments-proxy.txt')
	parser.add_argument('--dataset', help="Name of dataset",
				type=str, default='')
	parser.add_argument('--display', help="Path of the file containing AMRs of each sentence",
				type=bool, default=False)

	args = parser.parse_args(arguments)

	input_file = args.input_file
	dataset = args.dataset


	# '''
	# 'docs' is a list of 'documents', each 'document' is list a dictionary. Each dictionary contains
	# information about a sentence. Each dicitonary has 'alignments', 'amr' etc. keys. Corresponding
	# to each key we have the relevant information like the amr, text, alignment etc.
	# '''

	# Remove alignments from the new file
	os.system('cp '+ input_file +' auxiliary/temp')
	with codecs.open('auxiliary/temp', 'r') as data_file:
		original_data = data_file.readlines()

	os.system('sed -i \'s/~e.[	0-9]*//g\' auxiliary/temp')
	os.system('sed -i \'s/,[	0-9]*//g\' auxiliary/temp')

	with codecs.open('auxiliary/temp', 'r') as data_file:
		data = data_file.readlines()
	for index_line,line in enumerate(data):
		if line.startswith('#'):
			data[index_line] = original_data[index_line]

	with codecs.open('auxiliary/temp', 'w') as data_file:
		for line in data:
			data_file.write(line)

	input_file = 'auxiliary/temp'

	docs, target_summaries, stories = read_data(input_file)

	os.system('rm auxiliary/temp')
	save_stories(stories,'auxiliary/stories.txt')

	with open('auxiliary/target_summaries.txt','w') as f:
		for summary in target_summaries:
			f.write(tok_to_std_format_convertor(summary)+'\n')

	f = open('auxiliary/predicted_summaries.txt','w')
	summary_sentences_per_story = []
	# currently all the information of a node is stored as a list, changing it to a dictionary
	debug = False
	# 'document_amrs' is the list of document amrs formed after joining nodes and collapsing same entities etc.
	target_summaries_amrs = []
	predicted_summaries_amrs = []
	document_amrs = []
	selected_sents = []
	for index_doc, doc in enumerate(docs):
		current_doc_sent_amr_list = []
		current_target_summary_sent_amr_list = []
		for index_dict, dict_sentence in enumerate(doc):
			if dict_sentence['amr'] != []:
				if dict_sentence['tok'].strip()[-1] != '.': dict_sentence['tok'] = dict_sentence['tok'] + ' .' 
				# Get the AMR class for each sentence using just the text
				if dict_sentence['snt-type'] == 'summary':
					current_target_summary_sent_amr_list.append(AMR(dict_sentence['amr'],
													amr_with_attributes=False,
													text=dict_sentence['tok'],
													alignments=dict_sentence['alignments']))
				if dict_sentence['snt-type'] == 'body':
					docs[index_doc][index_dict]['amr'] = AMR(dict_sentence['amr'],
														amr_with_attributes=False,
														text=dict_sentence['tok'],
														alignments=dict_sentence['alignments'])
					current_doc_sent_amr_list.append(docs[index_doc][index_dict]['amr'])
		# merging the sentence AMRs to form a single AMR
		amr_as_list, document_text, document_alignments,var_to_sent = \
												merge_sentence_amrs(current_doc_sent_amr_list,debug=False)
		new_document_amr = AMR(text_list=amr_as_list,
							text=document_text,
							alignments=document_alignments,
							amr_with_attributes=True,
							var_to_sent=var_to_sent)
		document_amrs.append(new_document_amr)
		target_summaries_amrs.append(current_target_summary_sent_amr_list)
		imp_doc = index_doc
		if imp_doc == 1000:
			# just the first sentence of the story is the summary
			predicted_summaries_amrs.append([current_doc_sent_amr_list[0]])

		print index_doc 
		if index_doc == imp_doc:
			document_amrs[index_doc] = resolve_coref_doc_AMR(amr=document_amrs[index_doc],
									resolved=True,story=' '.join(document_amrs[index_doc].text),
									# location_of_resolved_story='auxiliary/human_corefs.txt',
									location_of_resolved_story='auxiliary/'+dataset+'_predicted_resolutions.txt',
									location_of_story_in_file=index_doc,
									location_of_resolver='.',
									debug=False)

			pr = document_amrs[index_doc].directed_graph.rank_sent_in_degree()
			ranks, weights = zip(*pr)
			print ranks
			print weights

			# get pairs in order of importance
			ranked_pairs = document_amrs[index_doc].directed_graph.rank_pairs(ranks=ranks,
							weights=weights,pairs_to_rank=3)
			# print 'ranked_pairs', ranked_pairs
			paths_and_sub_graphs = document_amrs[index_doc].directed_graph.max_imp_path(
							ordered_pairs=ranked_pairs)

			# add method to check no repeated sub_graph
			summary_paths = []
			summary_amrs = []
			summary_amrs_text = []
			for path_and_sub_graph in paths_and_sub_graphs:
				path, sub_graph, sent = path_and_sub_graph

				path_sent_dict = {}
				if sent == -1: path_sent_dict = document_amrs[index_doc].break_path_by_sentences(path=path)
				else: path_sent_dict[sent] = path

				for key in path_sent_dict.keys():
					temp_path = path_sent_dict[key]

					# path = document_amrs[index_doc].concept_relation_list.get_concepts_given_path(sent_index=key,path=temp_path)
					path = -1
					# key = 0
					if path == -1:	path = document_amrs[index_doc].get_sent_amr(sent_index=key)

					nodes, sub_graph = document_amrs[index_doc].directed_graph.get_name_path(nodes=path)

					new_amr_graph = document_amrs[index_doc].get_AMR_from_directed_graph(sub_graph=sub_graph)

					repeated_path = False
					# removing repreating sents/amrs
					for var_set in summary_paths:
						if set(var_set) == set(nodes):	repeated_path = True

					if repeated_path:	continue

					summary_paths.append(list(nodes))
					summary_amrs_text.append(new_amr_graph.print_amr(file=f,print_indices=False,
						write_in_file=True,one_line_output=True,return_str=True,to_print=False))
					print ''
					summary_amrs.append(new_amr_graph)

			final_summary_amrs_text = []
			final_summary_amrs = []
			for index, path in enumerate(summary_paths):
				indices_to_search_at = range(len(summary_paths))
				indices_to_search_at.remove(index)
				to_print = True
				for index_2 in indices_to_search_at:
					if set(path) < set(summary_paths[index_2]):
						to_print = False
				if to_print:
					final_summary_amrs_text.append(summary_amrs_text[index])
					final_summary_amrs.append(summary_amrs[index])
			
			for summary_amr in final_summary_amrs_text:
				try:	summary_sentences_per_story[index_doc] += 1
				except:	summary_sentences_per_story.append(1)

				print summary_amr

			predicted_summaries_amrs.append(final_summary_amrs)

	with open('auxiliary/'+dataset+'_eos_stories.txt','w') as f:
		for document_amr in document_amrs:
			f.write(' <eos> '.join(document_amr.text)+'\n')

	f.close()
	with open('auxiliary/num_sent_per_story.txt','w') as f3:
		pickle.dump(summary_sentences_per_story,f3)
	# save document AMR in file
	with open('auxiliary/text_amr.txt','w') as f2:
		f2.write('# :id PROXY_AFP_ENG_20050317_010.10 ::amr-annotator SDL-AMR-09  ::preferred ::snt-type body\n')
		f2.write('# ::snt On 21 March 2005\n')
		f2.write('# ::tok On 21 March 2005\n')
		if imp_doc >= 0 and imp_doc < len(document_amrs):
			for index_node, node in enumerate(document_amrs[imp_doc].amr):
				f2.write('\t'*node['depth']+node['text']+'\n')

		# an option to generate the graphical representations
		# return document_amrs
	target_summaries_nodes = []
	for target_summary_amrs in target_summaries_amrs:
		current_summary_nodes = []
		for target_summary_amr in target_summary_amrs:
			current_summary_nodes.extend(target_summary_amr.get_nodes() )
		target_summaries_nodes.append(current_summary_nodes)

	with open('auxiliary/target_summary_nodes.txt','w') as f6:
		for node_list in target_summaries_nodes:
			f6.write(' '.join([node for node in node_list]) + '\n')

	predicted_summaries_nodes = []
	for predicted_summary_amrs in predicted_summaries_amrs:
		current_summary_nodes = []
		for predicted_summary_amr in predicted_summary_amrs:
			current_summary_nodes.extend(predicted_summary_amr.get_nodes() )
		predicted_summaries_nodes.append(current_summary_nodes)

	with open('auxiliary/predicted_summary_nodes.txt','w') as f7:
		for node_list in predicted_summaries_nodes:
			f7.write(' '.join([node for node in node_list]) + '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
