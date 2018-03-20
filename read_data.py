import codecs

# Takes input the location of the gold_standard AMRs
# Returns the list of documents read
# Each document is list of dictionaries - dict['snt-type'] = \
# [corresponding AMR in a list, each element of the list is a line from the AMR]

def read_data(location_data):
	with codecs.open(location_data, 'r') as data_file:
		#List of stories in list 'data'
		data = []
		data = data_file.readlines()
	# save the documents in a list 'docs', as a list of dictionaries
	docs =[]
	target_summaries = []
	new_doc = []
	stories = []
	temp_taget_summary = ''
	temp_story = ''
	for index, line in enumerate(data):
		if '::snt-type date' in line:
			# Save the previous doc
			if '::snt-type date' in new_doc[0]:
				if len(temp_taget_summary) != 0 or len(temp_story) != 0:
					docs.append(new_doc)
					target_summaries.append(temp_taget_summary)
					temp_taget_summary = ''
					stories.append(temp_story)
					temp_story = ''
			# Create the new documet
			new_doc = []
			new_doc.append(line)
		else:
			new_doc.append(line)
		if '::snt-type summary' in line:
			temp_line = data[index+2].split('::tok')[1]
			if temp_line.strip()[-1] == '.' :
				temp_line = temp_line.strip()[:-1]
			else:
				temp_line = temp_line.strip()
			temp_taget_summary = temp_taget_summary + ' ' + temp_line + ' .'
		if '::snt-type body' in line:
			temp_line = data[index+2].split('::tok ')[1]
			if len(temp_line.strip()) == 0:
				temp_line = ' '
			elif temp_line.strip()[-1] == '.':
				temp_line = temp_line.strip()[:-1]
			else:
				temp_line = temp_line.strip()
			if len(temp_line) > 0:
				temp_story = temp_story + ' ' + temp_line + ' .'
	# Every doc is list of dictionaries, each dictionary contain 'snt_type', 'alignments', 'amr' etc as keys
	docs.append(new_doc)
	stories.append(temp_story)
	target_summaries.append(temp_taget_summary)
	for index, doc in enumerate(docs):
		read_amr = False
		temp_amr = []
		start_reading = False
		modified_doc = []
		new_dict = {}
		for line in doc:
			if 'snt-type' in line:
				tokens = line.split(' ')
				new_dict['snt-type'] =  tokens[tokens.index('::snt-type') + 1] 
				read_amr = True
			if '::alignments' in line:
				new_dict['alignments'] =  line.strip().split(' ')[2:]
				new_dict['alignments_line'] =  line
			if '::snt' in line:
				new_dict['snt'] =  line.split('::snt')[1].strip()
				if new_dict['snt'].strip()[-1] != '.': new_dict['snt'] = new_dict['snt'] + '.' 
			if '::tok' in line:
				new_dict['tok'] =  line.split('::tok ')[1].strip()
				if new_dict['tok'].strip()[-1] != '.': new_dict['tok'] = new_dict['tok'] + ' .' 
			if '::id' in line:
				new_dict['id'] =  line
			if read_amr:
				if line == '\n':
					read_amr = False
					start_reading = False
					new_dict['amr'] = temp_amr
					temp_amr = []
					modified_doc.append(new_dict)
					new_dict = {}
				if line[0] == '(':
					start_reading = True
				if start_reading == True:
					temp_amr.append(line)
		docs[index] = modified_doc
	return docs, target_summaries, stories
