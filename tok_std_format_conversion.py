import sys
import argparse
import itertools
import re

def std_to_tok_format_convertor(std_format_line='',without_end_period=True):
	line = std_format_line.split()
	# print line
	# break the '-'
	tok_format_line = []
	for word in line:
		if '--' not in word and '-' in word and word != '-':
			new_word = word.split('-')
			if new_word[0] == '':	
				tok_format_line.extend([new_word[0],'-@',new_word[1]])
				if len(new_word) == 2:	continue
			if new_word[1] == '':
				tok_format_line.extend([new_word[0],'@-',new_word[1]])
				continue
			tok_format_line.extend([' @-@ '.join(new_word)])
			continue
		tok_format_line.append(word)

	line = ' '.join(tok_format_line)
	line = line.split()
	# break the ' 's '
	tok_format_line = []
	for word in line:
		if '\'' in word:
			new_word = word.split('\'')
			# print new_word, new_word[0]
			if new_word[0] in ['hasn', 'don']:
				# print 'here'
				tok_format_line.extend([new_word[0][:-1],'n\''+new_word[1]])
				continue
			tok_format_line.extend([new_word[0],'\''+new_word[1]])
			continue
		tok_format_line.append(word)

	line = ' '.join(tok_format_line)
	line = line.split()

	# break the ','
	tok_format_line = []
	for word in line:
		if ',' in word:
			if not re.search('[a-zA-Z]', word) and word[-1]!=',':	tok_format_line.append(word)
			else:
				new_word = word.split(',')
				tok_format_line.extend([new_word[0],','+new_word[1]])
			continue
		tok_format_line.append(word)

	line = ' '.join(tok_format_line)
	line = line.split()

	# break the '('
	tok_format_line = []
	for word in line:
		if '(' in word:	
			new_word = word.split('(')
			tok_format_line.extend([new_word[0],'(',new_word[1]])
			continue
		tok_format_line.append(word)

	line = ' '.join(tok_format_line)
	line = line.split()

	# break the ')'
	tok_format_line = []
	for word in line:
		if ')' in word:	
			new_word = word.split(')')
			tok_format_line.extend([new_word[0],')',new_word[1]])
			continue
		tok_format_line.append(word)

	line = ' '.join(tok_format_line)
	line = line.split()

	# break the '$'
	tok_format_line = []
	for word in line:
		if '$' in word:	
			new_word = word.split('$')
			tok_format_line.extend([new_word[0],'$',new_word[1]])
			continue
		tok_format_line.append(word)

	line = ' '.join(tok_format_line)
	line = line.split()

	# print line

	# break alpha-numerics
	tok_format_line = []
	for word in line:
		if not re.search('[a-zA-Z]', word):	tok_format_line.append(word)
		else: tok_format_line.extend(["".join(x) for _, x in itertools.groupby(word, key=str.isdigit)])

	line = ' '.join(tok_format_line)
	if without_end_period:	return line

	# Hanlding the end period
	if line.endswith('.'):
		line = line[:-1] + ' .'
	line = line.split()

	return ' '.join(line)

def tok_to_std_format_convertor(tok_format_line=''):
	line = tok_format_line.split()
	# break the '-'
	std_format_line = ''
	for index, word in enumerate(line):
		try:
			if std_format_line[-1] in ['-','(','$'] and std_format_line[-2:-1] not in ['-']:
				if line[index-1] == '@-' and std_format_line[-1] in ['-']:
					std_format_line += ' ' + word
					continue	
				std_format_line += word
				continue
		except:	pass
		if '-' in word and '--' not in word:
			if '@-' == word:	
				std_format_line += '-'
			if '-@' == word:
				std_format_line += ' -'
			if '@-@' == word:
				std_format_line += '-'
			continue
		if word.startswith('\''):
			std_format_line += word
			continue
		if word.startswith('.'):
			std_format_line += word
			continue
		if word in ['th','',',',')','s','n\'t']:
			if word in ['s'] and std_format_line[-1] == '\'':
				std_format_line +=  ' ' + word
				continue
			std_format_line += word
			continue

		if std_format_line != '':	std_format_line += ' ' + word
		else:	std_format_line += word
	return std_format_line

def compare_std_to_tok(std_lines=[],tok_lines=[]):

	for index, std_line in enumerate(std_lines):
		# print std_line
		predicted_tok_line=std_to_tok_format_convertor(std_format_line=std_line)
		# print predicted_tok_line
		# 0/0
		if predicted_tok_line.strip() != tok_lines[index].strip():
			print 'std_to_tok conversion failed'
			print predicted_tok_line.strip()
			print tok_lines[index].strip()
			0/0

def compare_tok_to_std(std_lines=[],tok_lines=[]):

	for index, tok_line in enumerate(tok_lines):
		predicted_std_line=tok_to_std_format_convertor(tok_format_line=tok_line)
		if predicted_std_line.strip() != std_lines[index].strip():
			print 'tok_to_std conversion failed'
			print predicted_std_line.strip()
			print std_lines[index].strip()
			0/0

def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--std_file', help="Path of the file containing stories in 'std' format",
		type=str, default='std_stories.txt')
	parser.add_argument('--tok_file', help="Path of the file containing stories in 'tok' format",
		type=str, default='tok_stories.txt')

	args = parser.parse_args(arguments)
	std_file = args.std_file
	tok_file = args.tok_file

	with open(std_file,'r') as f:
		std_stories = f.readlines()
	with open(tok_file,'r') as f:
		tok_stories = f.readlines()

	std_lines = []
	for story in std_stories:
		std_lines.extend(story.split('<eos>'))


	tok_lines = []
	for story in tok_stories:
		tok_lines.extend(story.split('<eos>'))

	compare_std_to_tok(std_lines=std_lines,tok_lines=tok_lines)
	compare_tok_to_std(std_lines=std_lines,tok_lines=tok_lines)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
