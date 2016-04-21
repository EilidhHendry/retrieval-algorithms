import utils

def calculate_overlap(query_dict, doc_dict):
	result = {}
	# compare every query to every doc and find overlap
	for query_id in query_dict:
		# get the query text from the dictionary
		query_text = query_dict[query_id]
		# Remove duplicate words
		query_text = set(query_text)

		for doc_id in doc_dict:
			# retrieve the document text
			doc_text = doc_dict[doc_id]

			# remove duplicate words
			doc_text = set(doc_text)

			# find the overlap between query and document
			overlap = 0
			for word in query_text:
				if word in doc_text:
					overlap += 1

			# store overlap score in dictionary with id tuple as key
			result[(query_id, doc_id)] = overlap

	return result


if __name__ == '__main__':
	query_dict = utils.process_data('data/qrys.txt')
	doc_dict = utils.process_data('data/docs.txt')

	overlap_scores = calculate_overlap(query_dict, doc_dict)

	with open('results/overlap.top','w') as output_file:
		output_file = utils.write_result(overlap_scores, output_file)
