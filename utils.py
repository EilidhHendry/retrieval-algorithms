import string


def process_data(filename):
    # initialise result dictionary
    processed_dictionary = {}

    with open(filename) as input_file:
        for line in input_file:

            # lowercase text, remove punctuation, strip trailing newline
            clean_line = line.lower().translate(None, string.punctuation).strip()

            # extract id and text
            doc_id = clean_line.split()[0]
            text = clean_line.split()[1:]

            # store in dictionary
            processed_dictionary[doc_id] = text

    return processed_dictionary


def generate_data(filename):
    with open(filename) as input_file:
        for line in input_file:

            # lowercase text, remove punctuation, strip trailing newline
            clean_line = line.lower().translate(None, string.punctuation).strip()

            # extract id and text
            doc_id = clean_line.split()[0]
            text = clean_line.split()[1:]

            # yield each line as processed
            yield int(doc_id), text


def write_result(result_dict, output_file):
	# write dictionary to file in required format
	for query_id, doc_id in result_dict:
		output_file.write(str(query_id) + ' 0 ' + str(doc_id) + ' 0 ' \
			+ str(result_dict[(query_id, doc_id)]) + ' 0 \n')
	return output_file
