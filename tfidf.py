import utils
import math


def average_length(doc_dict):
    # Compute average document length (used to compute the tdf portion of the score)
    total_length = 0
    for doc_id in doc_dict:
        doc_length = len(doc_dict[doc_id])
        total_length += doc_length
    average_doc_length = float(total_length)/len(doc_dict)
    return average_doc_length


def calculate_tfidf(query_dict, doc_dict, k=2):

    # find the average doc length
    average_doc_length = average_length(doc_dict)

    # initialise dictionary for results
    score_dict = {}

    for query_id in query_dict:
        # get the text
        query_text = query_dict[query_id]

        # get term frequencies for each unique word in the query
        query_freqs = {}
        for word in set(query_text):
            query_freqs[word] = query_text.count(word)

        # compute document frequency for the words in the query
        doc_freqs = {}
        for word in set(query_text):
            if word not in doc_freqs:
                doc_freqs[word] = 0

        for doc_id in doc_dict:
            if word in doc_dict[doc_id]:    # If the word is in a document then
                doc_freqs[word] += 1     # add 1 to its doc frequency

        for doc_id in doc_dict:
            doc_text = doc_dict[doc_id]

            doc_freqs = {}
            for word in set(query_text):
                doc_freqs[word] = doc_text.count(word)

            score = 0
            for word in query_text:
                term_freq_q = query_freqs[word]     # term frewuency fo the query
                term_freq_d = doc_freqs[word]       # term frequency for the document
                if term_freq_d == 0:
                    continue
                df = doc_freqs[word]
                idf = math.log(len(doc_dict)/float(df))
                tdf = float(term_freq_d)/(term_freq_d + (k * len(doc_text) / average_doc_length))
                score += term_freq_q * tdf * idf

                score_dict[(query_id, doc_id)] = score

    return score_dict


if __name__ == "__main__":
    query_dict = utils.process_data('qrys.txt')
    doc_dict = utils.process_data('docs.txt')

    overlap_scores = calculate_tfidf(query_dict, doc_dict)

    with open('tfidf.top', 'w') as output_file:
        output_file = utils.write_result(overlap_scores, output_file)
