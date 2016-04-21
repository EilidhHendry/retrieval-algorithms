import utils
import math

doc_freqs={}

def average_length(doc_dict):
    # Compute average document length (used to compute the tdf portion of the score)
    total_length = 0
    for doc_id in doc_dict:
        doc_length = len(doc_dict[doc_id])
        total_length += doc_length
    average_doc_length = float(total_length)/len(doc_dict)
    return average_doc_length


def find_term_freq_doc(word, doc_dict):
    if word not in doc_freqs:
        doc_freq = len([1 for doc_id in doc_dict if word in doc_dict[doc_id]])
        doc_freqs[word] = doc_freq
        return doc_freq
    else:
        return doc_freqs[word]


def calculate_tfidf(query_dict, doc_dict, k=2):

    # find the average doc length
    average_doc_length = average_length(doc_dict)

    # initialise dictionary for results
    score_dict = {}

    for query_id in query_dict:
        # get the text
        query_text = query_dict[query_id]

        for doc_id in doc_dict:
            doc_text = doc_dict[doc_id]

            score = 0
            for word in set(query_text):
                # term frequency for the query and document
                term_freq_query = query_text.count(word)
                term_freq_doc = doc_text.count(word)

                # find number of occurrences of word in all documents
                dfw = find_term_freq_doc(word, doc_dict)

                # skip if 0, as query word not in docs
                if not dfw:
                    continue

                # idf = log(|C|/dfw)
                idf = math.log(float(len(doc_dict))/dfw)

                # tdf = tfwd / (tfwd + ((k * |D|) / avgD))
                tdf = float(term_freq_doc)/(term_freq_doc + (k * len(doc_text) / average_doc_length))

                # tfidf = tfqd * tdf * idf
                score += term_freq_query * tdf * idf

            score_dict[(query_id, doc_id)] = score

    return score_dict


if __name__ == "__main__":
    query_dict = utils.process_data('data/qrys.txt')
    doc_dict = utils.process_data('data/docs.txt')

    overlap_scores = calculate_tfidf(query_dict, doc_dict)

    with open('results/tfidf.top', 'w') as output_file:
        output_file = utils.write_result(overlap_scores, output_file)
