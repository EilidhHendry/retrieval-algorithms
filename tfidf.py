import utils
import math
import sortedcontainers
import nltk

# initialise dictionaries/list to ease computation
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


def calculate_tfidf(query_text, doc_dict, average_doc_length, k):
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

        yield doc_id, score


def standard_tfidf(query_dict, doc_dict, k=2):
    # find the average doc length
    average_doc_length = average_length(doc_dict)

    # initialise dictionary for results
    score_dict = {}

    for query_id in query_dict:
        # get the text
        query_text = query_dict[query_id]

        for doc_id, tfidf_score in calculate_tfidf(query_text, doc_dict, average_doc_length, k):
            score_dict[(query_id, doc_id)] = tfidf_score

    return score_dict


def find_ranking(top_ranking, doc_id, score, max_len):
    top_ranking.add((doc_id, score))
    # remove item with lowest score if length greater than max len
    if len(top_ranking) > max_len:
        del top_ranking[-1]
    return top_ranking


def tfidf_pseudo_relevance_feedback(query_dict, doc_dict, k=2, num_docs=15, num_words=30):
    # find the average doc length
    average_doc_length = average_length(doc_dict)

    # initialise dictionary for results
    score_dict = {}

    for query_id in query_dict:
        # get the text
        query_text = query_dict[query_id]
        # initialise sorted list which sorts by the score
        top_ranking=sortedcontainers.SortedListWithKey(key=lambda (key, value): (value, key))

        for doc_id, tfidf_score in calculate_tfidf(query_text, doc_dict, average_doc_length, k):
            top_ranking = find_ranking(top_ranking, doc_id, tfidf_score, num_docs)

        # go through top ranking docs gather all words
        # TODO: improve
        top_doc_word_list = []
        for doc_id, _ in top_ranking:
            doc_text = doc_dict[doc_id]
            top_doc_word_list+=doc_text

        # find the most frequent words
        freq_dist = nltk.FreqDist(word for word in top_doc_word_list)
        best_words = freq_dist.keys()[:num_words]

        # add to the query
        new_query = query_text + best_words

        # recalculate tfidf score and add to score dictionary
        for doc_id, tfidf_score in calculate_tfidf(new_query, doc_dict, average_doc_length, k):
            score_dict[query_id, doc_id] = tfidf_score

    return score_dict


if __name__ == "__main__":
    query_dict = utils.process_data('data/qrys.txt')
    doc_dict = utils.process_data('data/docs.txt')

    standard_tfidf_scores = standard_tfidf(query_dict, doc_dict)

    with open('results/tfidf.top', 'w') as output_file:
        output_file = utils.write_result(standard_tfidf_scores, output_file)

    tfidf_with_prf_scores = tfidf_pseudo_relevance_feedback(query_dict, doc_dict)

    with open('results/best.top', 'w') as output_file:
        output_file = utils.write_result(tfidf_with_prf_scores, output_file)
