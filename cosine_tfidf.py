class Cosine_tfidf(object):

    def __init__(self, idf_file):
        self.stored_idfs_dict = {}
        self.stored_idfs_dict = self.initialise_idf_data(idf_file)

    def initialise_idf_data(self, idf_file):
        with open(idf_file) as input_file:
            for line in input_file:
                idf_score, word = line.strip().split()
                self.stored_idfs_dict[word] = float(idf_score)
        return self.stored_idfs_dict

    def calculate_tfidf_stored_idf(self, word, word_freq, default_idf):
        if word in self.stored_idfs_dict:
            tfidf = word_freq * self.stored_idfs_dict[word]
        else:
            tfidf = word_freq * default_idf
        return tfidf

    def calculate_cosine_idf(self, word, freq_dict, default_idf):
        tfidf = 0
        if word in freq_dict:
            if word in self.stored_idfs_dict:
                tfidf = freq_dict[word] * self.stored_idfs_dict[word]
            else:
                tfidf = freq_dict[word] * default_idf
        return tfidf

    def calculate_cosine_tfidf(self, query_freq_dict, doc_freq_dict, default_idf = 13.6332):
        query_sum = 0
        doc_sum = 0
        joint_sum = 0
        for word in set(query_freq_dict.keys() + doc_freq_dict.keys()):
            query_tfidf = self.calculate_cosine_idf(word, query_freq_dict, default_idf)
            doc_tfidf = self.calculate_cosine_idf(word, doc_freq_dict, default_idf)

            query_sum += query_tfidf**2
            doc_sum += doc_tfidf**2

            joint_sum += query_tfidf * doc_tfidf

        cosine_score = float(joint_sum) / ((query_sum * doc_sum)**0.5)
        return cosine_score
