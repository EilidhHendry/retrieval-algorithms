import utils
from cosine_tfidf import Cosine_tfidf


def calculate_term_frequencies(text):
    freq_dict = {word: text.count(word) for word in text}
    return freq_dict


def indexed_comparison(input_file, idf_file, optimal_cosine_score = 0.2):
    term_freq_dict = {}
    articles = {}
    cosine_tfidf_instance = Cosine_tfidf(idf_file)
    for article_a_id, article_a_text in utils.generate_data(input_file):

        # calculate current article's term frequency and store
        article_a_term_freq = calculate_term_frequencies(article_a_text)
        articles[article_a_id] = article_a_term_freq

        current_cosine_score = 0
        best_article_id = 0
        # loop back through all the previous articles
        for article_b_id in articles:
            if article_b_id == article_a_id:
                continue

            article_b_term_freq = articles[article_b_id]
            new_cosine_score = cosine_tfidf_instance.calculate_cosine_tfidf(article_a_term_freq, article_b_term_freq)

            if new_cosine_score > current_cosine_score:
                best_article_id = article_b_id
                current_cosine_score = new_cosine_score

        if current_cosine_score > optimal_cosine_score:
            yield article_a_id, best_article_id


def rank_articles(input_file, idf_file, max_len = 10000):
    with open('results/pairs.out', 'w') as output_file:
        for article_a_id, article_b_id in indexed_comparison(input_file, idf_file):
            if article_a_id == max_len:
                output_file.flush()
                break
            if article_a_id % 500 == 0:
                print str(article_a_id) + ' articles processed'
                output_file.flush()
            output_file.write(str(article_a_id) + ' ' + str(article_b_id) + '\n')


if __name__ == '__main__':
    input_file = 'data/news.txt'
    idf_file = 'data/news.idf'
    rank_articles(input_file, idf_file)
