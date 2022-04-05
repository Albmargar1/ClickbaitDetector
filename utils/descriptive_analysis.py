import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

plt.rcParams['font.size'] = '13' # Font size in matplotlib figures

class bcolors:
    okblue = '\033[094'
    blue = '\033[34m'
    endc = '\033[0m'

def overall_info(data_train):
    legit_tweets = data_train[data_train['truthClass'] == 0]
    clickbait_tweets = data_train[data_train['truthClass'] == 1]

    legit_count = legit_tweets.shape[0]
    clickbait_count = clickbait_tweets.shape[0]
    bad_labeled_count = data_train.shape[0] - (legit_count + clickbait_count)
    clickbait_proportion = clickbait_count/(clickbait_count + legit_count+bad_labeled_count)

    print(bcolors.blue,
          '\n\nTweets legítimos:', legit_count, 
          '\nTweets clickbait:', clickbait_count, 
          '\nTweets mal etiquetados:', bad_labeled_count,
          '\nProporción tweets clickbait:', clickbait_proportion, 
          bcolors.endc)
          
def study_std(data_train): 
    truth_judgments = [eval(row) for row in data_train['truthJudgments']]
    truth_judgments = np.array(truth_judgments)
    truth_mean = np.mean(truth_judgments, axis=1)
    truth_std = np.std(truth_judgments, axis=1)

    f, ax = plt.subplots(figsize=(4,8))
    plt.boxplot(truth_std)
    plt.title('Variabilidad de los anotadores')
    print('\n\n') 
    plt.show()
    
def probability_clickbait_per_tweet_count_words(data_train):
    tweets_clickbaits = data_train['postText'].mask(data_train['truthClass']==0).dropna().reset_index(drop=True).str.split()
    tweets_noclickbaits = data_train['postText'].mask(data_train['truthClass']==1).dropna().reset_index(drop=True).str.split()
    assert tweets_clickbaits.shape[0] + tweets_noclickbaits.shape[0] == data_train.shape[0], 'Error, hay etiquetas diferentes de \'clickbait\' y \'no-clickbait\''

    clickbaits_word_len = list(tweets_clickbaits.str.len())
    noclickbaits_word_len = list(tweets_noclickbaits.str.len())
    max_tweet_word_len = max(max(clickbaits_word_len), max(noclickbaits_word_len))

    dic_clickbaits_word_freq = {i: clickbaits_word_len.count(i) for i in range(max_tweet_word_len+1)}
    dic_noclickbaits_word_freq = {i: noclickbaits_word_len.count(i) for i in range(max_tweet_word_len+1)}
    assert sum(dic_clickbaits_word_freq.values()) + sum(dic_noclickbaits_word_freq.values()) == data_train.shape[0], 'Error al contabilizar frecuencias'

    x = dic_clickbaits_word_freq.keys()
    dic_tweets_word_freq = [dic_clickbaits_word_freq[i] + dic_noclickbaits_word_freq[i] for i in range(max_tweet_word_len+1)]
    y1 = [dic_clickbaits_word_freq[i]   / dic_tweets_word_freq[i] for i in x] 
    y2 = [dic_noclickbaits_word_freq[i] / dic_tweets_word_freq[i] for i in x] 

    # Plot
    f, ax1 = plt.subplots(figsize=(12,8))
    ax2 = ax1.twinx()
    ax1.bar(x, 1, 0.4, label = 'prob. no-clickbait')
    ax1.bar(x, y1, 0.4, label = 'prob. clickbait')
    ax2.scatter(x, dic_tweets_word_freq, color='red', s=65, label = 'número tweets')
    ax1.set_xlabel("Número palabras en tweet")
    ax1.set_ylabel("Probabilidad clickbait")
    ax2.set_ylabel("Cantidad de tweets con un número especifico de palabras")
    plt.title("Probabilidad de clickbait en función del número de palabras por tweet")
    ax1.legend(loc="upper left")
    ax2.legend()
    ax1.set_ylim([0, 1.15])
    ax2.set_ylim([0, 2500])
    print('\n\n') 
    plt.show()
    
def given_word_check_probability_clickbait(data_train):
    def build_frequency(dataframe, column_label):
      counter = Counter()
      dataframe[column_label].str.lower().str.split().apply(counter.update)
      return counter

    def build_probabilities(counter, vocab):
      counter_all_words_freq = sum(counter.values())
      vocab_length = len(vocab)
      prob = {k: (counter.get(k, 0) + 1) / (counter_all_words_freq + vocab_length) for k in vocab.keys()}

      return prob

    def sigmoid(x):
      return 1/(1 + np.exp(-x))

    legit_tweets = data_train[data_train['truthClass'] == 0]
    clickbait_tweets = data_train[data_train['truthClass'] == 1]

    vocab_freq = build_frequency(data_train, 'postText')
    legit_freq = build_frequency(legit_tweets, 'postText')
    clickbait_freq = build_frequency(clickbait_tweets, 'postText')
      
    legit_word_prob = build_probabilities(legit_freq, vocab_freq)
    clickbait_word_prob = build_probabilities(clickbait_freq, vocab_freq)

    loglikelihood = {k: np.log(clickbait_word_prob.get(k) / legit_word_prob.get(k)) for k in vocab_freq.keys()}
    keys_list = [k for k in vocab_freq.keys()]
    word_loglikelihood_array = np.array([sigmoid(v) for v in loglikelihood.values()])
    word_count_array = np.array([v for v in vocab_freq.values()])

    mask = (word_count_array > 180) & (abs(word_loglikelihood_array - 0.5) > 0.2)
    mask_keys = {i: keys_list[i] for i in range(mask.shape[0]) if mask[i] == True}

    fig, ax = plt.subplots(figsize=(12,8))
    ax.scatter(word_loglikelihood_array[mask], word_count_array[mask])
    for k,v in mask_keys.items():
      ax.annotate(v, (word_loglikelihood_array[k], word_count_array[k]))
    ax.set_xlabel('Probabilidad de que la palabra pertenezca a un tweet clickbait')
    ax.set_ylabel('Número de veces que se repite la palabra en el dataset')
    plt.title('Probabilidad de que la palabra aparezca en un tweet clickbait')
    print('\n\n') 
    plt.show()