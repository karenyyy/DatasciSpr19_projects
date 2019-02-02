import numpy as np
import pandas as pd
import csv
from itertools import combinations
from itertools import chain

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns

colormap = cm.viridis_r

np.random.seed(100)

import argparse

parser = argparse.ArgumentParser(description='User input parameters')

parser.add_argument('-glove_path',
                    default='',
                    help='path where the 4 GLoVE files are stored')


import nltk
from nltk.corpus import wordnet, brown

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

args = parser.parse_args()
GLOVE_PATH = args.glove_path

GLOVE_50D_PATH = GLOVE_PATH + 'glove.6B/glove.6B.50d.txt'
GLOVE_100D_PATH = GLOVE_PATH + 'glove.6B/glove.6B.100d.txt'
GLOVE_200D_PATH = GLOVE_PATH + 'glove.6B/glove.6B.200d.txt'
GLOVE_300D_PATH = GLOVE_PATH + 'glove.6B/glove.6B.300d.txt'

NOUN_FILENAME = 'nouns.txt'
VERB_FILENAME = 'verbs.txt'

glove_50d = pd.read_table(GLOVE_50D_PATH, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
glove_100d = pd.read_table(GLOVE_100D_PATH, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
glove_200d = pd.read_table(GLOVE_200D_PATH, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
glove_300d = pd.read_table(GLOVE_300D_PATH, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)

'''
# Q1:

## sub 1:
# Randomly select 10 nouns and 10 verbs from a vocabulary resource

## sub 2:
# For each word wi, identify four synonyms, e.g.,  from synsets in WordNet, to generate a synonym list of length 5
# Repeat 5 times until you have 5 synonym lists of length 5 for nouns, and 5 synonym lists of length 5 for verbs.

'''


def create_word_list(testmode=False):
    """Randomly select nouns and verbs from a vocabulary resource"""
    token_list = glove_50d.index.tolist()
    word_list = [token for token in token_list if str(token).isalpha()]

    # test with words directly extracted from GLoVE to figure out why that is not preferred
    if testmode:

        np.random.shuffle(word_list)
        nouns_list = []
        verbs_list = []

        for w in word_list:
            if not isinstance(w, float):
                if nltk.pos_tag([w])[0][1] == 'NN':
                    nouns_list.append(w)
                elif nltk.pos_tag([w])[0][1] == 'VB':
                    verbs_list.append(w)
                else:
                    continue
            else:
                break

    else:
        nouns_list = [word for word, pos in brown.tagged_words() if pos == 'NN']
        verbs_list = [word for word, pos in brown.tagged_words() if pos == 'VB']

    np.random.shuffle(nouns_list)
    np.random.shuffle(verbs_list)

    return word_list, nouns_list, verbs_list


def create_syn_list(words, word_list, filename):
    """5 synonym lists of length 5 for nouns, and 5 synonym lists of length 5 for verbs"""
    count = 0
    with open(filename, 'w') as f_in:
        for w in words:
            if count < 5:
                try:
                    if filename.startswith('n'):
                        synonyms = wordnet.synsets(w, pos='n')
                    elif filename.startswith('v'):
                        synonyms = wordnet.synsets(w, pos='v')
                    else:
                        print('please double check your input filename.')

                    syn_list = list(set(chain.from_iterable([word.lemma_names() for word in synonyms])))
                    syn_list = [syn for syn in syn_list if syn != w][:4]
                    # the first one is the seed word
                    syn_list = [w] + syn_list
                    if len(syn_list) > 4 and all(elem in word_list for elem in syn_list):
                        # print(syn_list)
                        count += 1
                        f_in.write(",".join(syn_list))
                        f_in.write('\n')
                except Exception as e:
                    print(e)
            else:
                break


'''
## sub 3:
# Extract the 4 GLoVE vectors for each of the 50 words (d \in [50,100,200,300])
'''


def creat_glove_dict(filename):
    """Store the 4 GLoVE vectors for nouns and verbs into dictionary"""
    glove_dict = {}
    for word_list in open(filename, 'r'):
        words = word_list.split(',')
        for w in words:
            w = w.strip()
            glove_50d_vec = glove_50d.loc[w, :]
            glove_100d_vec = glove_100d.loc[w, :]
            glove_200d_vec = glove_200d.loc[w, :]
            glove_300d_vec = glove_300d.loc[w, :]
            glove_dict[w] = [list(glove_50d_vec),
                             list(glove_100d_vec),
                             list(glove_200d_vec),
                             list(glove_300d_vec)]
    return glove_dict


'''
# Q2:

## sub 1:
# read in a synonym list (5 words) and their GLoVE vectors for a given dimensionality d, compute the cosine similarities
# for every pair of distinct words (N=choose(5,2)), and save the results in a dictionary
# eg: {(word1, word2): cos_similarity}

## sub 2:
# Compute the mean and standard deviation of the 10 values of cosine similarity for each set of dimension d vectors for synonym list.
# Save code in a single python file named assignment1_<last name>.py
'''


def create_word_cos_dict(filename, glove_dict, dim):
    """compute the cosine similarities for each word pair and save the results in a dictionary"""
    word_cos_dict_lst = []

    for word_list in open(filename, 'r'):
        words = list(map(lambda x: x.strip(), word_list.split(',')))
        word_pairs = combinations(words, 2)
        word_cos_dict = {}
        for word_pair in word_pairs:
            w1, w2 = word_pair
            cos_dim = cosine_similarity([glove_dict[w1][dim]],
                                        [glove_dict[w2][dim]])
            word_cos_dict[(w1, w2)] = cos_dim[0]

        word_cos_dict_lst.append(word_cos_dict)
    return word_cos_dict_lst


'''
# Q3:

## For each synonym list, (N=10) create a csv file with a header row, then one additional row for each pair of distinct words
##  One column for each dimensionality  d \in [50,100,200,300] in ascending order, with cell values equal to the cosine similarity of the d-dimension vectors for word pair (wi,wj)
## Save each csv file as synonym-list-<C>-<N>.csv with category C instantiated as 'N' or 'V' (noun or verb), and N instantiated as an index from 1 to 5.
'''


def create_csv():
    """create 10 csv files to store cosine similarity for synonym lists of nouns and verbs"""
    glove_nouns_dict, glove_verbs_dict = creat_glove_dict('nouns.txt'), creat_glove_dict('verbs.txt')

    noun_50d_cos_dict_lst = create_word_cos_dict(NOUN_FILENAME, glove_nouns_dict, dim=0)
    noun_100d_cos_dict_lst = create_word_cos_dict(NOUN_FILENAME, glove_nouns_dict, dim=1)
    noun_200d_cos_dict_lst = create_word_cos_dict(NOUN_FILENAME, glove_nouns_dict, dim=2)
    noun_300d_cos_dict_lst = create_word_cos_dict(NOUN_FILENAME, glove_nouns_dict, dim=3)

    noun_cos_dict_lst = [noun_50d_cos_dict_lst,
                         noun_100d_cos_dict_lst,
                         noun_200d_cos_dict_lst,
                         noun_300d_cos_dict_lst]

    # cos similarity of verbs 50d, 100d, 200d, 300d

    verb_50d_cos_dict_lst = create_word_cos_dict(VERB_FILENAME, glove_verbs_dict, dim=0)
    verb_100d_cos_dict_lst = create_word_cos_dict(VERB_FILENAME, glove_verbs_dict, dim=1)
    verb_200d_cos_dict_lst = create_word_cos_dict(VERB_FILENAME, glove_verbs_dict, dim=2)
    verb_300d_cos_dict_lst = create_word_cos_dict(VERB_FILENAME, glove_verbs_dict, dim=3)

    verb_cos_dict_lst = [verb_50d_cos_dict_lst,
                         verb_100d_cos_dict_lst,
                         verb_200d_cos_dict_lst,
                         verb_300d_cos_dict_lst]
    noun_cos_mean = []
    noun_cos_sd = []
    verb_cos_mean = []
    verb_cos_sd = []

    for l in range(5):
        noun_cos_df = pd.concat([pd.DataFrame(noun_cos_dict_lst[i][l], index=[0]).transpose()
                                 for i in range(4)],
                                axis=1)
        noun_cos_df.columns = ['d=50', 'd=100', 'd=200', 'd=300']

        noun_cos_mean.append(noun_cos_df.apply(np.mean))
        noun_cos_sd.append(noun_cos_df.apply(np.std))
        print(noun_cos_df)
        noun_cos_df.to_csv('synonym-list-N-{}.csv'.format(l + 1))
        print('synonym-list-N-{}.csv saved!'.format(l + 1))

    for l in range(5):
        verb_cos_df = pd.concat([pd.DataFrame(verb_cos_dict_lst[i][l], index=[0]).transpose()
                                 for i in range(4)],
                                axis=1)
        verb_cos_df.columns = ['d=50', 'd=100', 'd=200', 'd=300']

        verb_cos_mean.append(verb_cos_df.apply(np.mean))
        verb_cos_sd.append(verb_cos_df.apply(np.std))
        print(verb_cos_df)
        verb_cos_df.to_csv('synonym-list-V-{}.csv'.format(l + 1))
        print('synonym-list-V-{}.csv saved!'.format(l + 1))
    return noun_cos_mean, noun_cos_sd, verb_cos_mean, verb_cos_sd


'''
# Q4:

## Create a csv file summarizing the results for nouns with a header row, and a row for each synonym list. Do the same for the results for verbs.
## column 1: file name stem (remove 'csv' extension),
## column 2: mean of cosine similarity of the vectors in that list,
## column 3: standard deviation.of cos sim
## Name the csv files as performance-five-noun-lists.csv, performance-five-verb-lists.csv
'''


def create_summary(noun_cos_mean, noun_cos_sd, verb_cos_mean, verb_cos_sd):
    """performance summary to store filename, mean, sd of cosine similarity and dimension"""
    noun_cos_summary = pd.concat(
        [pd.Series([y for x in [['synonym-list-N-{}'.format(l + 1)] * 4 for l in range(5)] for y in x]),
         pd.DataFrame(np.resize(pd.DataFrame(noun_cos_mean).values, (20, 1))),
         pd.DataFrame(np.resize(pd.DataFrame(noun_cos_sd).values, (20, 1))),
         pd.Series([y for x in [[50, 100, 200, 300] * 5] for y in x])], axis=1)

    noun_cos_summary.columns = ['filename', 'mean', 'sd', 'dim']

    print('sensitivity plot for nouns created!')
    noun_cos_summary.to_csv('performance-five-noun-lists.csv')
    print('performance-five-noun-lists.csv saved!')

    verb_cos_summary = pd.concat(
        [pd.Series([y for x in [['synonym-list-V-{}'.format(l + 1)] * 4 for l in range(5)] for y in x]),
         pd.DataFrame(np.resize(pd.DataFrame(verb_cos_mean).values, (20, 1))),
         pd.DataFrame(np.resize(pd.DataFrame(verb_cos_sd).values, (20, 1))),
         pd.Series([y for x in [[50, 100, 200, 300] * 5] for y in x])], axis=1)

    verb_cos_summary.columns = ['filename', 'mean', 'sd', 'dim']

    print('sensitivity plot for verbs created!')

    verb_cos_summary.to_csv('performance-five-verb-lists.csv')
    print('performance-five-verb-lists.csv saved!')


'''

# Q5:

## Create two plots, one for nouns and one for verbs
## 5 sets of four (x,y), each set has x = d, y = mean cosine similarity (use a whisker plot to show the range, mean, and sd.)
'''


def create_sensitivity_plot(mode='both'):
    """sensitivity plot"""
    if mode == 'both':
        fig = plt.figure(figsize=(15, 8))
        u = ['N', 'V']
        for i in range(2):
            for j in range(1, 6):
                filename = 'synonym-list-{}-{}.csv'.format(u[i], j)
                df = pd.read_csv(filename, sep=',')
                plt.subplot(2, 5, j + 5 * i)
                sns.boxplot(data=df)
                plt.title(filename.split('.')[0])
        plt.show()
        fig.savefig('whiskerplot.png')
    elif mode == 'noun':
        fig = plt.figure(figsize=(20, 5))
        for j in range(1, 6):
            filename = 'synonym-list-N-{}.csv'.format(j)
            df = pd.read_csv(filename, sep=',')
            plt.subplot(1, 5, j)
            sns.boxplot(data=df)
            plt.title(filename.split('.')[0])
        plt.show()
        fig.savefig('sensitivity_mean_noun_plot.png')
    elif mode == 'verb':
        fig = plt.figure(figsize=(20, 5))
        for j in range(1, 6):
            filename = 'synonym-list-V-{}.csv'.format(j)
            df = pd.read_csv(filename, sep=',')
            plt.subplot(1, 5, j)
            sns.boxplot(data=df)
            plt.title(filename.split('.')[0])
        plt.show()
        fig.savefig('sensitivity_mean_verb_plot.png')


def create_performance_plot(filepath_noun, filepath_verb, savefile):
    """performance plot"""
    df_noun = pd.read_csv(filepath_noun, sep=',')
    df_verb = pd.read_csv(filepath_verb, sep=',')
    fig, axarr = plt.subplots(2, 2, figsize=(20, 15))
    colormaps = [colors.rgb2hex(colormap(i)) for i in np.linspace(0, 1, 5)]

    for i, c in enumerate(colormaps):
        x_n = df_noun['dim'][4 * i:4 * (i + 1)]
        y_n_mean = df_noun['mean'][4 * i:4 * (i + 1)]
        y_n_sd = df_noun['sd'][4 * i:4 * (i + 1)]

        l_n = df_noun['filename'][4 * i]

        x_v = df_verb['dim'][4 * i:4 * (i + 1)]
        y_v_mean = df_verb['mean'][4 * i:4 * (i + 1)]
        y_v_sd = df_verb['sd'][4 * i:4 * (i + 1)]

        l_v = df_verb['filename'][4 * i]

        axarr[0, 0].scatter(x_n, y_n_mean, label=l_n, linewidth=2, c=c)
        axarr[0, 0].plot(x_n, y_n_mean, label=l_n, linewidth=2, c=c)

        axarr[0, 1].scatter(x_n, y_n_sd, label=l_n, linewidth=2, c=c)
        axarr[0, 1].plot(x_n, y_n_sd, label=l_n, linewidth=2, c=c)

        axarr[1, 0].scatter(x_v, y_v_mean, label=l_v, linewidth=2, c=c)
        axarr[1, 0].plot(x_v, y_v_mean, label=l_v, linewidth=2, c=c)

        axarr[1, 1].scatter(x_v, y_v_sd, label=l_v, linewidth=2, c=c)
        axarr[1, 1].plot(x_v, y_v_sd, label=l_v, linewidth=2, c=c)

    axarr[0, 0].legend(loc='upper right',
                       ncol=3,
                       fontsize=7)
    axarr[0, 1].legend(loc='upper right',
                       ncol=3,
                       fontsize=7)
    axarr[1, 0].legend(loc='upper right',
                       ncol=3,
                       fontsize=7)
    axarr[1, 1].legend(loc='upper right',
                       ncol=3,
                       fontsize=7)

    axarr[0, 0].set_title('mean cosine of nouns')
    axarr[0, 1].set_title('sd cosine of nouns')
    axarr[1, 0].set_title('mean cosine of verbs')
    axarr[1, 1].set_title('sd cosine of verbs')

    axarr[0, 0].set_xlabel('dimension')
    axarr[0, 1].set_xlabel('dimension')
    axarr[1, 0].set_xlabel('dimension')
    axarr[1, 1].set_xlabel('dimension')

    axarr[0, 0].set_ylabel('mean cosine similarity')
    axarr[0, 1].set_ylabel('sd cosine similarity')
    axarr[1, 0].set_ylabel('mean cosine similarity')
    axarr[1, 1].set_ylabel('sd cosine similarity')

    fig.savefig(savefile)
    plt.show()


def main(testmode):
    # word_list, nouns_list, verbs_list = create_word_list(testmode=testmode)
    #
    # create_syn_list(words=nouns_list, word_list=word_list, filename=NOUN_FILENAME)
    # create_syn_list(words=verbs_list, word_list=word_list, filename=VERB_FILENAME)

    noun_cos_mean, noun_cos_sd, verb_cos_mean, verb_cos_sd = create_csv()

    create_summary(noun_cos_mean, noun_cos_sd, verb_cos_mean, verb_cos_sd)

    create_sensitivity_plot('noun')
    create_sensitivity_plot('verb')

    # for report
    create_sensitivity_plot('both')
    create_performance_plot('performance-five-noun-lists.csv',
                            'performance-five-verb-lists.csv',
                            'mean_sd.png')


if __name__ == '__main__':
    main(testmode=False)
