from xml.dom import minidom
import os
from collections import defaultdict
import pandas as pd

PATH = os.path.dirname(os.path.abspath(__file__))

TRAIN_PATH = os.path.join(PATH, 'train')
DEV_PATH = os.path.join(PATH, 'dev')
TEST_NO_LEX_PATH = os.path.join(PATH, 'testdata_no_lex.xml')
TEST_WITH_LEX_PATH = os.path.join(PATH, 'testdata_unseen_with_lex.xml')

SRC_TRAIN_PATH = '/home/karen/workspace/OpenNMT-py/rdf2text/datasets/train.src'
SRC_DEV_PATH = '/home/karen/workspace/OpenNMT-py/rdf2text/datasets/dev.src'
SRC_TEST_PATH = '/home/karen/workspace/nmt/webnlg/nmt_data/nmt_data/test.vi'

TGT_TRAIN_PATH = '/home/karen/workspace/OpenNMT-py/rdf2text/datasets/train.tgt'
TGT_DEV_PATH = '/home/karen/workspace/OpenNMT-py/rdf2text/datasets/dev.tgt'
TGT_TEST_PATH = '/home/karen/workspace/nmt/webnlg/nmt_data/nmt_data/test.en'

SRC_VOCAB_PATH = '/home/karen/workspace/nmt/webnlg/nmt_data/nmt_data/vocab.vi'
TGT_VOCAB_PATH = '/home/karen/workspace/nmt/webnlg/nmt_data/nmt_data/vocab.en'

INFER_PATH = '/home/karen/workspace/nmt/webnlg/test_no_lex.vi'
CORPUS_PATH = '/home/karen/workspace/nmt/webnlg/text8'


def parser(xml_path, type='train'):
    if type == 'train':
        src = SRC_TRAIN_PATH
        dst = TGT_TRAIN_PATH
    elif type == 'dev':
        src = SRC_DEV_PATH
        dst = TGT_DEV_PATH
    elif type == 'test_no_lex':
        src = INFER_PATH
    # elif type == 'test_with_lex':
    #     src = SRC_TEST_PATH
    #     dst = TGT_TEST_PATH
    xmldoc = minidom.parse(xml_path)
    entry_lst = xmldoc.getElementsByTagName('entry')

    train_df = pd.read_csv('/home/karen/workspace/nmt/webnlg/dataset/challenge_data_train_dev/xml_count_train.csv')
    dev_df = pd.read_csv('/home/karen/workspace/nmt/webnlg/dataset/challenge_data_train_dev/xml_count_dev.csv')

    if type != 'test_no_lex':
        with open(src, 'a') as src_file, open(dst, 'a') as dst_file, open(SRC_TEST_PATH, 'a') as test_src, open(TGT_TEST_PATH, 'a') as test_tgt:
            cnt = 0
            for entry in entry_lst:
                triples_lst = entry.getElementsByTagName('mtriple')
                sentences_lst = entry.getElementsByTagName('lex')
                n_triples = len(triples_lst)
                n_sentences = len(sentences_lst)
                # print(n_triples, n_sentences)
                for n_sentence in range(n_sentences):
                    cnt += 1
                    sentence = sentences_lst[n_sentence].firstChild.nodeValue
                    sentence = sentence.replace('.', ' .').replace(',', ' ,').replace(';', ' ;')\
                            .replace(':', ' :').replace('\'', ' \'').replace('"', ' "')\
                            .replace('(', ' ( ').replace(')', ' ) ')
                    # print(sentence)
                    if type == 'train':
                        total = train_df[xml_path.split('/')[-1]]
                    else:
                        total = dev_df[xml_path.split('/')[-1]]
                    if cnt <= int(total*0.8):
                        dst_file.write(sentence + '\n')
                    else:
                        test_tgt.write(sentence + '\n')

                    triples = ''
                    for n_triple in range(n_triples):
                        triple = triples_lst[n_triple].firstChild.nodeValue
                        triple = triple.replace('  ', ' ').replace('\"', '').replace('(', ' ( ').replace(')', ' ) ')
                        if n_triple != n_triples - 1:
                            triples += triple + ' , '
                        else:
                            triples += triple
                    triples = triples.strip()
                    # print(triples)
                    if cnt <= int(total*0.8):
                        src_file.write(sentence + '\n')
                    else:
                        test_src.write(sentence + '\n')

    else:
        with open(src, 'a') as src_file:
            for entry in entry_lst:
                triples_lst = entry.getElementsByTagName('mtriple')
                n_triples = len(triples_lst)

                triples = ''
                for n_triple in range(n_triples):
                    triple = triples_lst[n_triple].firstChild.nodeValue
                    triple = triple.replace('  ', ' ').replace('\"', '').replace('(', ' ( ').replace(')', ' ) ')
                    if n_triple != n_triples - 1:
                        triples += triple + ' , '
                    else:
                        triples += triple
                triples = triples.strip()
                # print(triples)
                src_file.write(triples + '\n')
    return cnt


def pair_parse(type='train'):
    if type == 'train':
        path = TRAIN_PATH
    elif type == 'dev':
        path = DEV_PATH
    xml_count_dict = {}
    for i_triples in os.listdir(path):
        # print(i_triples)
        for triples_xml in os.listdir(os.path.join(path, i_triples)):
            # print(triples_xml)
            cnt = parser(os.path.join(os.path.join(path, i_triples), triples_xml),
                   type=type)
            xml_count_dict[triples_xml] = cnt
    print(xml_count_dict)
    xml_c_df = pd.DataFrame(xml_count_dict, index=[1])
    # xml_c_df.to_csv('xml_count_{}.csv'.format(type), sep=',', index=False)


def create_train_dev_test():
    try:
        os.remove(SRC_TRAIN_PATH)
        os.remove(TGT_TRAIN_PATH)
        os.remove(SRC_DEV_PATH)
        os.remove(TGT_DEV_PATH)
        os.remove(SRC_TEST_PATH)
        os.remove(TGT_TEST_PATH)
        os.remove(INFER_PATH)
    except Exception as e:
        print(e)
    pair_parse(type='train')
    pair_parse(type='dev')
    # parser(TEST_NO_LEX_PATH, type='test_no_lex')
    # parser(TEST_WITH_LEX_PATH, type='test_with_lex')


def create_vocab_helper(file, vocab):
    for line in file:
        for item in line.split(' '):
            item = item.strip()
            for word in item.split(' '):
                word = word.strip()
                if len(word) > 0 and 'http' not in word:
                        vocab[word] += 1
    return vocab


def create_vocab(train_file_path, dev_file_path, test_file_path, vocab_file_path):
    try:
        os.remove(vocab_file_path)
    except Exception as e:
        print(e)
    vocab = defaultdict(int)
    word_lst = []
    with open(train_file_path, 'r') as trainfile, \
            open(dev_file_path, 'r') as devfile, \
            open(test_file_path, 'r') as testfile, \
            open(vocab_file_path, 'a') as vocabfile:
        vocabfile.write('<unk>' + '\n')
        vocabfile.write('<s>' + '\n')
        vocabfile.write('</s>' + '\n')
        files = [trainfile, devfile, testfile]
        for file in files:
            vocab = create_vocab_helper(file, vocab)
            for word, count in vocab.items():
                # if vocab[word] >= 5 and word not in word_lst:
                #     word_lst.append(word)
                if word not in word_lst:
                    word_lst.append(word)
        word_lst = sorted(word_lst)
        for w in word_lst:
            vocabfile.write(w + '\n')


def create_corpus_helper(file, lst):
    for line in file:
        for word in line.split(' '):
            word = word.strip()
            if len(word) > 0:
                lst.append(word)
    return lst


def create_corpus_for_glove_training(train_path, dev_path, test_path, corpus_path):
    with open(train_path, 'r') as train, \
            open(dev_path, 'r') as dev, \
            open(test_path, 'r') as test, \
            open(corpus_path, 'a') as text8:
        files = [train, dev, test]
        lst = []
        for file in files:
            lst = create_corpus_helper(file, lst)
        text8.write(' '.join(lst))


if __name__ == '__main__':
    # create_train_dev_test()

    # create_vocab(train_file_path=SRC_TRAIN_PATH,
    #              dev_file_path=SRC_DEV_PATH,
    #              test_file_path=SRC_TEST_PATH,
    #              vocab_file_path=SRC_VOCAB_PATH)
    #
    # create_vocab(train_file_path=TGT_TRAIN_PATH,
    #              dev_file_path=TGT_DEV_PATH,
    #              test_file_path=TGT_TEST_PATH,
    #              vocab_file_path=TGT_VOCAB_PATH)

    # try:
    #     os.remove(CORPUS_PATH)
    # except Exception as e:
    #     print(e)
    #
    # create_corpus_for_glove_training(TGT_TRAIN_PATH,
    #                                  TGT_DEV_PATH,
    #                                  TGT_TEST_PATH,
    #                                  CORPUS_PATH)


