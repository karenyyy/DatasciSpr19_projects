from preprocess.parse_xml import *

SAVE_PATH = '/home/karen/workspace/cse597/nmt_preprocessing/nmt_data_1-7/'
TRIPLES_XML_PATH = '/home/karen/workspace/cse597/nmt_preprocessing/new_webnlg_release/webnlg-corpus-release'

UNSEEN_TEST_PATH = SAVE_PATH + 'testdata_unseen_with_lex.xml'

SRC_TRAIN_PATH = SAVE_PATH + 'train.vi'
SRC_DEV_PATH = SAVE_PATH + 'dev.vi'
SRC_TEST_PATH = SAVE_PATH + 'test.vi'

TGT_TRAIN_PATH = SAVE_PATH + 'train.en'
TGT_DEV_PATH = SAVE_PATH + 'dev.en'
TGT_TEST_PATH = SAVE_PATH + 'test.en'

SRC_VOCAB_PATH = SAVE_PATH + 'vocab.vi'
TGT_VOCAB_PATH = SAVE_PATH + 'vocab.en'

RELEX_DEV_PATH = SAVE_PATH + 'relex.dev'
RELEX_TEST_PATH = SAVE_PATH + 'relex.test'


parser(TRIPLES_XML_PATH,
       SRC_TRAIN_PATH,
       TGT_TRAIN_PATH,
       SRC_DEV_PATH,
       TGT_DEV_PATH,
       SRC_TEST_PATH,
       TGT_TEST_PATH,
       RELEX_DEV_PATH,
       RELEX_TEST_PATH)

