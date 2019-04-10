from xml.dom import minidom
import os
import glob
from collections import defaultdict
import pandas as pd

from preprocess.lexicalize import lexicalize
from util.sparql_util import get_resource_type
from util.text_util import *


def single_xml_parser(xml_path,
                      src_train_path,
                      tgt_train_path,
                      src_dev_path,
                      tgt_dev_path,
                      src_test_path,
                      tgt_test_path,
                      relex_dev_path,
                      relex_test_path):

    xmldoc = minidom.parse(xml_path)
    entry_lst = xmldoc.getElementsByTagName('entry')
    total = len(entry_lst)
    print('total: ', total)
    cnt = 0
    with open(src_train_path, 'a+') as src_train_file, \
            open(tgt_train_path, 'a+') as tgt_train_file, \
            open(src_dev_path, 'a+') as src_dev_file, \
            open(tgt_dev_path, 'a+') as tgt_dev_file, \
            open(src_test_path, 'a+') as src_test_file, \
            open(tgt_test_path, 'a+') as tgt_test_file, \
            open(relex_dev_path, 'a+') as relex_dev_file, \
            open(relex_test_path, 'a+') as relex_test_file:
        for entry in entry_lst:
            category = xml_path.split('_')[-3]
            entity_id = 1
            entity_dict = {}
            cnt += 1
            triples_lst = entry.getElementsByTagName('mtriple')
            sentences_lst = entry.getElementsByTagName('lex')
            n_triples = len(triples_lst)
            n_sentences = len(sentences_lst)

            for n_sentence in range(n_sentences):
                sentence = sentences_lst[n_sentence].firstChild.nodeValue
                sentence = sentence.replace(',', ' ,') \
                    .replace(';', ' ;') \
                    .replace(':', ' :') \
                    .replace('\'', ' \'')
                if sentence[-1] == '.':
                    sentence = sentence[:-1] + ' .'

                triples = ''
                for n_triple in range(n_triples):
                    triple = triples_lst[n_triple].firstChild.nodeValue
                    subject, property, object = triple.split(' | ')
                    subject = subject.strip()
                    object = object.strip()
                    subject_type = get_resource_type(subject)
                    object_type = get_resource_type(object)
                    sub_entity_name = subject
                    obj_entity_name = object
                    entity_id, entity_dict, sub_entity_name = update_entity_dict(entity_type=subject_type,
                                                                                 entity=subject,
                                                                                 entity_name=sub_entity_name,
                                                                                 entity_dict=entity_dict,
                                                                                 entity_id=entity_id,
                                                                                 property=category,
                                                                                 sentence=sentence)

                    entity_id, entity_dict, obj_entity_name = update_entity_dict(entity_type=object_type,
                                                                                 entity=object,
                                                                                 entity_name=obj_entity_name,
                                                                                 entity_dict=entity_dict,
                                                                                 entity_id=entity_id,
                                                                                 property=property,
                                                                                 sentence=sentence)

                    new_property = split_property(property)

                    if len(subject) > len(object):
                        triple = triple.replace(subject, sub_entity_name).replace(object, obj_entity_name)
                    else:
                        triple = triple.replace(object, obj_entity_name).replace(subject, sub_entity_name)
                    triple = triple.replace(' | ', ' ').replace(property, new_property)

                    if n_triple != n_triples - 1:
                        triples += triple + ' '
                    else:
                        triples += triple

                triples = re.sub("\s\s+", " ", triples)
                triples = triples.replace('(', '( ') \
                    .replace(')', ' )')
                triples = triples.strip()

                sentence = lexicalize(entity_dict, sentence)

                sentence = re.sub("\s\s+", " ", sentence)
                sentence = sentence.replace('(', '( ') \
                    .replace(')', ' )')
                sentence = sentence.strip()
                print(triples, sentence)

                print('cnt: ', cnt)
                if 0 < cnt <= int(total * 0.8):
                    src_train_file.write(triples + '\n')
                    tgt_train_file.write(sentence + '\n')
                    print('Written to {} and {} ! \n'.format(src_train_path.split('/')[-1],
                                                             tgt_train_path.split('/')[-1]))

                elif int(total * 0.8) < cnt <= int(total * 0.9):
                    src_dev_file.write(triples + '\n')
                    tgt_dev_file.write(sentence + '\n')
                    sorted_keys = sorted(entity_dict, key=lambda x: x[1])
                    entity_dict = {k: entity_dict[k] for k in sorted_keys}
                    relex_dev_file.write(' '.join(entity_dict.keys()) + '\n')
                    print('Written to {} and {} ! \n'.format(src_dev_path.split('/')[-1],
                                                             tgt_dev_path.split('/')[-1]))
                    print('Written to {} ! \n'.format(relex_dev_path.split('/')[-1]))

                else:
                    src_test_file.write(triples + '\n')
                    tgt_test_file.write(sentence + '\n')
                    sorted_keys = sorted(entity_dict, key=lambda x: x[1])
                    entity_dict = {k: entity_dict[k] for k in sorted_keys}
                    relex_test_file.write(' '.join(entity_dict.keys()) + '\n')
                    print('Written to {} and {} ! \n'.format(src_test_path.split('/')[-1],
                                                             tgt_test_path.split('/')[-1]))
                    print('Written to {} ! \n'.format(relex_test_path.split('/')[-1]))


def parser(path,
           src_train_path,
           tgt_train_path,
           src_dev_path,
           tgt_dev_path,
           src_test_path,
           tgt_test_path,
           relex_dev_path,
           relex_test_path):
    path_list = sorted(os.listdir(path))
    print('triples file path: ', path_list)
    for i_triples in path_list:
        print(i_triples)
        triple_list = sorted(os.listdir(os.path.join(path, i_triples)))
        print('triples file list: ', triple_list)
        for triples_xml in triple_list:
            print('triples xml file: ', triples_xml)
            single_xml_parser(os.path.join(os.path.join(path, i_triples), triples_xml),
                              src_train_path,
                              tgt_train_path,
                              src_dev_path,
                              tgt_dev_path,
                              src_test_path,
                              tgt_test_path,
                              relex_dev_path,
                              relex_test_path)
