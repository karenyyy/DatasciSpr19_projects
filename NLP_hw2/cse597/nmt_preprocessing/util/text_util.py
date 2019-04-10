import difflib
import re
from nltk.tokenize import word_tokenize
from nltk import ngrams
from dateutil import parser as date_parser
from util.sparql_util import *

# hacks for some terms that are not recognized in sparql
COUNTRY_LIST = ['Denmark', 'Ireland', 'India', 'US', 'U.S.',
                'Australia', 'Poland', 'Pakistan', 'Angola',
                'Scotland', 'Yugoslavia', 'Uruguay', 'Canada',
                'Iraq', 'Greece', 'Turkmenistan']
STATE_LIST = ['Texas', 'California']
CITY_LIST = ['Madrid', 'Antwerp', 'Curitiba', 'Tirstrup', 'Alcobendas', 'Amsterdam']
ETHNICGROUP_LIST = ['Kashubians']


def is_date(text):
    try:
        date_parser.parse(text)
        return True
    except:
        return False


def is_float(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def split_property(property):
    """
    eg:
    elevationAboveTheSeaLevel -> elevation above the sea level
    """
    first_cap_re = re.compile('(.)([A-Z][a-z]+)')
    all_cap_re = re.compile('([a-z0-9])([A-Z])')
    s1 = first_cap_re.sub(r'\1 \2', property)
    return all_cap_re.sub(r'\1 \2', s1).lower().replace('_', ' ').replace('/', ' ')


def get_ngrams(text, N=None, char=False):
    Ngrams = []
    if not char:
        tokens = word_tokenize(text)
        if N is None:
            N = len(tokens)
        for n in range(1, N + 1):
            Ngrams.extend(ngrams(tokens, n))
    else:
        if N is None:
            N = len(text)
        for n in range(2, N + 1):
            Ngrams.extend(ngrams(text, n))

    return Ngrams


def num_cap_chars(text):
    return [w for w in word_tokenize(text) if w[0].isupper()]


def get_abbreviations(text):
    all_inits = [w[0] for w in text.split() if not w[0].isdigit()]
    capital_inits = [w[0].upper() for w in text.split() if not w[0].isdigit()]
    init_caps_only = [c for c in all_inits if c.isupper()]

    abbreviations = []

    for chars in (all_inits, capital_inits, init_caps_only):

        c_ngrams = [''.join(g) for g in get_ngrams(chars, char=True)] + \
            [' '.join(g) for g in get_ngrams(chars, char=True)] + \
            ['.'.join(g) + '.' for g in get_ngrams(chars, char=True)] + \
            ['.'.join(g) for g in get_ngrams(chars, char=True)] + \
            ['. '.join(g) + '.' for g in get_ngrams(chars, char=True)]
        abbreviations.extend(c_ngrams)
    return sorted(set(abbreviations), key=len, reverse=True)


def find_best_match(entity_str, text_lex):
    lex_ngrams = [' '.join(ngrams) for ngrams in get_ngrams(text_lex)]
    best_matches = difflib.get_close_matches(entity_str, lex_ngrams)
    return best_matches[0] if best_matches else None


def update_entity_dict(entity_type, entity, entity_name, entity_dict, entity_id, property, sentence):

    print('current entity: ', entity, entity_type)

    if property.endswith(')'):
        left_idx = property.find('(')
        right_idx = property.find(')')
        property = property[left_idx:right_idx].strip()
        if '_' in property:
            property = property.split('_')[-1]
        else:
            property = property.split(' ')[-1]

    updated = False

    if not updated:
        if entity in COUNTRY_LIST:
            postfix = 'COUNTRY'
            updated = True
        elif entity in STATE_LIST:
            postfix = 'STATE'
            updated = True
        elif entity in CITY_LIST:
            postfix = 'CITY'
            updated = True
        elif entity in ETHNICGROUP_LIST:
            postfix = 'ETHNICGROUP'
            updated = True
        else:
            postfix = ''

        if postfix != '':
            if entity not in entity_dict.keys():
                entity_name = 'ENTITY_{}[{}]'.format(entity_id, postfix)
                entity_dict[entity] = entity_name
                entity_id += 1
            else:
                entity_name = entity_dict[entity]

    if not updated:
        if entity_type != 'UNK':
            if entity not in entity_dict.keys():
                entity_name = 'ENTITY_{}[{}]'.format(entity_id, entity_type)
                entity_dict[entity] = entity_name
                entity_id += 1
            else:
                entity_name = entity_dict[entity]
            updated = True

    if not updated:
        if is_float(entity.replace('"', '')) or entity.replace('"', '').isdigit():
            if entity not in entity_dict.keys():
                entity_name = 'ENTITY_{}[{}]'.format(entity_id, property.upper())
                entity_dict[entity] = entity_name
                entity_id += 1
            else:
                entity_name = entity_dict[entity]
            updated = True

    if not updated:
        if is_date(entity.replace('"', '')):
            if entity not in entity_dict.keys():
                entity_name = 'ENTITY_{}[DATE]'.format(entity_id)
                entity_dict[entity] = entity_name
                entity_id += 1
            else:
                entity_name = entity_dict[entity]
            updated = True

    if not updated:
        if entity.endswith(')'):
            left_idx = entity.find('(')
            entity_no_par = entity[:left_idx].strip()
            entity_no_par = entity_no_par.replace('_', '')
            entity_no_par_type = get_resource_type(entity_no_par.replace('"', ''))
            if entity_no_par_type != 'UNK':
                if entity not in entity_dict.keys():
                    entity_name = 'ENTITY_{}[{}]'.format(entity_id, entity_no_par_type)
                    entity_dict[entity] = entity_name
                    entity_id += 1
                else:
                    entity_name = entity_dict[entity]
                updated = True

            if is_float(entity_no_par.replace('"', '')):
                if entity not in entity_dict.keys():
                    entity_name = 'ENTITY_{}[{}]'.format(entity_id, property.upper())
                    entity_dict[entity] = entity_name
                    entity_id += 1
                else:
                    entity_name = entity_dict[entity]
                updated = True

            if is_date(entity_no_par.replace('"', '')):
                if entity not in entity_dict.keys():
                    entity_name = 'ENTITY_{}[DATE]'.format(entity_id)
                    entity_dict[entity] = entity_name
                    entity_id += 1
                else:
                    entity_name = entity_dict[entity]
                updated = True

    if not updated:
        best_match = find_best_match(entity, sentence)
        if best_match and best_match not in entity:
            best_match_type = get_resource_type(best_match)
            if best_match_type and best_match_type != 'UNK':
                if entity not in entity_dict.keys():
                    entity_name = 'ENTITY_{}[{}]'.format(entity_id, best_match_type)
                    entity_dict[entity] = entity_name
                    entity_id += 1
                else:
                    entity_name = entity_dict[entity]
                updated = True

    if not updated:
        freq_word_list = []
        with open('/home/karen/workspace/cse597/nmt_preprocessing/en-2012/en.txt', 'r') as corpus:
            for line in corpus:
                freq_word_list.append(line.split(' ')[0])
        if entity.lower() not in freq_word_list:
            if entity not in entity_dict.keys():
                entity_name = 'ENTITY_{}[{}]'.format(entity_id, property.upper())
                entity_dict[entity] = entity_name
                entity_id += 1
            else:
                entity_name = entity_dict[entity]

    return entity_id, entity_dict, entity_name

