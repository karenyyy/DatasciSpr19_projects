from dateutil import parser as date_parser
from util.text_util import *


def lexicalize(entity_dict, sentence):
    original_sentence = sentence
    for entity in entity_dict.keys():
        lex = False
        if entity in sentence:
            sentence = sentence.replace(entity, ' ' + entity_dict[entity] + ' ')
            lex = True

        elif entity.replace('"', '') in sentence:
            sentence = sentence.replace(entity.replace('"', ''), ' ' + entity_dict[entity] + ' ')
            lex = True

        elif ' '.join(entity.split('_')) in sentence:
            sentence = sentence.replace(' '.join(entity.split('_')), ' ' + entity_dict[entity] + ' ')
            lex = True

        elif entity.lower() in sentence.lower():
            start_idx = sentence.lower().find(entity.lower())
            end_idx = start_idx + len(entity)
            sentence = sentence[:start_idx] + ' ' + entity_dict[entity] + ' ' + sentence[end_idx + 1:]
            lex = True

        elif ' '.join(entity.split('_')).lower() in sentence.lower():
            new_entity = ' '.join(entity.split('_'))
            start_idx = sentence.lower().find(new_entity.lower())
            end_idx = start_idx + len(entity)
            sentence = sentence[:start_idx] + ' ' + entity_dict[entity] + ' ' + sentence[end_idx + 1:]
            lex = True

        if is_date(entity.replace('"', '')):
            entity_ngrams = get_ngrams(original_sentence)

            entity_ngrams = [' '.join(word_tokenize(' '.join(ngram)))
                             for ngram in entity_ngrams]

            dates = [entity_ngram for entity_ngram in entity_ngrams
                     if is_date(entity_ngram)]

            dates.sort(key=len, reverse=True)
            # print('dates: ', dates)

            if dates:
                best_match = dates[0]
                sentence = sentence.replace(best_match, ' ' + entity_dict[entity] + ' ')
                lex = True

        if not lex:
            print('not lex: ', entity)
            best_match = find_best_match(entity, sentence)
            print('best match', best_match, entity_dict[entity])
            if best_match:
                sentence = sentence.replace(best_match, ' ' + entity_dict[entity] + ' ')
                lex = True

        if entity.endswith(')') and not lex:
            left_idx = entity.find('(')
            entity = entity[:left_idx].strip()

            if len(num_cap_chars(entity)) > 1:
                candidates = get_abbreviations(entity)

                candidates.sort(key=len, reverse=True)
                print('abbr_candidates: ', candidates)

                text_unigrams = get_ngrams(original_sentence, N=1)
                text_unigrams = [' '.join(unigram) for unigram in text_unigrams]

                print('abbr: ', candidates)
                print('text unigram: ', text_unigrams)

                for abbreviation in candidates:

                    n_caps = len([c for c in abbreviation if c.isupper()])

                    if abbreviation in text_unigrams and n_caps > 1:
                        sentence = sentence.replace(abbreviation, ' ' + entity.ID + ' ')
    return sentence

