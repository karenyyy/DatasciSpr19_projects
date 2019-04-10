"""
A module for RDF Entity, RDF Property, and FactGraph.
"""

from utils import text_utils, rdf_utils, sparql_utils
import xml.etree.ElementTree as et
import re
import string
from collections import defaultdict

prop2schema = {}

with open('/home/karen/workspace/OpenNMT-py/rdf2text/metadata/prop_schema.list', encoding="utf8") as f:
    for line in f.readlines():
        (p, d, r) = line.strip().split()
        prop2schema[p] = (d, r)

entity2type = {}

with open('/home/karen/workspace/OpenNMT-py/rdf2text/metadata/entity_type.list', encoding="utf8") as f:
    for line in f.readlines():
        last_space_idx = line.rfind(' ')
        entity = line[:last_space_idx]
        stype = line[last_space_idx:]

        hash_idx = stype.find('#')
        stype = stype[hash_idx + 1:]

        entity2type[entity] = stype.strip()

class RDFEntity:

    def __init__(self, ID, o_rdf_entity, m_rdf_entity, semantic_type=None):

        self.ID = 'ENTITY_' + str(ID)

        self.olex_form = ' '.join(self.text_split(o_rdf_entity).split())
        self.mlex_form = ' '.join(self.text_split(m_rdf_entity).split())

        if semantic_type is None:
            self.stype = entity2type[o_rdf_entity]
        else:
            hash_idx = semantic_type.find('#')
            semantic_type = semantic_type[hash_idx + 1:]
            self.stype = semantic_type

        self.aliases = self.get_aliases()

    def text_split(self, entity_str):
        return ' '.join(entity_str.split('_'))

    def get_aliases(self):
        return [self.mlex_form, self.mlex_form.lower()]

    def set_stype(self, semantic_type):
        self.stype = semantic_type


class RDFProperty:
    def __init__(self, o_rdf_property, m_rdf_property):
        self.olex_form = self.text_split(o_rdf_property)
        self.mlex_form = self.text_split(m_rdf_property)
        self.type_form = o_rdf_property.upper()
        self.domain = prop2schema[o_rdf_property][0]
        self.range = prop2schema[o_rdf_property][1]

    def text_split(self, property_string):
        return text_utils.camel_case_split(property_string)


class FactGraph:

    def __init__(self, tripleset_tuple, lexicalization=None):

        otripleset, mtripleset = tripleset_tuple

        self.o_rdf_triples = otripleset[0].triples

        self.m_rdf_triples = mtripleset.triples

        assert len(self.o_rdf_triples) == len(self.m_rdf_triples), \
            "Original and modified tripleset are not the same length."

        if lexicalization:
            self.lexicalization = lexicalization

        self.entities = {}

        self.id2entity = {}

        self.properties = {}

        self.subj2obj = {}
        self.obj2subj = {}

        self._contruct_graph()

    def _contruct_graph(self):

        entityID = 0

        for (otriple, mtriple) in zip(self.o_rdf_triples, self.m_rdf_triples):

            o_subj = otriple.subject
            o_obj = otriple.object
            o_prop = otriple.property

            m_subj = mtriple.subject
            m_obj = mtriple.object
            m_prop = mtriple.property

            if o_prop not in self.properties:
                self.properties[o_prop] = RDFProperty(o_prop, m_prop)

            if o_subj not in self.entities:
                entityID += 1

                if self.properties[o_prop].domain != '*':
                    self.entities[o_subj] = RDFEntity(entityID, o_subj, m_subj,
                                                      self.properties[o_prop].domain)


                else:
                    self.entities[o_subj] = RDFEntity(entityID, o_subj, m_subj)

                self.id2entity[entityID] = self.entities[o_subj].mlex_form

            if o_obj not in self.entities:
                entityID += 1

                if self.properties[o_prop].range != '*':
                    self.entities[o_obj] = RDFEntity(entityID, o_obj, m_obj,
                                                     self.properties[o_prop].range)


                else:
                    self.entities[o_obj] = RDFEntity(entityID, o_obj, m_obj)

                    if self.entities[o_obj].stype == 'THING':
                        self.entities[o_obj].set_stype(self.properties[o_prop].type_form)

                self.id2entity[entityID] = self.entities[o_obj].mlex_form

            propFound = False

            if o_subj not in self.subj2obj:
                self.subj2obj[o_subj] = [(o_prop, [o_obj])]

            else:

                for i, (p, o) in enumerate(self.subj2obj[o_subj]):

                    if p == o_prop:
                        propFound = True

                        self.subj2obj[o_subj][i][1].append(o_obj)
                        break

                if not propFound:
                    self.subj2obj[o_subj].append((o_prop, [o_obj]))

            propFound = False

            if o_obj not in self.obj2subj:
                self.obj2subj[o_obj] = [(o_prop, [o_subj])]

            else:

                for i, (p, s) in enumerate(self.obj2subj[o_obj]):

                    if p == o_prop:
                        propFound = True
                        self.obj2subj[o_obj][i][1].append(o_subj)
                        break

                if not propFound:
                    self.obj2subj[o_obj].append((o_prop, [o_subj]))

    def delexicalize_text(self, advanced=False):

        no_match_list = []
        original_text = ' '.join(re.sub('\s+', ' ', self.lexicalization).split())
        delex_text = original_text

        for entity in self.entities.values():

            matchFound = False

            entity_str = entity.mlex_form.replace('"', '')
            print('entity_str: ', entity_str)

            if entity_str in self.lexicalization:
                delex_text = delex_text.replace(entity_str,
                                                ' ' + entity.ID + ' ')
                matchFound = True

            elif entity_str.lower() in self.lexicalization.lower():
                start_idx = delex_text.lower().find(entity_str.lower())
                end_idx = start_idx + len(entity_str)

                delex_text = delex_text[:start_idx] + ' ' \
                             + entity.ID + ' ' + delex_text[end_idx + 1:]
                matchFound = True

            elif entity_str.endswith(')'):
                left_idx = entity_str.find('(')

                entity_str = entity_str[:left_idx].strip()

                if entity_str in self.lexicalization:
                    delex_text = delex_text.replace(entity_str,
                                                    ' ' + entity.ID + ' ')
                    matchFound = True

            if matchFound or not advanced:
                continue

            if text_utils.is_date_format(entity_str):
                entity_ngrams = text_utils.find_ngrams(self.lexicalization)

                entity_ngrams = [text_utils.tokenize_and_concat(' '.join(ngram))
                                 for ngram in entity_ngrams]

                date_strings = [d_str for d_str in entity_ngrams
                                if text_utils.is_date_format(d_str)]

                date_strings.sort(key=len, reverse=True)
                print('data strings', date_strings, self.lexicalization)

                if date_strings:
                    best_match = date_strings[0]
                    delex_text = text_utils.tokenize_and_concat(delex_text)
                    delex_text = delex_text.replace(best_match, ' ' + entity.ID + ' ')

                    matchFound = True

            if len(text_utils.get_capitalized(entity_str)) > 1 and not matchFound:
                print('caps? entity str', entity_str)
                abbr_candidates = text_utils.generate_abbrs(entity_str)
                abbr_candidates.sort(key=len, reverse=True)
                print('abbr_candidates: ', abbr_candidates)

                text_unigrams = text_utils.find_ngrams(self.lexicalization, N=1)
                text_unigrams = [' '.join(unigram) for unigram in text_unigrams]

                print('abbr: ', abbr_candidates)
                print('text unigram: ', text_unigrams)

                for abbr in abbr_candidates:

                    nCaps = len([c for c in abbr if c.isupper()])

                    if abbr in text_unigrams and nCaps > 1:
                        print('before:', entity_str, abbr, delex_text)
                        delex_text = delex_text.replace(abbr, ' ' + entity.ID + ' ')
                        print('after:', entity_str, abbr, delex_text)
                        matchFound = True

            if not matchFound:
                delex_text = text_utils.tokenize_and_concat(delex_text)
                print('left behind: ', entity_str)
                best_match = text_utils.find_best_match(entity_str, delex_text)
                print('best match: ', best_match)
                if best_match:
                    delex_text = delex_text.replace(best_match,
                                                    ' ' + entity.ID + ' ')

                    matchFound = True

            if not matchFound:
                no_match_list.append((entity_str, self.lexicalization))

        final_delex = text_utils.tokenize_and_concat(delex_text)

        final_delex = final_delex if final_delex[-1] == '.' else final_delex + ' .'

        return final_delex

    def get_entityGraph(self):

        return self.subj2obj

    def linearize_graph(self, structured=False, incoming_edges=False):

        if not structured:
            seq = ''

            for triple in self.o_rdf_triples:
                subj = triple.subject
                obj = triple.object
                prop = triple.property

                seq = ' '.join(
                    [
                        seq,
                        self.entities[subj].ID,
                        self.entities[subj].stype,
                        self.properties[prop].mlex_form,
                        self.entities[obj].ID,
                        self.entities[obj].stype,
                    ]
                )
        else:

            if incoming_edges:
                entityGraph = self.obj2subj
            else:
                entityGraph = self.subj2obj

            seq = '('

            for (attr, val) in entityGraph.items():
                seq = ' '.join([seq, '('])
                seq = ' '.join(
                    [
                        seq,
                        self.entities[attr].ID,
                        self.entities[attr].stype
                    ]
                )

                for prop, obj_list in val:
                    seq = ' '.join([seq, '(', self.properties[prop].mlex_form])

                    for obj in obj_list:
                        seq = ' '.join(
                            [
                                seq,
                                '(',
                                self.entities[obj].ID,
                                self.entities[obj].stype,
                                ')'
                            ]
                        )
                    seq = ' '.join([seq, ')'])
                seq = ' '.join([seq, ')'])
            seq = ' '.join([seq, ')'])

        return seq.lstrip()
