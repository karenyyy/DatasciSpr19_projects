"""
This module contains some useful classes and functions to deal with RDF data and
read and process XML files.
"""

from nltk.tokenize import word_tokenize
import xml.etree.ElementTree as et
from collections import defaultdict
import argparse
import os


class Triple:

    def __init__(self, s, p, o):
        self.subject = s
        self.object = o
        self.property = p


class Tripleset:

    def __init__(self):
        self.triples = []

    def fill_tripleset(self, t):
        for xml_triple in t:
            s, p, o = xml_triple.text.split(' | ')

            triple = Triple(s, p, o)
            self.triples.append(triple)


class Lexicalisation:

    def __init__(self, lex, comment, lid):
        self.lex = lex
        self.comment = comment
        self.id = lid


class Entry:

    def __init__(self, category, size, eid):
        self.originaltripleset = []
        self.modifiedtripleset = Tripleset()
        self.lexs = []
        self.category = category
        self.size = size
        self.id = eid

    def fill_originaltriple(self, xml_t):
        otripleset = Tripleset()
        self.originaltripleset.append(otripleset)
        otripleset.fill_tripleset(xml_t)

    def fill_modifiedtriple(self, xml_t):
        self.modifiedtripleset.fill_tripleset(xml_t)

    def create_lex(self, xml_lex):
        comment = xml_lex.attrib['comment']
        lid = xml_lex.attrib['lid']
        lex = Lexicalisation(xml_lex.text, comment, lid)
        self.lexs.append(lex)

    def count_lexs(self):
        return len(self.lexs)


class RDFInstance(object):

    def __init__(self, category, size, otripleset, mtripleset, lex=None):

        self.category = category
        self.size = size
        self.originaltripleset = otripleset
        self.modifiedtripleset = mtripleset

        if lex:
            self.Lexicalisation = lex

        self.entities = set()
        self.properties = set()

        self._populate_sets()

    def _populate_sets(self):

        for tset in self.originaltripleset:
            for triple in tset.triples:
                s = triple.subject
                p = triple.property
                o = triple.object

                self.entities.update((s, o))
                self.properties.add(p)


def parseXML(xml_file):
    entries = []

    tree = et.parse(xml_file)
    root = tree.getroot()

    for xml_entry in root.iter('entry'):

        tags = [c.tag for c in xml_entry]

        if "lex" not in tags:
            continue

        entry_id = xml_entry.attrib['eid']
        category = xml_entry.attrib['category']
        size = xml_entry.attrib['size']

        entry = Entry(category, size, entry_id)

        for element in xml_entry:
            if element.tag == 'originaltripleset':
                entry.fill_originaltriple(element)
            elif element.tag == 'modifiedtripleset':
                entry.fill_modifiedtriple(element)
            elif element.tag == 'lex':
                entry.create_lex(element)

        print('entry: ', entry.modifiedtripleset.triples[0].subject, entry.modifiedtripleset.triples[0].property, entry.modifiedtripleset.triples[0].object)
        entries.append(entry)

    return entries


def generate_instances(dir, extended=False, eval=False):
    subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]  # triples

    instances = defaultdict(list)

    global_entities = set()
    global_properties = set()

    for d in sorted(subfolders):
        xml_files = [f for f in os.listdir(d) if f.endswith('.xml')]

        for f in xml_files:
            entries = parseXML(d + '/' + f)

            if eval:
                for entry in entries:
                    rdfInstance = RDFInstance(entry.category,
                                              entry.size,
                                              entry.originaltripleset,
                                              entry.modifiedtripleset,
                                              entry.lexs)

                    instances[entry.size].append(rdfInstance)


            else:
                for entry in entries:
                    for lex in entry.lexs:
                        rdfInstance = RDFInstance(entry.category,
                                                  entry.size,
                                                  entry.originaltripleset,
                                                  entry.modifiedtripleset,
                                                  lex)

                        instances[entry.size].append(rdfInstance)

    return instances, global_entities, global_properties
