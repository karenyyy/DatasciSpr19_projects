# -*- coding: utf-8 -*-
from xml.dom.minidom import parse
import xml.dom.minidom
import importlib, sys
import os, shutil
import random

def extract(inpath, outpathvi, outpathen):
    DOMTree = xml.dom.minidom.parse(inpath)
    root = DOMTree.documentElement
    entries = root.getElementsByTagName('entry')
    length = 0
    vi = open(outpathvi, 'a')
    en = open(outpathen, 'a')
    for entry in entries:
        modifieds = entry.getElementsByTagName('mtriple')
        lexs = entry.getElementsByTagName('lex')
        vivalue = ''
        for modified in modifieds:
            vivalue += modified.childNodes[0].data.strip()+' | '
        vivalue = vivalue[:len(vivalue)-3]
        vivalue = vivalue.replace('_',' _ ')
        vivalue = vivalue.replace('&',' & ')
        vivalue = vivalue.replace(',',' , ')
        vivalue = vivalue.replace('(',' ( ')
        vivalue = vivalue.replace(')',' ) ')
        vivalue = vivalue.replace('<',' < ')
        vivalue = vivalue.replace('>',' > ')
        vivalue = vivalue.replace('"',' " ')
        vivalue = vivalue.replace("'"," ' ")
        vivalue = vivalue.replace("'"," ' ")
        vivalue = vivalue.replace('/',' / ')
        vivalue = vivalue.replace(':',' : ')
        vivalue = ' '.join(vivalue.split())
        for lex in lexs:
            envalue = lex.childNodes[0].data
            # modify variable, add space between all special characters
            envalue = envalue.replace('_',' _ ')
            envalue = envalue.replace('&',' & ')
            envalue = envalue.replace(',',' , ')
            envalue = envalue.replace('(',' ( ')
            envalue = envalue.replace(')',' ) ')
            envalue = envalue.replace('<',' < ')
            envalue = envalue.replace('>',' > ')
            envalue = envalue.replace('"',' " ')
            envalue = envalue.replace("'"," ' ")
            envalue = envalue.replace('/',' / ')
            envalue = envalue.replace(':',' : ')
            for i in range(0, len(envalue)-2):
                if envalue[i]=='.' and envalue[i+1]==' ' and envalue[i+2].istitle():
                    envalue = envalue[:i-1]+' '+envalue[i:]
                if envalue[i]=="'":
                    if envalue[i+1]=='s' and envalue[i+2]==' ':
                        envalue = envalue[:i-1]+' '+envalue[i:]
                    else:
                        envalue = envalue[:i-1]+' '+envalue[i]+' '+envalue[i+1:]
            envalue = envalue[:-1]+' '+envalue[-1]
            envalue = ' '.join(envalue.split())
            length+=1
            vi.write(vivalue+'\n')
            en.write(envalue+'\n')
    vi.close()
    en.close()
    return length

def buildvocab():
    # build set for vi
    viset = set()
    dev_vi = open('dev.vi','r')
    for i in dev_vi.readlines():
        words=i.split()
        for w in words:
            viset.add(w)
    dev_vi.close()
    train_vi = open('train_test.vi','r')
    for i in train_vi.readlines():
        words=i.split()
        for w in words:
            viset.add(w)
    train_vi.close()
    vilist = []
    for vi in viset:
        vilist.append(vi)
    vilist.sort()
    vilist.insert(0,'<unk>')
    vilist.insert(1,'<s>')
    vilist.insert(2,'</s>')
    vocabvi = open('vocab.vi', 'w')
    for vi in vilist:
        vocabvi.write(vi+'\n')
    # build set for en
    enset = set()
    dev_en = open('dev.en','r')
    for i in dev_en.readlines():
        words=i.split()
        for w in words:
            enset.add(w)
    dev_en.close()
    train_en = open('train_test.en','r')
    for i in train_en.readlines():
        words=i.split()
        for w in words:
            enset.add(w)
    train_en.close()
    enlist = []
    for en in enset:
        enlist.append(en)
    enlist.sort()
    enlist.insert(0,'<unk>')
    enlist.insert(1,'<s>')
    enlist.insert(2,'</s>')
    vocaben = open('vocab.en', 'w')
    for en in enlist:
        vocaben.write(en+'\n')

def split_train_test(size, ratio):
    index = set()
    s = size//ratio
    while len(index) < s:
        index.add(random.randint(0,size-1))
    train_vi = open('train.vi', 'w')
    test_vi = open('test.vi', 'w')
    with open('train_test.vi','r') as fvi:
        for i, line in enumerate(fvi):
            if i in index:
                test_vi.write(line)
            else:
                train_vi.write(line)
    train_en = open('train.en', 'w')
    test_en = open('test.en', 'w')
    with open('train_test.en','r') as fen:
        for i, line in enumerate(fen):
            if i in index:
                test_en.write(line)
            else:
                train_en.write(line)


def main():
    importlib.reload(sys)
    count = 0
    work_dir = '../challenge_data_train_dev/dev'
    for parent, dirnames, filenames in os.walk(work_dir,  followlinks=True):
        for filename in filenames:
            file_path = os.path.join(parent, filename)
            if os.path.splitext(file_path)[1]==".xml":
                extract(file_path, 'dev.vi', 'dev.en')
                count+=1
    print('Handle %d dev files in total.' % count)
    count = 0
    size = 0
    work_dir = '../challenge_data_train_dev/train'
    for parent, dirnames, filenames in os.walk(work_dir,  followlinks=True):
        for filename in filenames:
            file_path = os.path.join(parent, filename)
            if os.path.splitext(file_path)[1]==".xml":
                size = size + extract(file_path, 'train_test.vi', 'train_test.en')
                count+=1
    print('Handle %d train files in total.' % count)
    # print(size)
    # extract('testdata_unseen_with_lex.xml', 'test.vi', 'test.en')
    buildvocab()
    split_train_test(size, 5)
    os.remove("train_test.en")
    os.remove("train_test.vi")


if __name__ == '__main__':
            main()
