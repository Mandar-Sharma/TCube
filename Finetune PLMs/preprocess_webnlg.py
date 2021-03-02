#!/usr/bin/env python
# coding: utf-8


from xml.dom import minidom
from pathlib import Path
import re
import unidecode
import os
import pandas as pd

def clean_node(node):
    node = node.strip()
    node = node.replace('(', '')
    node = node.replace('\"', '')
    node = node.replace(')', '')
    node = node.replace(',', ' ')
    node = node.replace('_', ' ')
    node = unidecode.unidecode(node)
    return node

def clean_edge(edge):
    edge = edge.replace('(', '')
    edge = edge.replace(')', '')
    edge = edge.strip()
    edge = edge.split()
    edge = "_".join(edge)
    return edge

def camel_case_split(text):
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', text)
    d = [m.group(0) for m in matches]
    new_d = []
    for token in d:
        token = token.replace('(', '')
        token_split = token.split('_')
        for t in token_split:
            new_d.append(t.lower())
    return ' '.join(new_d)

def process_triples(mtriples):
    nodes = []

    for triple in mtriples:
        triple_text = triple.firstChild.nodeValue
        individual_triples = triple_text.strip().split(' | ')
        h = clean_node(individual_triples[0])
        r = camel_case_split(clean_edge(individual_triples[1]))
        t = clean_node(individual_triples[2])
        
        nodes.append('<H>')
        nodes.extend(h.split())

        nodes.append('<R>')
        nodes.extend(r.split())

        nodes.append('<T>')
        nodes.extend(t.split())

    return nodes

def get_data(file):
    return_data = []
    xmldoc = minidom.parse(file)
    entries = xmldoc.getElementsByTagName('entry')
    for e in entries:
        mtriples = e.getElementsByTagName('mtriple')
        nodes = ' '.join(process_triples(mtriples))
        lexs = e.getElementsByTagName('lex')
        for l in lexs:
            l = l.firstChild.nodeValue.strip()
            l = unidecode.unidecode(l)
            return_data.append((nodes,l))
    return return_data


files_train = Path(os.getcwd() + "/Datasets/WebNLGv3/train").rglob('*.xml')
files_train = sorted(list(files_train))

files_dev = Path(os.getcwd() + "/Datasets/WebNLGv3/dev").rglob('*.xml')
files_dev = sorted(list(files_dev))

files_test = [os.getcwd() + "/Datasets/WebNLGv3/test/rdf-to-text-generation-test-data-with-refs-en.xml"]

train_dict={"input_text":[], "target_text":[]}
dev_dict={"input_text":[], "target_text":[]}
test_dict={"input_text":[], "target_text":[]}

for filename in files_train:
    filename = str(filename)
    data = get_data(filename)
    for data_point in data:
        input_text = data_point[0]
        target_text = data_point[1]
        train_dict['input_text'].append(input_text)
        train_dict['target_text'].append(target_text)

for filename in files_dev:
    filename = str(filename)
    data = get_data(filename)
    for data_point in data:
        input_text = data_point[0]
        target_text = data_point[1]
        dev_dict['input_text'].append(input_text)
        dev_dict['target_text'].append(target_text)
        
for filename in files_test:
    filename = str(filename)
    data = get_data(filename)
    for data_point in data:
        input_text = data_point[0]
        target_text = data_point[1]
        test_dict['input_text'].append(input_text)
        test_dict['target_text'].append(target_text)
        
train = pd.DataFrame(train_dict)
dev = pd.DataFrame(dev_dict)
test = pd.DataFrame(test_dict)

train.to_csv('./Datasets/webnlg_train.csv', index=False)
dev.to_csv('./Datasets/webnlg_dev.csv', index=False)
test.to_csv('./Datasets/webnlg_test.csv', index=False)