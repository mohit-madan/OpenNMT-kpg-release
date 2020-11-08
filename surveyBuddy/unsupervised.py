exec('from __future__ import unicode_literals')

import os
import sys
import random
import csv
import copy
from tqdm import tqdm
module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join('../onmt'))
if module_path not in sys.path:
    sys.path.append(module_path)

from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
from onmt.translate.translator import build_translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from kp_gen_eval import _get_parser


from nltk.corpus import stopwords
stoplist = stopwords.words('english')

from string import punctuation
import onmt.keyphrase.pke as pke
from nltk.corpus import stopwords

import onmt.keyphrase.kp_inference as kp_inference

import importlib
importlib.reload(kp_inference)


def tfidf(text_to_extract, top_k):
    # tf-idf
    dataset_name = 'test'
    dataset_path = '../data/%s/' % dataset_name
    output = kp_inference.extract_pke(text_to_extract, method='tfidf' , dataset_path=dataset_path,
                df_path=os.path.abspath(dataset_path + '../%s.df.tsv.gz' % dataset_name), top_k=top_k)
    return list(zip(*output))


def yake(text_to_extract, top_k):
    # yake
    output = kp_inference.extract_pke(text_to_extract, method='yake', top_k=top_k)
    return list(zip(*output))


def text_rank(text_to_extract, top_percent, top_k):
    # text rank

    # define the set of valid Part-of-Speeches
    pos = {'NOUN', 'PROPN', 'ADJ'}

    # 1. create a TextRank extractor.
    extractor = pke.unsupervised.TextRank()

    # 2. load the content of the document.
    extractor.load_document(input=text_to_extract,
                            language='en_core_web_sm',
                            normalization=None)

    # 3. build the graph representation of the document and rank the words.
    #    Keyphrase candidates are composed from the 33-percent
    #    highest-ranked words.
    extractor.candidate_weighting(window=2,
                                  pos=pos,
                                  top_percent=top_percent)

    # 4. get the 10-highest scored candidates as keyphrases
    keyphrases = extractor.get_n_best(n=top_k)
    # for kp_id, kp in enumerate(keyphrases):
    #     print('\t%d: %s (%.4f)' % (kp_id+1, kp[0], kp[1]))
    return list(zip(*keyphrases))


def removeSubstrings(stringList):
    outputList = []
    for item in stringList:
        temp = copy.deepcopy(stringList)
        temp.remove(item)
        subString = "/".join(temp)
        if subString.find(item) == -1:
            outputList.append(item)
    return outputList


def main(top_k, top_percent):
    with open('Citi_ExcelCoding.csv', 'r') as read_file:
        with open('Citi_Unsupervised.csv', 'w', newline='') as write_file:
            csv_reader = csv.reader(read_file, delimiter='\t')
            csv_writer = csv.writer(write_file, )
            line_count = 0
            # tfidf_output = []
            # yake_output = []
            # text_rank_output = []
            for row in tqdm(csv_reader):
                if line_count != 0:
                    text_to_extract = row[0]
                    # target = row[1]
                    # tfidf_output = tfidf(text_to_extract, top_k)
                    # if (len(tfidf_output) > 0):
                        # tfidf_str = '/'.join(tfidf_output[0])
                        # tfidf_score = '/'.join([str(elem) for elem in tfidf_output[1]])
                    # else:
                        # tfidf_str = ""
                        # tfidf_score = ""

                    yake_output = yake(text_to_extract, top_k)

                    if len(yake_output) > 0:
                        yake_output_cleaned = removeSubstrings(list(yake_output[0]))
                        yake_str = '/'.join(yake_output[0])
                        yake_str_cleaned = '/'.join(yake_output_cleaned)
                        # yake_score = '/'.join([str(elem) for elem in yake_output[1]])
                    else:
                        yake_str = "None"
                        yake_str_cleaned = "None"
                        # yake_score = ""

                    text_rank_output = text_rank(text_to_extract, top_percent, top_k)
                    if len(text_rank_output) > 0:
                        text_rank_output_cleaned = removeSubstrings(list(text_rank_output[0]))
                        text_rank_str = '/'.join(text_rank_output[0])
                        text_rank_str_cleaned = '/'.join(text_rank_output_cleaned)
                        # text_rank_score = '/'.join([str(elem) for elem in text_rank_output[1]])
                    else:
                        text_rank_str = ""
                        text_rank_str_cleaned = ""
                        # text_rank_score = ""
                    csv_writer.writerow([text_to_extract, yake_str, yake_str_cleaned, text_rank_str, text_rank_str_cleaned])
                else:
                    csv_writer.writerow(["Input sentence", "YAKE Prediction", "YAKE Prediction Cleaned", "TEXT_RANK Prediction", "TEXT_RANK Prediction cleaned"])

                line_count += 1


if __name__ == "__main__":
    top_k = 5
    top_percent = 0.5
    main(top_k, top_percent)