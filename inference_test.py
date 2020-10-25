# -*- coding: utf-8 -*-
"""
Python File Template 
"""

exec('from __future__ import unicode_literals')

import os
import sys
import random
import json

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
module_path = os.path.abspath(os.path.join('../onmt'))
if module_path not in sys.path:
    sys.path.append(module_path)

from itertools import repeat

from onmt.utils.logging import init_logger
from onmt.utils.misc import split_corpus
import onmt.translate.translator as translator

import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from kp_gen_eval import _get_parser


from nltk.corpus import stopwords
stoplist = stopwords.words('english')

from string import punctuation
import onmt.keyphrase.pke as pke
from nltk.corpus import stopwords

import onmt.keyphrase.kp_inference as kp_inference

__author__ = "Rui Meng"
__email__ = "rui.meng@pitt.edu"

if __name__ == '__main__':
    data_root_path = 'data/keyphrase/json/duc/duc_test.json'
    doc_dicts = []
    with open(data_root_path, 'r') as data_file:
        doc_dicts = [json.loads(l) for l in data_file]

    print('Loaded #(docs)=%d' % (len(doc_dicts)))

    doc_id = random.randint(0, len(doc_dicts))
    doc = doc_dicts[doc_id]
    print(doc.keys())
    text_to_extract = doc['abstract']
    print(doc_id)
    print(text_to_extract)

    parser = _get_parser()
    config_path = 'config/translate/config-rnn-keyphrase.yml'
    print(os.path.abspath('../config/translate/config-rnn-keyphrase.yml'))
    print(os.path.exists(config_path))
    # one2seq_ckpt_path = '/zfs1/pbrusilovsky/rum20/kp/OpenNMT-kpg/models/keyphrase/meng17-one2seq/meng17-one2seq-kp20k/kp20k-meng17-verbatim_append-rnn-BS64-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Reusetrue-Covtrue-PEfalse-Contboth-IF1_step_50000.pt'
    one2seq_ckpt_path = 'models/keyphrase/meng17-one2seq-kp20k-topmodels/kp20k-meng17-verbatim_append-rnn-BS64-LR0.05-Layer1-Dim150-Emb100-Dropout0.0-Copytrue-Reusetrue-Covtrue-PEfalse-Contboth-IF1_step_50000.pt'
    opt = parser.parse_args('-config %s' % (config_path))
    setattr(opt, 'models', [one2seq_ckpt_path])

    translator = translator.build_translator(opt, report_score=False)

    scores, predictions = translator.translate(
        src=[text_to_extract],
        tgt=None,
        src_dir=opt.src_dir,
        batch_size=opt.batch_size,
        attn_debug=opt.attn_debug,
        opt=opt
    )
    print('Paragraph:\n\t' + text_to_extract)
    print('Top predictions:')
    keyphrases = [kp.strip() for kp in predictions[0] if (not kp.lower().strip() in stoplist) and (kp != '<unk>')]
    for kp_id, kp in enumerate(keyphrases[: min(len(keyphrases), 20)]):
        print('\t%d: %s' % (kp_id + 1, kp))

