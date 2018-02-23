#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 17:28:35 2018

@author: Xinghao
"""

from read import *
from constants import *
from train import *
from helper import *
from model import *

import random


input_lang, output_lang, output_lang3, pairs = prepareData('input', 'output', 'output3',False)
print(random.choice(pairs))


encoder1 = EncoderRNN(input_lang.n_words, hidden_size)

attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1)
attn_decoder3 = AttnDecoderRNN(hidden_size, output_lang3.n_words, dropout_p = 0.1)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = attn_decoder1.cuda()
    att_decoder3 = attn_decoder3.cuda()
    
trainIters(encoder1, attn_decoder1, attn_decoder3, input_lang, output_lang, output_lang3, pairs, 105000, print_every=7000)

now = str(datetime.datetime.now())[:16].replace('-','_').replace(':', '_').replace(' ', '_')
save('./savedModel/three_sentences_' + now ,encoder1, attn_decoder1, attn_decoder3)

