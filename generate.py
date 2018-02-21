#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 15:25:34 2018

@author: Xinghao
"""

from read import *
from helper import *
import torch
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random

input_lang, output_lang, output_lang3, pairs = prepareData('input', 'output', 'output3',False)

filename = './savedModel/three_sentences_0220'
encoder1 = torch.load(filename + '_encoder1.pth')
attn_decoder1 = torch.load(filename + '_attn_decoder1.pth')
attn_decoder3 = torch.load(filename + '_attn_decoder3.pth')
print('load successfully')


######################################################################
# Evaluation
# ==========
#
# Evaluation is mostly the same as training, but there are no targets so
# we simply feed the decoder's predictions back to itself for each step.
# Every time it predicts a word we add it to the output string, and if it
# predicts the EOS token we stop there. We also store the decoder's
# attention outputs for display later.
#

def evaluate(encoder, decoder, decoder3, sentence, max_length=MAX_LENGTH):
    input_variable = variableFromSentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    #new
    decoder_outputs = Variable(torch.zeros(max_length, decoder.hidden_size))
    decoder_outputs = decoder_outputs.cuda() if use_cuda else decoder_outputs
    

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    #new
    decoder3_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    decoder3_input.cuda() if use_cuda else decoder3_input

    decoder_hidden = encoder_hidden

    decoded_words = []
    #new
    decoded3_words = []
    
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_output_temp, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs)
        #new
        decoder_outputs[di] = decoder_outputs[di] + decoder_output_temp[0][0]
        
        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    #new
    decoder3_hidden = decoder_hidden
    for mi in range(max_length):
        decoder3_output, decoder3_output_temp, decoder3_hidden, decoder3_attention = decoder3(
                decoder3_input, decoder3_hidden, decoder_outputs)
        topv3, topi3 = decoder3_output.data.topk(1)
        ni3 = topi3[0][0]
        if ni3 == EOS_token:
            decoded3_words.append('<EOS>')
            break
        else:
            decoded3_words.append(output_lang3.index2word[ni3])
        
        decoder3_input = Variable(torch.LongTensor([[ni3]]))
        decoder3_input = decoder3_input.cuda() if use_cuda else decoder3_input
        

    return decoded_words, decoded3_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, decoder3, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1], pair[2])
        output_words, output_words3, attentions = evaluate(encoder, decoder, decoder3, pair[0])
        output_sentence = ' '.join(output_words)
        output_sentence3 = ' '.join(output_words3)
        print('<', output_sentence, output_sentence3)
        print('')


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, output_words3, attentions = evaluate(
        encoder1, attn_decoder1, attn_decoder3, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words), ' '.join(output_words3))
    #showAttention(input_sentence, output_words, attentions)



    
evaluateRandomly(encoder1, attn_decoder1, attn_decoder3)
output_words, output_words3, attentions = evaluate(
    encoder1, attn_decoder1, attn_decoder3, "i hope so .")


evaluateAndShowAttention("I want to go with you !".lower())

evaluateAndShowAttention("What are you doing ?".lower())

evaluateAndShowAttention("who knows ?")

evaluateAndShowAttention("Relax .".lower())

evaluateAndShowAttention("Fuck you !".lower())

evaluateAndShowAttention("Do you love me ?".lower())

evaluateAndShowAttention("When can we marry ?".lower())