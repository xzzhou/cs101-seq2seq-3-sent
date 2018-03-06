#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 16:38:18 2018

@author: Xinghao
"""
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from constants import *
from helper import *
import datetime


######################################################################
# Training the Model
# ------------------
#
# To train we run the input sentence through the encoder, and keep track
# of every output and the latest hidden state. Then the decoder is given
# the ``<SOS>`` token as its first input, and the last hidden state of the
# encoder as its first hidden state.
#
# "Teacher forcing" is the concept of using the real target outputs as
# each next input, instead of using the decoder's guess as the next input.
# Using teacher forcing causes it to converge faster but `when the trained
# network is exploited, it may exhibit
# instability <http://minds.jacobs-university.de/sites/default/files/uploads/papers/ESNTutorialRev.pdf>`__.
#
# You can observe outputs of teacher-forced networks that read with
# coherent grammar but wander far from the correct translation -
# intuitively it has learned to represent the output grammar and can "pick
# up" the meaning once the teacher tells it the first few words, but it
# has not properly learned how to create the sentence from the translation
# in the first place.
#
# Because of the freedom PyTorch's autograd gives us, we can randomly
# choose to use teacher forcing or not with a simple if statement. Turn
# ``teacher_forcing_ratio`` up to use more of it.
#



def train(input_variable, target_variable, target_variable3, encoder, decoder, decoder3, encoder_optimizer, decoder_optimizer, decoder_optimizer3, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    #new
    decoder_optimizer3.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    #new
    target3_length = target_variable3.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    #new
    decoder_outputs = Variable(torch.zeros(max_length, decoder.hidden_size))
    decoder_outputs = decoder_outputs.cuda() if use_cuda else decoder_outputs

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    #new
    decoder3_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder3_input = decoder3_input.cuda() if use_cuda else decoder3_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_output_temp, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            #new construct
            decoder_outputs[di] = decoder_output_temp[0][0]
            
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing
        
        #the third sentence
        decoder3_hidden = decoder_hidden
        for mi in range(target3_length):
            decoder3_output, decoder3_output_temp, decoder3_hidden, decoder3_attention = decoder3(
                    decoder3_input, decoder3_hidden, decoder_outputs)
            loss += criterion(decoder3_output, target_variable3[mi])
            decoder3_input = target_variable3[mi]
        

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_output_temp, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            decoder_outputs[di] = decoder_output_temp[0][0]
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break
        
        #the third sentence
        decoder3_hidden = decoder_hidden
        for mi in range(target3_length):
            decoder3_output, decoder3_output_temp, decoder3_hidden, decoder3_attention = decoder3(
                    decoder3_input, decoder3_hidden, decoder_outputs)
            topv3, topi3 = decoder3_output.data.topk(1)
            ni3 = topi3[0][0]
            
            decoder3_input = Variable(torch.LongTensor([[ni3]]))
            decoder3_input = decoder3_input.cuda() if use_cuda else decoder3_input
            
            loss += criterion(decoder3_output, target_variable3[mi])
            if ni3 == EOS_token:
                break
            

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    decoder_optimizer3.step()

    return loss.data[0] / (target_length + target3_length)




######################################################################
# The whole training process looks like this:
#
# -  Start a timer
# -  Initialize optimizers and criterion
# -  Create set of training pairs
# -  Start empty losses array for plotting
#
# Then we call ``train`` many times and occasionally print the progress (%
# of examples, time so far, estimated time) and average loss.
#

def trainIters(encoder, decoder, decoder3, input_lang, output_lang, output_lang3, pairs, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    train_loss = []
    test_loss = []

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    decoder_optimizer3 = optim.SGD(decoder3.parameters(), lr=learning_rate)
    
    pairs_length = len(pairs)
    training_num = int(0.85 * pairs_length)
    random.shuffle(pairs)
    training_pairs = [variablesFromPair(input_lang, output_lang, output_lang3, \
                                        pairs[random.randrange(training_num)])
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]
        target_variable3 = training_pair[2]

        loss = train(input_variable, target_variable, target_variable3, encoder,
                     decoder, decoder3, encoder_optimizer, decoder_optimizer, decoder_optimizer3, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            train_loss.append(print_loss_avg)
            testing_error = testError(encoder, decoder, decoder3, input_lang, output_lang, output_lang3, pairs, criterion)
            test_loss.append(testing_error)
            print('%s (%d %d%%) train = %.4f test = %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg, testing_error))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    
    now = str(datetime.datetime.now())[:16].replace('-','_').replace(':', '_').replace(' ', '_')
    
    save_list('./loss_data/r3_' + now + '_train_loss.txt', train_loss)
    save_list('./loss_data/r3_' + now + '_test_loss.txt', test_loss)

    #showPlot(plot_losses)

#@return the test set error
def testError(encoder, decoder, decoder3, input_lang, output_lang, output_lang3, pairs, criterion, max_length = MAX_LENGTH):
    pairs_length = len(pairs)
    training_num = int(0.85 * pairs_length)
    testing_num = pairs_length - training_num
    testing_pairs = [variablesFromPair(input_lang, output_lang, output_lang3, \
                                       pairs[i]) for i in range(training_num, pairs_length)]
    total_loss = 0

    for i in range(len(testing_pairs)):
        encoder_hidden = encoder.initHidden()
        testing_pair = testing_pairs[i]
        input_variable = testing_pair[0]
        target_variable = testing_pair[1]
        target_variable3 = testing_pair[2]
        
        input_length = input_variable.size()[0]
        target_length = target_variable.size()[0]
        target3_length = target_variable3.size()[0]
        
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
        
        decoder3_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
        decoder3_input = decoder3_input.cuda() if use_cuda else decoder3_input
        
        decoder_hidden = encoder_hidden
        loss = 0
        
        for di in range(target_length):
            decoder_output, decoder_output_temp, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            decoder_outputs[di] = decoder_output_temp[0][0]
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break
        
        decoder3_hidden = decoder_hidden
        
        for mi in range(target3_length):
            decoder3_output, decoder3_output_temp, decoder3_hidden, decoder3_attention = decoder3(
                    decoder3_input, decoder3_hidden, decoder_outputs)
            topv3, topi3 = decoder3_output.data.topk(1)
            ni3 = topi3[0][0]
            
            decoder3_input = Variable(torch.LongTensor([[ni3]]))
            decoder3_input = decoder3_input.cuda() if use_cuda else decoder3_input
            
            loss += criterion(decoder3_output, target_variable3[mi])
            if ni3 == EOS_token:
                break
        total_loss += loss.data[0] / (target_length + target3_length)
    return total_loss / testing_num

