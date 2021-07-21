# Building a ChatBot using Deep NLP

import numpy as np
import tensorflow as tf
import re
import time

#### Part 1 - Data Processing ####

# Importing dataset:
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# Creating dictionary that maps each line's id to its text:
id2line = {}    #This dictinary maps each line (key) with its text (value)
for line in lines:
    temp_line = line.split(' +++$+++ ')
    if len(temp_line) == 5:
        id2line[temp_line[0]] = temp_line[4]    # 0 index is id, 4 index is the text
        
# Creating list of all conversations
conversations_ids = []    #This list will hold all the line-ids of conversations as sub-lists
for conversation in conversations[:-1]: # last row is empty
    temp_conversation = conversation.split(' +++$+++ ')[-1] # last part of each row is the part i need
    # Removing the square brackets, quatation marks and spaces from ids:
    temp_conversation = temp_conversation[1:-1] 
    temp_conversation = temp_conversation.replace("'", "")
    temp_conversation = temp_conversation.replace(" ", "")
    # Adding temp_conversation as a sub-list to conversations_ids
    conversations_ids.append(temp_conversation.split(","))

# Separate between question lines and answer lines (which will translate to inputs and outputs of the rnn):
# Answers[i] will be the answer for the qustion at questions[i]
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation)-1):
        question = id2line[conversation[i]]
        answer = id2line[conversation[i+1]]
        questions.append(question)
        answers.append(answer)

# First cleaning of texts (lowercase, it's -> it is, etc.)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"didn't", "did not", text)
    text = re.sub(r"don't", "do not", text)
    text = re.sub(r"wasn't", "was not", text)
    text = re.sub(r"weren't", "were not", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"there's", "there is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", "will", text)
    text = re.sub(r"\'ve", "have", text)
    text = re.sub(r"\'re", "are", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "can not", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

# Cleaning questions:
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

# Creating dictionary that maps each word to its number of occurances, in order to remove the words that appear less than 5% of corpus:
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
        
# Creating 2 dictionaries: 
# One that maps each word in questions to a unique integer
# Second that maps each word in answers to a unique integer
threshold = 20  # ~5% of corpus
questWord2int = {}
ansWord2int = {}
word_number = 0
for word, count in word2count.items():
    if count > threshold:
        questWord2int[word] = word_number
        word_number += 1
word_number = 0
for word, count in word2count.items():
    if count > threshold:
        ansWord2int[word] = word_number
        word_number += 1

# Adding special tokens to questWord2int and to ansWord2int:
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']   # <out> will replace all the less frequent words that were filtered before
for token in tokens:
    questWord2int[token] = len(questWord2int) + 1
for token in tokens:
    ansWord2int[token] = len(ansWord2int) + 1

# Creating inverse dictionary to the ansWord2int dictionary - will be needed for the seq2seq model:
ansInt2word = {w_i : w for w, w_i in ansWord2int.items()}

# Add EOS token to the ending of every answer - EOS token is needed at the end of the decoding layers of the seq2seq model:
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Translate all questions and answers to ints, replace all words that were filtered out by <OUT>
questions2int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questWord2int:
            ints.append(questWord2int['<OUT>'])
        else:
            ints.append(questWord2int[word])
    questions2int.append(ints)
answers2int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in ansWord2int:
            ints.append(ansWord2int['<OUT>'])
        else:
            ints.append(ansWord2int[word])
    answers2int.append(ints)

# Sort questions and answers by length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 26): # 1 is shortest question (like "what"), not including questions of lenth bigger than to 25.
    for i in enumerate(questions2int):  # i[0] is the index of the question, i[1] is the list of tokens representing the text of the question
        if len(i[1]) == length:
            sorted_clean_questions.append(questions2int[i[0]])
            sorted_clean_answers.append(answers2int[i[0]])

#### Part 2 - Building the Seq2Seq model ####

# Creating placeholders for inputs and targets:
# *in tensorflow all variables are arranged in tensors (an advanced array) - like numpy arrays in numpy
# **placeholders are even more advanced data structure, containing tensors and other features
def model_inputs(): # convert inputs and targets to placeholders
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    # *learning rate is a hyperparameter that controls how much to change the model in response to the estimated error each time the model weights are updated
    # ** too big lr = learning a sub-optimal set of weights, to small lr = long training process
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')  # will be used to control dropout rate
    # *dropout is the rate of the neurons I choose to overwrite/ignore during ont iterations in the training.
    # **A dropout on the input means that for a given probability, the data on the input connection to each LSTM block will be excluded from node activation and weight updates.
    return inputs, targets, lr, keep_prob

# Preprocessing the targets - creating the batches and add SOS token at start:
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>']) # Vector of SOS token of length (batch_size)
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])     # Matrix (batch_size rows) of all targets without last token (eos)
    preprocessed_targets = tf.concat([left_side, right_side], 1)    # the batch will be the concatination of the vector and the matrix
    return preprocessed_targets

# Create the Encoder RNN:
# *rnn size = number of input tensors of the encoder
# **sequence_length = length of each question in batch
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    # now to wrap this lstm to have a lstm with droupout applied - so that for a given probability, the data on the input connection to each LSTM block will be excluded from node activation and weight updates.
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers) # encoder cell composed of several lstm layers
    # in the following line, only need encoder_state:
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell, 
                                                       cell_bw = encoder_cell, sequence_length = sequence_length, 
                                                       inputs = rnn_inputs, dtype = tf.float32)
    # bidirectional rnn process the word sequence in both directions - in fact 2 separate rnns are used, fw and bw
    return encoder_state

# Create the Decoder RNN in three steps:
# 1- Decoding the training set (for the observations of the training set) - this function will decode the observations of the training set and return the output of the decoder:
# *embedding is a mapping from words to vectors of real numbers. each word is a vector.
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size]) # 3-dimentional matrix of 0s, batch_size lines, one column and depth of output size
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    # attention keys = the keys to be compared with the target states
    # attention values= values i will use to construct the context vectors (context is returned by the encoder and used by the decoder)
    # attention_score_function is used to compute the similarity between keys and target states
    # training_decoder_function is a function for the training of the decoder:
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys, 
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    # only need decoder_output:
    decoder_output, decoder_final_state, secoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    # applying dropout to the decoder output with the keep_prob parameter, then returning it on the right format with output_function:
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# 2- Decoding the test/validation set (will not be used for training) - more or less like the previous function but for the validations set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size]) # 3-dimentional matrix of 0s
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = 'bahdanau', num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys, 
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix, 
                                                                              sos_id, 
                                                                              eos_id, 
                                                                              maximum_length, 
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    # only need test_predictions:
    test_predictions, decoder_final_state, secoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell, 
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions

# 3- Creating the Decoder RNN - this function will create a fully connected layer which will be the last layer of the rnn (the layer of the decoder):
# *encoder_state is the output of the encoder, which is now used as input for the decoder
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    # decoding_scope is an advanced tf object that wraps tensors
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob) # similar to what i ded with encoder
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers) # like the encoder, i want several layers
        weights = tf.truncated_normal_initializer(stddev = 0.1) #initialize weights that will be associated to the neurons of the fully connected layer inside the decoder (last layer)
        biases = tf.zeros_initializer() #initialize biases as zeros
        # output function will return fully connected layer layers that will be used for the last layer of the rnn (the fully connected layer comes at the end of the rnn, after all lstms)
        output_function = lambda x: tf.contrib.layers.fully_connected(x, # x is the inputs
                                                                      num_words, #number of outputs
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, word2int['<SOS>'],
                                           word2int['<EOS>'], sequence_length-1, num_words, decoding_scope, output_function,
                                           keep_prob, batch_size)
    return training_predictions, test_predictions

# Building the seq2seq model - assemble encoder and decoder, returning training and test predictions
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs, answers_num_words+1, encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words+1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, 
                                                         encoder_state, questions_num_words, sequence_length,
                                                         rnn_size, num_layers, questionswords2int, keep_prob, batch_size) 
    return training_predictions, test_predictions
    
#### Part 3 - Training the Seq2Seq model ####
    
# Setting the Hyperparameters:
epochs = 100    # number of times the seq2seq model will pass over all the observations (fw and bw propagating over all batches)
batch_size = 64     # after each batch the weights will be updated
rnn_size = 512  
num_layers = 3  # how many lines in encoder & decoder networks
encoding_embedding_size = 512   #column size for embedddings matrix (each line is a token in corpus)
decoding_embedding_size = 512
learning_rate = 0.01    # to high = learning not optimized, too low = too much time
learning_rate_decay = 0.9   # by which percentage the learning rate is reduced over iterations. decaying the learning rate helps the network converge to a local minimum and avoid oscillation. 1 = no decay
min_learning_rate = 0.0001  # preventing lr to be too low, from the decay.
keep_probability = 0.5  #prevent overfitting by dropout. this value optimizes it.

# Defining a session:
tf.reset_default_graph()    # reseting graph first
session = tf.InteractiveSession()

# Loading the model inputs:
inputs, targets, lr, keep_prob = model_inputs()

# Setting the sequence length:
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')   # meaning in the training we won't use questions and answers that are longer than 25 words

# Getting the shape of the inputs tensor - will be argument for 'ones' function:
input_shape = tf.shape(inputs)

# Getting the training and test predictions:
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]), targets, keep_prob, batch_size,
                                                       sequence_length, len(ansWord2int), len(questWord2int),
                                                       encoding_embedding_size, decoding_embedding_size, rnn_size, 
                                                       num_layers, questWord2int)

# Setting up the Loss Error, the Optimizer and Gradient Clipping:
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions, targets, 
                                                  tf.ones([input_shape[0], sequence_length])) # representing the weights
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error) # compute gradients of the last error with respect to the weights of each neuron
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None] # tf function meant to clip gradients in order to avoid vanishing or exploding gradient issues 
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

# Padding the sequence with the <PAD> token:
# Question: [ 'who', 'are', 'you' ]                       ---> [ 'who', 'are', 'you', <PAD>, <PAD>, <PAD>, <PAD> ] 
# Answer: [ <SOS>, 'I', 'am', 'a', 'bot', '.', <EOS> ]    ---> [ <SOS>, 'I', 'am', 'a', 'bot', '.', <EOS>, <PAD> ]
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]

# Splitting the data into batches of questions and answers:
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(len(questions) // batch_size):
        start_index = batch_index * batch_size
        questions_in_batch = questions[start_index : start_index + batch_size]
        answers_in_batch = answers[start_index : start_index + batch_size]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questWord2int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, ansWord2int))
        yield padded_questions_in_batch, padded_answers_in_batch

# Splitting the questions and answers into training and validation sets:
training_validation_split = int(len(sorted_clean_questions) * 0.15)
training_questions = sorted_clean_questions[training_validation_split:]
training_answers = sorted_clean_answers[training_validation_split:]
validation_questions = sorted_clean_questions[:training_validation_split]
validation_answers = sorted_clean_answers[:training_validation_split]

# Training the model:
batch_index_check_training_loss = 100 # meaning every 100 batches the training loss will be checked
batch_index_check_validation_loss = ((len(training_questions)) // batch_size // 2) - 1 # Check in the middle
total_training_loss_error = 0   # sum of training losses on 100 batches
list_validation_loss_error = []    # list of loss errors in validation. will include all the losses we got
early_stopping_check = 0    # number of times the validation loss wasnt reduced. once it reaches some point (early_stopping_stop), the training should be stopped.
early_stopping_stop = 1000
checkpoint = "chatbot_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:  #every 100 batches
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:   
                learning_rate = min_learning_rate   #lr is always minimal
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):    # validation error was lower then all previous ones collected so far
                print("I speak better now.")
                early_stopping_check = 0    # reset to 0, because its incremented only when there is no improvement
                saver = tf.train.saver() # save the model
                saver.save(session, checkpoint)
            else:
                print("Sorry, I do not speak better, I need to practice more.")
                early_stopping_check += 1   # because there was no improvment
                if early_stopping_check == early_stopping_stop: # reached 1000 times with no improvement? if so, stop the loop
                    break
    if early_stopping_check == early_stopping_stop:
        print("My apologies, I cannot speak better anymore - This is the best I can do.")
        break
print("Game Over")


                
#### Part 4 - Training the Seq2Seq model ####

# Loading the weights and running the session:
checkpoint = "./chatbot_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(session, checkpoint)

# Converting the questions from strings to lists of encoding integers:
def convert_string2int(question, word2int):
    question = clean_text(question)
    return [word2int.get(word, word2int['<OUT>']) for word in question.split()] # return the int representing the word if it is in dictionary, elae <out>

# Setting up the chat:
while True:
    question = input("You: ")
    if question == "Goodbye":
        break
    question = convert_string2int(question, questWord2int) # translate to integers
    question = question + [questWord2int['<PAD>']] * (threshold - len(question)) # apply padding - the questions that were use foe training have length of 20
    # the rnn can only read questions in batches, so making a fake_batch for the question:
    fake_batch = np.zeros((batch_size, threshold))
    fake_batch[0] = question
    predicted_answer = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    answer = ""
    for i in np.argmax(predicted_answer, 1):    # i will get the values of the different token ids in the predicted answer
        if ansInt2word[i] == 'i':
            token = 'I'
        elif ansInt2word[i] == '<EOS>':
            token = '.'
        elif ansInt2word[i] == '<OUT>':
            token = 'out'
        else:
            token = ' ' + ansInt2word[i]
        answer += token
        if token == '.':
            break
    print("ChatBot: " + answer)