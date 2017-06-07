import tensorflow as tf
import numpy as np
from pprint import pprint
import pickle
from utils import smart_reader, bitext_reader
# check vocabulary.py to see how the Vocabulary class is defined
from vocabulary import OrderedCounter, Vocabulary
from utils import iterate_minibatches, prepare_data
# check neuralibm1.py for the Model code
from models.neuralibm1 import NeuralIBM1Model
# check neuralibm1trainer.py for the Trainer code
from trainers.neuralibm1trainer import NeuralIBM1Trainer



# the paths to our training and validation data, English side
train_e_path = '../data/training/hansards.36.2.e.gz'
train_f_path = '../data/training/hansards.36.2.f.gz'
dev_e_path = '../data/validation/dev.e.gz'
dev_f_path = '../data/validation/dev.f.gz'
dev_wa = '../data/validation/dev.wa.nonullalign'


# Using only 1000 words will result in many UNKs, but
# it will make training a lot faster.
# If you have a fast computer, a GPU, or a lot of time,
# try with 10000 instead.
max_tokens=1000

corpus_e = smart_reader(train_e_path)
vocabulary_e = Vocabulary(corpus=corpus_e, max_tokens=max_tokens)
pickle.dump(vocabulary_e, open("vocabulary_e.pkl", mode="wb"))
print("English vocabulary size: {}".format(len(vocabulary_e)))

corpus_f = smart_reader(train_f_path)
vocabulary_f = Vocabulary(corpus=corpus_f, max_tokens=max_tokens)
pickle.dump(vocabulary_f, open("vocabulary_f.pkl", mode="wb"))
print("French vocabulary size: {}".format(len(vocabulary_f)))
print()

# run
tf.reset_default_graph()

with tf.Session() as sess:

  # some hyper-parameters
  # tweak them as you wish
  batch_size= 25  # on CPU, use something much smaller e.g. 1-16
  max_length= 20
  lr = 0.0005
  lr_decay = 0.0  # set to 0.0 when using Adam optimizer (default)
  emb_dim = 64
  mlp_dim = 128
  num_epochs = 3

  # our model
  model = NeuralIBM1Model(
    x_vocabulary=vocabulary_e, y_vocabulary=vocabulary_f,
    batch_size=batch_size, emb_dim=emb_dim, mlp_dim=mlp_dim, session=sess)

  # our trainer
  trainer = NeuralIBM1Trainer(
    model, train_e_path, train_f_path,
    dev_e_path, dev_f_path, dev_wa,
    num_epochs=num_epochs, batch_size=batch_size,
    max_length=max_length, lr=lr, lr_decay=lr_decay, session=sess)

  # now first TF needs to initialize all the variables
  print("Initializing variables..")
  sess.run(tf.global_variables_initializer())

  # now we can start training!
  print("Training started..")
  trainer.train()
