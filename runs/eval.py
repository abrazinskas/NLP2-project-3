import tensorflow as tf
from models.neuralibm1_context import NeuralIBM1ContextModel
from vocabulary import OrderedCounter, Vocabulary
from utils import smart_reader, bitext_reader
import pickle

# the paths to our training and validation data, English side
train_e_path = 'data/training/hansards.36.2.e.gz'
train_f_path = 'data/training/hansards.36.2.f.gz'
dev_e_path = 'data/validation/dev.e.gz'
dev_f_path = 'data/validation/dev.f.gz'
dev_wa = 'data/validation/dev.wa.nonullalign'





# Using only 1000 words will result in many UNKs, but
# it will make training a lot faster.
# If you have a fast computer, a GPU, or a lot of time,
# try with 10000 instead.
max_tokens = 1000

corpus_e = smart_reader(train_e_path)
vocabulary_e = Vocabulary(corpus=corpus_e, max_tokens=max_tokens)
pickle.dump(vocabulary_e, open("vocabulary_e.pkl", mode="wb"))
print("English vocabulary size: {}".format(len(vocabulary_e)))

corpus_f = smart_reader(train_f_path)
vocabulary_f = Vocabulary(corpus=corpus_f, max_tokens=max_tokens)
pickle.dump(vocabulary_f, open("vocabulary_f.pkl", mode="wb"))
print("French vocabulary size: {}".format(len(vocabulary_f)))



dev_corpus = list(bitext_reader(
        smart_reader(dev_e_path),
        smart_reader(dev_f_path)))

# some hyper-parameters
# tweak them as you wish
batch_size = 25  # on CPU, use something much smaller e.g. 1-16
max_length = 20
lr = 0.0001
lr_decay = 0.0  # set to 0.0 when using Adam optimizer (default)
emb_dim = 64
mlp_dim = 128
num_epochs = 1
gated = True




# Add ops to save and restore all the variables.
# saver = tf.train.import_meta_graph('tmp/model.ckpt.meta')
with tf.Session() as sess:
    with tf.device("/cpu:0"):
        # our model
        model = NeuralIBM1ContextModel(
        x_vocabulary=vocabulary_e, y_vocabulary=vocabulary_f,
        batch_size=batch_size, emb_dim=emb_dim, mlp_dim=mlp_dim, gated=gated, session=sess)
        saver = tf.train.Saver()

        saver.restore(sess, tf.train.latest_checkpoint('tmp'))
        print(model.mlp_b.eval())

        # evaluate on development set
        val_aer, val_acc, val_loss = model.evaluate(dev_corpus, dev_wa, batch_size=batch_size)

        print("val_aer {:1.2f} val_acc {:1.2f} val_loss {:6f}".format(val_aer, val_acc, val_loss))