import numpy as np
import tensorflow as tf
from aer import read_naacl_alignments, AERSufficientStatistics
from utils import iterate_minibatches, prepare_data
from support import create_history

# for TF 1.1
import tensorflow

try:
    from tensorflow.contrib.keras.initializers import glorot_uniform
except:  # for TF 1.0
    from tensorflow.contrib.layers import xavier_initializer as glorot_uniform


class NeuralIBM1CollocationsModel:
    """Our Neural IBM1 model with collocations"""


    def __init__(self, batch_size=8, x_vocabulary=None, y_vocabulary=None, emb_dim=32, mlp_dim=64, session=None):
        self.batch_size = batch_size
        self.emb_dim = emb_dim
        self.mlp_dim = mlp_dim

        self.x_vocabulary = x_vocabulary
        self.y_vocabulary = y_vocabulary
        self.x_vocabulary_size = len(x_vocabulary)
        self.y_vocabulary_size = len(y_vocabulary)

        self._create_placeholders()
        self._create_weights()
        self._build_model()

        self.saver = tf.train.Saver()
        self.session = session

    def _create_placeholders(self):
        """We define placeholders to feed the data to TensorFlow."""
        # "None" means the batches may have a variable maximum length.
        self.x = tf.placeholder(tf.int64, shape=[None, None])
        self.y = tf.placeholder(tf.int64, shape=[None, None])
        self.history = tf.placeholder(tf.int64, shape=[None, None])

    def _create_weights(self):
        """Create weights for the model."""
        with tf.variable_scope("MLP") as scope:

            # for P(C|F_prev)
            self.a_ = tf.get_variable(
                name="a_", initializer=tf.zeros_initializer(),
                shape=[self.emb_dim, 1])

            # for P(F|F_prev)
            self.W_tF = tf.get_variable(
                name="W_tF", initializer=glorot_uniform(),
                shape=[self.emb_dim, self.y_vocabulary_size])
            self.b_tF = tf.get_variable(
                name="b_tF", initializer=tf.zeros_initializer(),
                shape=[self.y_vocabulary_size])

            # for P(F|E)
            self.W_tE = tf.get_variable(
                name="W_tE", initializer=glorot_uniform(),
                shape=[self.emb_dim, self.y_vocabulary_size])
            self.b_tE = tf.get_variable(
                name="b_tE", initializer=tf.zeros_initializer(),
                shape=[self.y_vocabulary_size])

            # Let's create a (source) word embeddings matrix.
            self.x_embeddings = tf.get_variable(
                name="x_embeddings", initializer=tf.random_uniform_initializer(),
                shape=[self.x_vocabulary_size, self.emb_dim])

            # Let's create a (target) word embeddings matrix.
            self.y_embeddings = tf.get_variable(
                name="y_embeddings", initializer=tf.random_uniform_initializer(),
                shape=[self.y_vocabulary_size, self.emb_dim])


    def save(self, session, path="model.ckpt"):
        """Saves the model."""
        return self.saver.save(session, path)


    def _build_model(self):
        """Builds the computational graph for our model."""



        # Now we start defining our graph.

        # This looks up the embedding vector for each word given the word IDs in self.x.
        # Shape: [B, M, emb_dim] where B is batch size, M is (longest) source sentence length.
        x_embedded = tf.nn.embedding_lookup(self.x_embeddings, self.x)

        # same for history
        # [B, N, emb_dim]
        history_embedded = tf.nn.embedding_lookup(self.y_embeddings, self.history)

        # 2. Now we define the generative model P(Y | X=x)

        # first we need to know some sizes from the current input data
        batch_size = tf.shape(self.x)[0]
        longest_x = tf.shape(self.x)[1]  # longest M
        longest_y = tf.shape(self.y)[1]  # longest N

        # It's also useful to have masks that indicate what
        # values of our batch we should ignore.
        # Masks have the same shape as our inputs, and contain
        # 1.0 where there is a value, and 0.0 where there is padding.
        x_mask = tf.cast(tf.sign(self.x), tf.float32)  # Shape: [B, M]
        y_mask = tf.cast(tf.sign(self.y), tf.float32)  # Shape: [B, N]
        # TODO : use history mask to mask the first empty history item
        # history_mask = tf.cast(tf.sign(self.history), tf.float32)  # Shape: [B, M]

        x_len = tf.reduce_sum(tf.sign(self.x), axis=1)  # Shape: [B]
        y_len = tf.reduce_sum(tf.sign(self.y), axis=1)  # Shape: [B]

        # 2.a Build an alignment model P(A | X, M, N)

        # This just gives you 1/length_x (already including NULL) per sample.
        # i.e. the lengths are the same for each word y_1 .. y_N.
        lengths = tf.expand_dims(x_len, -1)  # Shape: [B, 1]
        pa_x = tf.div(x_mask, tf.cast(lengths, tf.float32))  # Shape: [B, M]


        pa_x = tf.expand_dims(pa_x, 2)  # Shape: [B, M, 1]
        pa_x = tf.expand_dims(pa_x, 3)  # Shape: [B, M, 1, 1]
        # Now we perform the tiling:
        # pa_x = tf.tile(pa_x, [1, 1, longest_y, self.y_vocabulary_size])  # [B, M, N, y_vocab_size]

        # run NN
        py_xa = self.ffnn(x_embedded, history_embedded, batch_size, longest_x, longest_y)

        # 2.c Marginalise alignments: \sum_a P(a|x) P(Y|x,a)
        py_x = tf.reduce_sum(tf.multiply(pa_x, py_xa), axis=1)  # Shape: [B, N, Vy]

        # This calculates the accuracy, i.e. how many predictions we got right.
        predictions = tf.argmax(py_x, axis=2)
        acc = tf.equal(predictions, self.y)
        acc = tf.cast(acc, tf.float32) * y_mask
        acc_correct = tf.reduce_sum(acc)
        acc_total = tf.reduce_sum(y_mask)
        acc = acc_correct / acc_total

        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=tf.reshape(self.y, [-1]),
            logits=tf.log(tf.reshape(py_x, [batch_size * longest_y, self.y_vocabulary_size])),
            name="logits"
        )

        cross_entropy = tf.reshape(cross_entropy, [batch_size, longest_y])
        cross_entropy = tf.reduce_sum(cross_entropy * y_mask, axis=1)
        cross_entropy = tf.reduce_mean(cross_entropy, axis=0)

        self.pa_x = pa_x
        self.py_x = py_x
        self.py_xa = py_xa
        self.loss = cross_entropy
        self.predictions = predictions
        self.accuracy = acc
        self.accuracy_correct = tf.cast(acc_correct, tf.int64)
        self.accuracy_total = tf.cast(acc_total, tf.int64)


    def ffnn(self, x_embedded, history_embedded, batch_size, longest_x, longest_y):
        """
        Performs a feed-forward pass. Depending on the self.gate attribute will run different versions of the pass.
        p(f|e,f_prev) where sum over c is already performed.
        """
        history_embedded = tf.reshape(history_embedded, [batch_size*longest_y, self.emb_dim])
        x_embedded = tf.reshape(x_embedded, [batch_size * longest_x, self.emb_dim])

        # 1. compute P(C|F_prev) param
        s_f = tf.sigmoid(tf.matmul(history_embedded, self.a_))  # [B*N, 1]

        # 2. compute P(F|F_prev) param
        i_f = tf.nn.softmax(tf.matmul(history_embedded, self.W_tF) + self.b_tF)  # [B*N, vocab_size]

        # 3. compute P(F|E) param
        t_e = tf.nn.softmax(tf.matmul(x_embedded, self.W_tE) + self.b_tE)  # [B*M, vocab_size]

        # expand and reshape

        s_f = tf.reshape(s_f, [batch_size, longest_y])
        s_f = tf.expand_dims(s_f, axis=1)
        s_f = tf.expand_dims(s_f, axis=3)
        # s_f = tf.tile(s_f, [1, longest_x, 1, self.y_vocabulary_size])  # [B, M, N, vocab_size]

        i_f = tf.reshape(i_f, [batch_size, longest_y, self.y_vocabulary_size])  # [B, N, vocab_size]
        i_f = tf.expand_dims(i_f, axis=1)
        # i_f = tf.tile(i_f, [1, longest_x, 1, 1])  # [B, M, N, vocab_size]

        t_e = tf.reshape(t_e, [batch_size, longest_x, self.y_vocabulary_size])
        self.t_e = t_e # we will do inference based on it
        t_e = tf.expand_dims(t_e, axis=2)
        # t_e = tf.tile(t_e, [1, 1, longest_y, 1]) # [B, M, N, vocab_size]

        # produce distribution
        res = (1. - s_f) * t_e + s_f * i_f
        return res


    def evaluate(self, data, ref_alignments, batch_size=4):
        """Evaluate the model on a data set."""

        ref_align = read_naacl_alignments(ref_alignments)

        ref_iterator = iter(ref_align)
        metric = AERSufficientStatistics()
        accuracy_correct = 0
        accuracy_total = 0
        loss_total = 0
        steps = 0.

        for batch_id, batch in enumerate(iterate_minibatches(data, batch_size=batch_size)):
            x, y = prepare_data(batch, self.x_vocabulary, self.y_vocabulary)
            history = create_history(y)

            y_len = np.sum(np.sign(y), axis=1, dtype="int64")

            align, prob, acc_correct, acc_total, loss = self.get_viterbi(x, y, history)
            accuracy_correct += acc_correct
            accuracy_total += acc_total
            loss_total += loss
            steps += 1

            for alignment, N, (sure, probable) in zip(align, y_len, ref_iterator):
                # the evaluation ignores NULL links, so we discard them
                # j is 1-based in the naacl format
                pred = set((aj, j) for j, aj in enumerate(alignment[:N], 1) if aj > 0)
                metric.update(sure=sure, probable=probable, predicted=pred)
                # print(batch[s])
                print(alignment[:N])
                # print(pred)
                #       s +=1

        accuracy = accuracy_correct / float(accuracy_total)
        return metric.aer(), accuracy, loss_total/float(steps)

    def get_viterbi(self, x, y, history):
        """Returns the Viterbi alignment for (x, y)"""

        feed_dict = {
            self.x: x,  # English
            self.y: y,  # French
            self.history: history
        }

        # run model on this input
        t_e, py_xa, acc_correct, acc_total, loss = self.session.run(
            [self.t_e, self.py_xa, self.accuracy_correct, self.accuracy_total, self.loss],
            feed_dict=feed_dict)

        # things to return
        batch_size, longest_y = y.shape
        alignments = np.zeros((batch_size, longest_y), dtype="int64")
        probabilities = np.zeros((batch_size, longest_y), dtype="float32")

        for b, sentence in enumerate(y):
            for j, french_word in enumerate(sentence):
                if french_word == 0:  # Padding
                    break

                # probs = py_xa[b, :, j, french_word]
                probs = t_e[b, :, french_word]
                a_j = probs.argmax()
                p_j = probs[a_j]

                alignments[b, j] = a_j
                probabilities[b, j] = p_j

        return alignments, probabilities, acc_correct, acc_total, loss