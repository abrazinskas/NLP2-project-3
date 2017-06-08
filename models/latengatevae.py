import numpy as np
import tensorflow as tf
from aer import read_naacl_alignments, AERSufficientStatistics
from utils import iterate_minibatches, prepare_data

# for TF 1.1
import tensorflow
try:
  from tensorflow.contrib.keras.initializers import glorot_uniform
except:  # for TF 1.0
  from tensorflow.contrib.layers import xavier_initializer as glorot_uniform

class LatentGateVAE:
  """Our Neural IBM1 model."""

  def __init__(self, batch_size=8,
               x_vocabulary=None, y_vocabulary=None,
               emb_dim=32, mlp_dim=64,
               session=None, prev_f_weight=0.5, eps=1e-8):

    self.batch_size = batch_size
    self.emb_dim = emb_dim
    self.mlp_dim = mlp_dim
    self.prev_f_weight = prev_f_weight
    self.eps = eps

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
    self.is_training = tf.placeholder(tf.bool)

  def _create_weights(self):
    """Create weights for the model."""
    with tf.variable_scope("MLP") as scope:


        # ============= phi weights ============
        self.phi_W_ha = tf.get_variable(
            name="phi_W_ha", initializer=glorot_uniform(),
            shape=[self.emb_dim, self.mlp_dim])

        self.phi_b_ha = tf.get_variable(
            name="phi_b_ha", initializer=tf.zeros_initializer(),
            shape=[self.mlp_dim])

        self.phi_W_hb = tf.get_variable(
            name="phi_W_hb", initializer=glorot_uniform(),
            shape=[self.emb_dim, self.mlp_dim])

        self.phi_b_hb = tf.get_variable(
            name="phi_b_hb", initializer=tf.zeros_initializer(),
            shape=[self.mlp_dim])

        self.phi_W_a = tf.get_variable(
            name="phi_W_a", initializer=glorot_uniform(),
            shape=[self.mlp_dim, 1])

        self.phi_b_a = tf.get_variable(
            name="phi_b_a", initializer=tf.zeros_initializer(),
            shape=[1])

        self.phi_W_b = tf.get_variable(
            name="phi_W_b", initializer=glorot_uniform(),
            shape=[self.mlp_dim, 1])

        self.phi_b_b = tf.get_variable(
            name="phi_b_b", initializer=tf.zeros_initializer(),
            shape=[1])

        # ============= theta weights ============
        self.mlp_W_ref = tf.get_variable(
            name="W_ref", initializer=glorot_uniform(),
            shape=[self.emb_dim, self.mlp_dim])

        self.mlp_b_ref = tf.get_variable(
            name="b_ref", initializer=tf.zeros_initializer(),
            shape=[self.mlp_dim])

        self.mlp_W_t = tf.get_variable(
            name="W_t", initializer=glorot_uniform(),
            shape=[self.mlp_dim, self.y_vocabulary_size])

        self.mlp_b_t = tf.get_variable(
            name="b_t", initializer=tf.zeros_initializer(),
            shape=[self.y_vocabulary_size])

        self.th_W_ha = tf.get_variable(
            name="th_W_ha", initializer=glorot_uniform(),
            shape=[self.emb_dim, self.mlp_dim])

        self.th_b_ha = tf.get_variable(
            name="th_b_ha", initializer=tf.zeros_initializer(),
            shape=[self.mlp_dim])

        self.th_W_hb = tf.get_variable(
            name="th_W_hb", initializer=glorot_uniform(),
            shape=[self.emb_dim, self.mlp_dim])

        self.th_b_hb = tf.get_variable(
            name="th_b_hb", initializer=tf.zeros_initializer(),
            shape=[self.mlp_dim])

        self.th_W_a = tf.get_variable(
            name="th_W_a", initializer=glorot_uniform(),
            shape=[self.mlp_dim, 1])

        self.th_b_a = tf.get_variable(
            name="th_b_a", initializer=tf.zeros_initializer(),
            shape=[1])

        self.th_W_b = tf.get_variable(
            name="th_W_b", initializer=glorot_uniform(),
            shape=[self.mlp_dim, 1])

        self.th_b_b = tf.get_variable(
            name="th_b_b", initializer=tf.zeros_initializer(),
            shape=[1])

  def save(self, session, path="model.ckpt"):
    """Saves the model."""
    return self.saver.save(session, path)

  def _build_model(self):
    """Builds the computational graph for our model."""

    x_embeddings = tf.get_variable(
      name="x_embeddings", initializer=tf.random_uniform_initializer(),
      shape=[self.x_vocabulary_size, self.emb_dim])
    y_embeddings = tf.get_variable(
        name="y_embeddings", initializer=tf.random_uniform_initializer(),
        shape=[self.y_vocabulary_size, self.emb_dim])

    batch_size = tf.shape(self.x)[0]
    longest_x = tf.shape(self.x)[1]  # longest M
    longest_y = tf.shape(self.y)[1]  # longest N

    x_embedded = tf.nn.embedding_lookup(x_embeddings, self.x)
    y_embedded = tf.nn.embedding_lookup(y_embeddings, self.y)
    padding = tf.zeros((batch_size, 1), dtype=tf.int32)
    padding = tf.nn.embedding_lookup(y_embeddings, padding)
    y_prev_embedded = tf.concat([padding, y_embedded[:, :-1, :]], axis=1)

    # ========== Getting a sample s_j for all f_j ============

    # Take a weighted average of the previous french word and the current french word.
    r_ff = (1.0 - self.prev_f_weight) * y_embedded + self.prev_f_weight * y_prev_embedded
    r_ff = tf.reshape(r_ff, [batch_size * longest_y, self.emb_dim])

    # Compute Kuramaswamy param alpha.
    ha = tf.matmul(r_ff, self.phi_W_ha) + self.phi_b_ha
    ha = tf.tanh(ha)
    alpha = tf.exp(tf.matmul(ha, self.phi_W_a) + self.phi_b_a)  # [B * N, 1]

    # Compute Kumaraswamy param beta.
    hb = tf.matmul(r_ff, self.phi_W_hb) + self.phi_b_hb    # affine transformation
    hb = tf.tanh(hb)                                       # non-linearity
    beta = tf.exp(tf.matmul(hb, self.phi_W_b) + self.phi_b_b)      # affine transformation [B * N, 1]

    # TODO This doesn't seem to work for numerical issues.
    # alpha = tf.maximum(alpha, 10)
    # beta = tf.maximum(beta, 10)

    # Sample some random uniform numbers. Then calculate s using a and b
    # which is then Kumaraswamy distributed.
    u = tf.random_uniform(tf.shape(alpha), minval=0., maxval=1.)
    s = tf.pow((1.0 - tf.pow(u, tf.pow(beta + self.eps, -1))), tf.pow(alpha + self.eps, -1)) # [B * N, 1]

    # ========== Compute a and b for the Beta distribution. ===========
    y_prev_embedded = tf.reshape(y_prev_embedded, [batch_size * longest_y, self.emb_dim])
    ha = tf.matmul(y_prev_embedded, self.th_W_ha) + self.th_b_ha    # affine transformation
    ha = tf.tanh(ha)                                                # non-linearity
    a = tf.exp(tf.matmul(ha, self.th_W_a) + self.th_b_a)            # affine transformation [B * N, 1]

    # Compute Kumaraswamy param beta.
    hb = tf.matmul(y_prev_embedded, self.th_W_hb) + self.th_b_hb               # affine transformation
    hb = tf.tanh(hb)                                                # non-linearity
    b = tf.exp(tf.matmul(hb, self.th_W_b) + self.th_b_b)            # affine transformation [B * N, 1]

    # TODO This doesn't seem to work for numerical issues.
    # a = tf.maximum(a, 10)
    # b = tf.maximum(b, 10)

    # Change s to the Beta mean if we're not training
    s = tf.cond(self.is_training, lambda: s, lambda: a / (a + b))

    x_mask = tf.cast(tf.sign(self.x), tf.float32)  # Shape: [B, M]
    y_mask = tf.cast(tf.sign(self.y), tf.float32)  # Shape: [B, N]
    x_len = tf.reduce_sum(tf.sign(self.x), axis=1)  # Shape: [B]
    y_len = tf.reduce_sum(tf.sign(self.y), axis=1)  # Shape: [B]

    lengths = tf.expand_dims(x_len, -1)  # Shape: [B, 1]
    pa_x = tf.div(x_mask, tf.cast(lengths, tf.float32))  # Shape: [B, M]
    pa_x = tf.expand_dims(pa_x, 2)  # Shape: [B, M, 1]
    pa_x = tf.expand_dims(pa_x, 3)  # Shape: [B, M, 1, 1]

    s = tf.reshape(s, [batch_size, longest_y, 1])
    s = tf.expand_dims(s, 1)
    s = tf.tile(s, [1, longest_x, 1, self.emb_dim])

    r_ff = tf.reshape(r_ff, [batch_size, longest_y, self.emb_dim])
    r_ff = tf.expand_dims(tf.tanh(r_ff), 1)  # [B, 1, N, emb]
    r_ff = tf.tile(r_ff, [1, longest_x, 1, 1])  # [B, M, N, emb]

    # expand source embeddings
    x_embedded = tf.expand_dims(tf.tanh(x_embedded), 2)  # [B, M, 1, emb]
    x_embedded = tf.tile(x_embedded, [1, 1, longest_y, 1])  # [B, M, N, emb]

    h = s * r_ff + (1. - s) * x_embedded

    h = tf.reshape(h, [batch_size * longest_x * longest_y, self.emb_dim])
    h = tf.matmul(h, self.mlp_W_ref) + self.mlp_b_ref        # affine transformation [B * M, N, Vy]
    h = tf.tanh(h)                                           # non-linearity
    h = tf.matmul(h, self.mlp_W_t) + self.mlp_b_t            # affine transformation [B * M, Vy]

    # Now we perform a softmax which operates on a per-row basis.
    py_xa = tf.nn.softmax(h)
    py_xa = tf.reshape(py_xa, [batch_size, longest_x, longest_y, self.y_vocabulary_size])
    py_x = tf.reduce_sum(tf.multiply(pa_x, py_xa), axis=1)

    # This calculates the accuracy, i.e. how many predictions we got right.
    predictions = tf.argmax(py_x, axis=2)
    acc = tf.equal(predictions, self.y)
    acc = tf.cast(acc, tf.float32) * y_mask
    acc_correct = tf.reduce_sum(acc)
    acc_total = tf.reduce_sum(y_mask)
    acc = acc_correct / acc_total

    # =========== KL Part ==============
    # Numerical stability
    alpha = tf.clip_by_value(alpha, 0.001, 10)
    beta = tf.clip_by_value(beta, 0.001, 10)
    a = tf.clip_by_value(a, 0.001, 10)
    b = tf.clip_by_value(b, 0.001, 10)

    KL = ((alpha - a) / (alpha)) * (-np.euler_gamma - tf.digamma(beta) - (1.0 / beta))
    KL += tf.log(alpha * beta)
    KL += tf.lbeta(tf.concat([tf.expand_dims(a , -1), tf.expand_dims(b, -1)], axis=-1))
    KL -= (beta - 1.) / (beta)

    # Taylor approximation
    taylor_approx = tf.zeros(tf.shape(a))
    for m in range(1, 1 + 10):
        taylor_approx += (1.0 / (m + alpha * beta)) * tf.exp(tf.lbeta(tf.concat([tf.expand_dims(m/alpha, -1), \
                tf.expand_dims(beta, -1)], axis=-1)))
    KL += (b - 1.0) * beta * taylor_approx

    KL = tf.reshape(KL, [batch_size, longest_y])
    KL = tf.reduce_sum(KL * y_mask, axis=1)
    KL = tf.reduce_mean(KL, axis=0)
    self.KL = KL

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(self.y, [-1]),
      logits=tf.log(tf.reshape(py_x,[batch_size * longest_y, self.y_vocabulary_size])),
      name="logits"
    )
    cross_entropy = tf.reshape(cross_entropy, [batch_size, longest_y])
    cross_entropy = tf.reduce_sum(cross_entropy * y_mask, axis=1)
    cross_entropy = tf.reduce_mean(cross_entropy, axis=0)

    ELBO = cross_entropy + KL

    self.pa_x = pa_x
    self.py_x = py_x
    self.py_xa = py_xa
    self.loss = ELBO
    self.predictions = predictions
    self.accuracy = acc
    self.accuracy_correct = tf.cast(acc_correct, tf.int64)
    self.accuracy_total = tf.cast(acc_total, tf.int64)

  def evaluate(self, data, ref_alignments, batch_size=4):
    """Evaluate the model on a data set."""

    ref_align = read_naacl_alignments(ref_alignments)

    ref_iterator = iter(ref_align)
    metric = AERSufficientStatistics()
    accuracy_correct = 0
    accuracy_total = 0

    for batch_id, batch in enumerate(iterate_minibatches(data, batch_size=batch_size)):
      x, y = prepare_data(batch, self.x_vocabulary, self.y_vocabulary)
      y_len = np.sum(np.sign(y), axis=1, dtype="int64")

      align, prob, acc_correct, acc_total = self.get_viterbi(x, y)
      accuracy_correct += acc_correct
      accuracy_total += acc_total

#       if batch_id == 0:
#         print(batch[0])
#      s = 0

      for alignment, N, (sure, probable) in zip(align, y_len, ref_iterator):
        # the evaluation ignores NULL links, so we discard them
        # j is 1-based in the naacl format
        pred = set((aj, j) for j, aj in enumerate(alignment[:N], 1) if aj > 0)
        metric.update(sure=sure, probable=probable, predicted=pred)
 #       print(batch[s])
 #       print(alignment[:N])
 #       print(pred)
 #       s +=1

    accuracy = accuracy_correct / float(accuracy_total)
    return metric.aer(), accuracy

  def get_viterbi(self, x, y):
    """Returns the Viterbi alignment for (x, y)"""

    feed_dict = {
        self.x: x,  # English
        self.y: y,  # French
        self.is_training: False # Use Beta distr mean instead of a Kuma sample.
    }

    # run model on this input
    py_xa, acc_correct, acc_total, loss = self.session.run(
        [self.py_xa, self.accuracy_correct, self.accuracy_total, self.loss],
        feed_dict=feed_dict)

    # things to return
    batch_size, longest_y = y.shape
    alignments = np.zeros((batch_size, longest_y), dtype="int64")
    probabilities = np.zeros((batch_size, longest_y), dtype="float32")

    for b, sentence in enumerate(y):
        for j, french_word in enumerate(sentence):
            if french_word == 0:  # Padding
                break

            probs = py_xa[b, :, j, french_word]
            a_j = probs.argmax()
            p_j = probs[a_j]

            alignments[b, j] = a_j
            probabilities[b, j] = p_j

    return alignments, probabilities, acc_correct, acc_total, loss
