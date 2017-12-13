import tensorflow as tf
from lib.MyEnv import MyEnv

import sklearn.preprocessing
import sklearn.pipeline
import numpy as np
from sklearn.kernel_approximation import RBFSampler

# env = gym.envs.make("MountainCarContinuous-v0")
env = MyEnv("MountainCarContinuous-v0", 100)
print env.action_space().low[0]

# Feature Preprocessing: Normalize to zero mean and unit variance
# We use a few samples from the observation space to do this
observation_examples = np.array([env.observation_space_sample() for x in range(10000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

# Used to converte a state to a featurizes represenation.
# We use RBF kernels with different variances to cover different parts of the space
featurizer = sklearn.pipeline.FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100))
])
featurizer.fit(scaler.transform(observation_examples))


def featurize_state(state, scaler, featurizer):
    """
    Returns the featurized representation for a state.
    """
    scaled = scaler.transform([state])
    featurized = featurizer.transform(scaled)
    return featurized[0]


class ValueEstimator:
    """
    Value Function approximator.
    """

    def __init__(self, scaler, featurizer, learning_rate=0.1, scope="value_estimator"):
        with tf.variable_scope(scope):
            self.scaler = scaler
            self.featurizer = featurizer
            self.state = tf.placeholder(tf.float32, [400], "state")
            self.target = tf.placeholder(dtype=tf.float32, name="target")

            # This is just linear classifier
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=tf.expand_dims(self.state, 0),
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.zeros_initializer)

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

            self.saver = tf.train.import_meta_graph('model/100-950.ckpt.meta')


    def predict(self, state, sess=None):
        sess = tf.Session()
        self.saver.restore(sess, tf.train.latest_checkpoint('./'))
        state = featurize_state(state, self.scaler, self.featurizer)
        return sess.run(self.value_estimate, {self.state: state})




value_estimator = ValueEstimator(scaler, featurizer, learning_rate=0.1)
state = np.array([0, 0, 50])
value_next = value_estimator.predict(state)
print value_next
