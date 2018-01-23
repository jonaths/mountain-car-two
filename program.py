import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections
from lib.MyEnv import MyEnv
# from lib.PolicyEstimator import PolicyEstimator
# from lib.ValueEstimator import ValueEstimator

import sklearn.pipeline
import sklearn.preprocessing

if "../" not in sys.path:
    sys.path.append("../")
# from lib.envs.cliff_walking import CliffWalkingEnv
from lib import plotting

from sklearn.kernel_approximation import RBFSampler

matplotlib.style.use('ggplot')



def run(budget, episodes):

    # env = gym.envs.make("MountainCarContinuous-v0")
    env = MyEnv("MountainCarContinuous-v0", budget)
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

    class PolicyEstimator:
        """
        Policy Function approximator.
        """

        def __init__(self, scaler, featurizer, learning_rate=0.01, scope="policy_estimator"):
            with tf.variable_scope(scope):
                self.scaler = scaler
                self.featurizer = featurizer
                self.state = tf.placeholder(tf.float32, [400], "state")
                self.target = tf.placeholder(dtype=tf.float32, name="target")

                # Clasificador lineal ###############################
                self.mu = tf.contrib.layers.fully_connected(
                    # convierte un vector de n en una matriz n x 1
                    inputs=tf.expand_dims(self.state, 0),
                    num_outputs=1,
                    activation_fn=None,
                    weights_initializer=tf.zeros_initializer)
                self.mu = tf.squeeze(self.mu)
                ######################################################

                # Red neuronal #######################################
                # mu_input_layer = tf.contrib.layers.fully_connected(
                #     inputs=tf.expand_dims(self.state, 0),
                #     num_outputs=8,
                #     activation_fn=tf.nn.relu,
                #     weights_initializer=tf.zeros_initializer)
                #
                # mu_hidden_layer = tf.contrib.layers.fully_connected(
                #     inputs=mu_input_layer,
                #     num_outputs=4,
                #     activation_fn=tf.nn.relu,
                #     weights_initializer=tf.zeros_initializer)
                #
                # mu_output_layer = tf.contrib.layers.fully_connected(
                #     inputs=mu_hidden_layer,
                #     num_outputs=1,
                #     activation_fn=None,
                #     weights_initializer=tf.zeros_initializer)
                # self.mu = tf.squeeze(mu_output_layer)
                ######################################################

                # Clasificador lineal ##################################
                # self.sigma = tf.contrib.layers.fully_connected(
                #     inputs=tf.expand_dims(self.state, 0),
                #     num_outputs=1,
                #     activation_fn=None,
                #     weights_initializer=tf.zeros_initializer)
                # self.sigma = tf.squeeze(self.sigma)
                ######################################################

                # Red neuronal #########################################
                # sigma_input_layer = tf.contrib.layers.fully_connected(
                #     inputs=tf.expand_dims(self.state, 0),
                #     num_outputs=4,
                #     activation_fn=None,
                #     weights_initializer=tf.zeros_initializer)
                #
                # sigma_hidden_layer = tf.contrib.layers.fully_connected(
                #     inputs=sigma_input_layer,
                #     num_outputs=3,
                #     activation_fn=None,
                #     weights_initializer=tf.zeros_initializer)
                #
                # sigma_output_layer = tf.contrib.layers.fully_connected(
                #     inputs=sigma_hidden_layer,
                #     num_outputs=1,
                #     activation_fn=None,
                #     weights_initializer=tf.zeros_initializer)
                # self.sigma = tf.squeeze(sigma_output_layer)
                ######################################################

                # IMPORTANTE: descomentar para usar con clasificador lineal o NN
                # self.sigma = tf.nn.softplus(self.sigma) + 1e-5

                # Valor constante ####################################
                self.sigma = tf.constant(1, dtype='float32')
                ######################################################

                # crea una distribucion normal
                self.normal_dist = tf.contrib.distributions.Normal(self.mu, self.sigma)
                # muestrea un valor con la distribucion anterior
                self.action = self.normal_dist._sample_n(1)
                # clip
                self.action = tf.clip_by_value(self.action, env.action_space().low[0], env.action_space().high[0])

                # Loss and train op
                self.loss = -self.normal_dist.log_prob(self.action) * self.target
                # Add cross entropy cost to encourage exploration
                self.loss -= 1e-1 * self.normal_dist.entropy()

                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_op = self.optimizer.minimize(
                    self.loss, global_step=tf.contrib.framework.get_global_step())

        def predict(self, state, sess=None):
            sess = sess or tf.get_default_session()
            state = featurize_state(state, self.scaler, self.featurizer)
            return sess.run(self.action, {self.state: state})

        def update(self, state, target, action, sess=None):
            sess = sess or tf.get_default_session()
            state = featurize_state(state, self.scaler, self.featurizer)
            feed_dict = {self.state: state, self.target: target, self.action: action}
            _, loss = sess.run([self.train_op, self.loss], feed_dict)
            return loss

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

                # fc1 = tf.contrib.layers.fully_connected(
                #     inputs=tf.expand_dims(self.state, 0),
                #     num_outputs=256,
                #     activation_fn=tf.nn.relu)
                # fc2 = tf.contrib.layers.fully_connected(
                #     inputs=fc1,
                #     num_outputs=256,
                #     activation_fn=tf.nn.relu)
                # self.output_layer = tf.contrib.layers.fully_connected(
                #     inputs=fc2,
                #     num_outputs=1,
                #     activation_fn=None)

                self.value_estimate = tf.squeeze(self.output_layer)
                self.loss = tf.squared_difference(self.value_estimate, self.target)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
                self.train_op = self.optimizer.minimize(
                    self.loss, global_step=tf.contrib.framework.get_global_step())

        def predict(self, state, sess=None):
            sess = sess or tf.get_default_session()
            state = featurize_state(state, self.scaler, self.featurizer)
            return sess.run(self.value_estimate, {self.state: state})

        def update(self, state, target, sess=None):
            sess = sess or tf.get_default_session()
            state = featurize_state(state, self.scaler, self.featurizer)
            feed_dict = {self.state: state, self.target: target}
            _, loss = sess.run([self.train_op, self.loss], feed_dict)
            return loss

    def actor_critic(env, estimator_policy, estimator_value, num_episodes, discount_factor=1.0):
        """
        Actor Critic Algorithm. Optimizes the policy
        function approximator using policy gradient.

        Args:
            env: OpenAI environment.
            estimator_policy: Policy Function to be optimized
            estimator_value: Value function approximator, used as a baseline
            num_episodes: Number of episodes to run for
            discount_factor: Time-discount factor

        Returns:
            An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
        """



        # Keeps track of useful statistics
        stats = plotting.EpisodeStats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes),
            episode_shaped_rewards=np.zeros(num_episodes),
            episode_spent=np.zeros(num_episodes),
            # 3 posibles razones por las que puede terminar
            episode_budget_count=np.zeros(num_episodes),
            )

        Transition = collections.namedtuple(
            "Transition", ["state", "action", "reward", "next_state", "done"])

        log = ''

        for i_episode in range(num_episodes):

            print "Episodio:", i_episode

            # Reset the environment and pick the fisrst action
            # [car position [-1.2, 0.6], car velocity[-0.07, 0.07], budget]
            state = env.reset()

            episode = []

            # One step in the environment
            for t in itertools.count():

                # env.render()

                # Take a step
                action = estimator_policy.predict(state)
                next_state, reward, shaped_reward, done, c = env.step(action, i_episode, t)

                log += np.array_str(action) + ", " + np.array_str(next_state) + ", " + str(reward) + ";\n"

                # Keep track of the transition
                episode.append(Transition(
                    state=state, action=action, reward=reward, next_state=next_state, done=done))

                # Update statistics
                stats.episode_rewards[i_episode] += reward
                stats.episode_shaped_rewards[i_episode] += shaped_reward
                stats.episode_lengths[i_episode] = t
                stats.episode_budget_count[i_episode] += c['episode_budget_count']

                # Calculate TD Target
                value_next = estimator_value.predict(next_state)
                # td_target = reward + discount_factor * value_next
                td_target = shaped_reward + discount_factor * value_next
                td_error = td_target - estimator_value.predict(state)

                # Update the value estimator
                estimator_value.update(state, td_target)

                # Update the policy estimator
                # using the td error as our advantage estimate
                estimator_policy.update(state, td_error, action)

                # Print out which step we're on, useful for debugging.
                # print("\rStep {} @ Episode {}/{} ({})".format(
                #     t, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]))

                if done:
                    print "  ", next_state, reward
                    break

                state = next_state

            # agrega el total gastado para el episodio
            stats.episode_spent[i_episode] = env.calculate_spent()

            if i_episode % 50 == 0:
                save_path = saver.save(sess, "model/"+str(budget)+"-"+str(i_episode)+".ckpt")

                # a partir de aqui guarda la funcion de valor para
                # varios valores de los estados
                items = 40

                # genera las columnas
                x1 = np.linspace(-1.2, 0.6, num=items)
                x2 = np.linspace(-0.07, 0.07, num=items)

                x1x, y1y = np.meshgrid(x1, x2)

                # enumera del 10 a budget incluyedolo
                step = 10
                bs = range(10, budget + step, step)

                # el numero de filas que contiene el arreglo (una para cada combinacion b x1 x2)
                length = items ** 2 * len(bs)

                # el numero de columnas del arreglo (una para cada parametro y la ultima para y)
                columns = 4

                v = np.zeros((length, columns))

                index = 0
                for b in bs:
                    v[index * items ** 2: (index + 1) * items ** 2, 0] = np.full((1, items ** 2), b)
                    v[index * items ** 2: (index + 1) * items ** 2, 1] = x1x.ravel()
                    v[index * items ** 2: (index + 1) * items ** 2, 2] = y1y.ravel()
                    index += 1

                # para cada fila de v
                for r in v:
                    # estima segun el modelo
                    r[-1] = estimator_value.predict(r[0:3])

                # bandera de no existe el archivo
                new = False

                ep_name = "{0:0>4}".format(i_episode)
                b_name = "{0:0>4}".format(budget)
                filename = 'values/b-' + b_name + '_ep-' + ep_name + '.npy'

                try:
                    arr = np.load(filename)
                    print "arr found", arr.shape
                    pass
                except IOError:
                    arr = np.zeros((1, length, 4))
                    arr[0] = v
                    new = True
                    print "arr created", arr.shape
                    pass

                if new:
                    # si es nuevo no hagas nada
                    pass
                else:
                    # si no es nuevo agregale el ultimo elemento
                    arr = np.append(arr, [v], axis=0)

                # guarda el archivo
                np.save(filename, arr)

        return stats, log

    tf.reset_default_graph()

    global_step = tf.Variable(0, name="global_step")
    policy_estimator = PolicyEstimator(scaler, featurizer, learning_rate=0.001)
    value_estimator = ValueEstimator(scaler, featurizer, learning_rate=0.1)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Note, due to randomness in the policy the number of episodes you need varies
        # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.
        stats, log = actor_critic(env, policy_estimator, value_estimator, episodes, discount_factor=0.95)

    # with tf.Session() as sess:
    #     sess.run(tf.initialize_all_variables())
    #     model_path = 'model/100-50.ckpt.index'
    #     saver.restore(sess, model_path)
    #
    #     # Note, due to randomness in the policy the number of episodes you need varies
    #     # TODO: Sometimes the algorithm gets stuck, I'm not sure what exactly is happening there.
    #     stats = actor_critic(env, policy_estimator, value_estimator, episodes, discount_factor=0.95)

    return stats, log
