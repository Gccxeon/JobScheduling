class Trainer():
  # A helper class used to take care of the training process
  def __init__(self,
               env,
               agent,
               collector,
               replay_buffer,
               samplizer,
               transitionizer,
               episodes):
    """
    Args:

      env: The training environment;
      agent: The agent being trained to adapt the environment;
      replay_buffer: The replay memory container;
      collecter: The collecter used to collect experice into replay_buffer;
      episodes: How many episodes do you need to train?

    """

    self._batch_size = agent.batch_size()
    if self._batch_size > len(replay_buffer):
      raise ValueError("The size of replay buffer is {}, which is smaller than"
                       "the batch size: {}!".format(len(replay_buffer),
                                                    self._batch_size))
    self._env = env
    self._env.reset()
    self._agent = agent
    self._replay_buffer = replay_buffer
    self._collector = collector
    self._samplizer = samplizer
    self._transitionizer = transitionizer
    self._episodes = episodes
    self._global_step = 0

  def train(self, episodes=None):
    episodes = episodes if episodes else self._episodes
    # start with a random action
    action = int(self._replay_buffer.sample(1)[0].action)

    for i in range(episodes + 1):
      if i>0:
        print("Episode {:2d}, loss: {:3f}, iterations: {:5d}, cum_reward: {:3f}".
              format(i,
                     self._agent.loss(),
                     self._global_step,
                     self._env.cum_reward()))

      self._env.reset()
      while not self._env.is_terminal():
        sample = self._replay_buffer.sample(self._batch_size)
        self._agent.train_step(sample)
        unprocessed = self._env.step(action)
        self._collector.collect_single_t(unprocessed)
        transition = self._transitionizer(*unprocessed)
        single_sample = self._samplizer.process(transition)
        action = int(self._agent.default_policy(single_sample))
        self._global_step += 1

