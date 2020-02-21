class Trainer():
  # A helper class used to take care of the training process
  def __init__(self,
               env,
               agent,
               collector,
               replay_buffer,
               postprocessor,
               preprocessor,
               transition_type,
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
    self._postprocessor = postprocessor
    self._preprocessor = preprocessor
    self._transition_type = transition_type
    self._episodes = episodes
    self._global_step = 0

  def train(self):
    # start with a random action
    action = int(self._replay_buffer.sample(1)[0].action)

    for i in range(self._episodes):
      print("Episode {}, loss: {}, iterations: {}, cum_reward: {}".
            format(i,
                   self._agent.loss(),
                   self._global_step,
                   self._env.cum_reward()))

      self._env.reset()
      while not self._env.is_terminal():
        raw_sample = self._replay_buffer.sample(self._batch_size)
        sample = self._preprocessor(raw_sample)
        self._agent.train_step(sample)
        unprocessed = self._env.step(action)
        raw_transition = self._transition_type(*unprocessed)
        transition = self._postprocessor.process(raw_transition)
        transition = self._preprocessor(transition)
        action = int(self._agent.default_policy(transition.state))
        self._collector.collect_single_t(unprocessed)
        self._global_step += 1

