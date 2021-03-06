
class Generator(object):
  # the Generator object is used to generate a given number of objects, the
  # object being generated must have a __name__ attribute.
  def __init__(self, object_prototype, init_params_range):
    """
      Args:
        object_prototype: a prototype of the object to be generated;
        number_of_objects: the number of objects to generate;
        init_params_range: a dict contains the range of every param used to initialize
        the object, the key should be the name of the paramater, the value of each key
        should either be a list (for non-numerical type) or a tuple (the lower and
        upper bound of the numerical type)


    """
    self._ob_proto = object_prototype
    self._params_range = init_params_range

  def generate(self, num_to_generate, method='random'):
    # 'method' argument only works on the numerical objects
    objects = []
    for i in range(num_to_generate):
      init_params = self.build_params(method=method)
      objects.append( self._ob_proto(**init_params) )
    return objects

  def build_params(self,  method='random'):
    init_params = {}
    for param_name, param_range in self._params_range.items():
      if type(param_range) == list:
        param = random.sample(param_range, 1)[0]
      elif type(param_range) == tuple:
        low, upp = param_range
        param = random.uniform(low, upp)
      else:
        raise TypeError("The param range: {} is neither list nor tuple!"
                        .format(param_range))
      init_params[param_name] = param
    return init_params

class Collector():
  """
  A class that handles the data collections
  """
  def __init__(self,
               env,
               transitionizer,
               samplizer,
               data_type,
               saver,
               size,
               policy):
    """
    Args:
      env: The environment that gives out the transitions. Must support the
      'step()' function;
      date_type: the type of individual transition. This will keep the data
        type in the final collection consistent;
      saver: The object used to store the transitions. Must support the 'add'
        function.
      size: The amount of transitions needed to collect;
      policy: The policy that will be used to collect the transitions. The
        policy must has an 'action' function that will return the legit action
        when called.
    """

    self._env = env
    self._transitionizer = transitionizer
    self._samplizer = samplizer
    self._data_type = data_type
    self._saver = saver
    self._policy = policy
    self.collect(size)
    self._size = size

  def collect(self, sample_size):
    for i in range(sample_size):
      action = self._policy.action()

      # Get raw data from env
      unprocessed = self._env.step(action)
      # Process the transition
      transition = self._transitionizer(*unprocessed)
      self._transition_check(transition)
      transition = self._samplizer.process(transition)
      self._saver.add(transition)

  def collect_single_t(self, transition):
    transition = self._transitionizer(*transition)
    self._transition_check(transition)
    transition = self._samplizer.process(transition)
    self._saver.add(transition)


  def _transition_check(self, transition):
    if type(transition) != self._data_type:
      raise TypeError("The given transition type doesn't match the requirement. "
                      "Required {}, got {}.".format(self._data_type, transition))

def flatten21(arr):
  flated_elements = []
  def recur_get(item):
    if hasattr(item, '__iter__'):
      for ele in item:
        recur_get(ele)
    else:
      flated_elements.append(item)
  recur_get(arr)
  if len(flated_elements)==1:
    return flated_elements[0]
  return flated_elements
