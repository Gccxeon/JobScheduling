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