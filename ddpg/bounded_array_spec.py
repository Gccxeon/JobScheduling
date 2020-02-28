import torch
from collections import namedtuple

# A 'hard' method that checks if the given object is a sequence that supports
# slicing and len methods
def is_sequence(obj):
  try:
    len(obj)
    obj[0:0]
    return True
  except TypeError:
    return False

class BoundedArraySpec():
  """
  This class is used to add the numerical constraints to the torch tensor.
  """
  def __init__(self,
               name,
               minimum,
               maximum,
               data,
               dtype=None,
               allow_min=True,
               allow_max=True):
    """
    Args:
      name: Name of the data, for example, "action";
      minimum: The minimum the data can be;
      maximum: The maximium the data can be;
      data: A single or a sequence of python number(s);
      dtype: torch.int32, troch.float64 ...;
      allow_min: Allow the data to be the minimum;
      allow_max: Allow the data to be the maximum.
    """


    self._check_bounds(minimum, maximum, allow_min, allow_max, dtype)

    self._name = name
    self._minimum = minimum
    self._maximum = maximum
    self._allow_min = allow_min
    self._allow_max = allow_max
    self._dtype = dtype

    self._check_data(data)

    if dtype in [int, float]:
      self._dtype = dtype
    elif dtype:
      self._dtype = getattr(torch, dtype)

    self._data = torch.tensor(data, dtype=self._dtype)
    self._dtype = self._data.dtype

    reprentation = namedtuple(name, ("data", "dtype", "minimum", "maximum"))
    self._spec = reprentation(self._data, self._dtype, minimum, maximum)


  def __repr__(self):
    return self._spec.__repr__()

  # return the data of the spec
  @property
  def data(self):
    return self._data

  # return name of the spec
  @property
  def name(self):
    return self._name

  # return the minimum of the spec
  @property
  def minimum(self):
    return self._minimum

  # return the maximum of the spec
  @property
  def maximum(self):
    return self._maximum

  # Check if the boundary condition has any logical error
  def _check_bounds(self, minimum, maximum, allow_min, allow_max, dtype):
    valid_int = [int, "int8", "int16", "int32", "int64"]
    valid_float = [None, float, "float8", "float16", "float32", "float64"]

    if dtype not in (valid_int + valid_float):
      raise TypeError("This bounded array spec only supports data type "
                      "including {}, while the given dtype {} is not in them."
                      .format(valid_int+valid_float, dtype))
    if minimum > maximum:
      raise ValueError("The minimum {} is bigger than the maximum {}!"
                       .format(minimum, maximum))
    elif minimum == maximum:
      if not (allow_min and allow_max):
        raise ValueError("The minimum {} and maximum {} are the same, "
                         "But the boundary cannot be reached"
                         .format(minimum, maximum))
    elif dtype in valid_int:
      if (minimum == maximum - 1) and not(allow_min and allow_max):
        raise ValueError("The space between minimum {} and maximum {} "
                         "are not enough to put any data in since the "
                         "boundary can't be reached".format(minimum, maximum))


  # Check if the data is consistent with the boundary condition
  def _check_data(self, data):

    # Handling the sequence input
    if is_sequence(data):
      for data_ in data:
        self._check_data(data_)
    else:
      # Check if the data is inside the interval
      if not(data <= self._maximum and data >= self._minimum):
        raise ValueError("The given data {} is not in the valid period {}!"
                         .format(data, (self._minimum, self.maximum)))

      # Check if the data violates the boundary condition
      if data == self._minimum and not(self._allow_min):
        raise ValueError("The given data {} equals minimum boundary, "
                         "but the boundary is not reachable!")
      if data == self._maximum and not(self._allow_max):
        raise ValueError("The given data {} equals maximum boundary, "
                         "but the boundary is not reachable!")
