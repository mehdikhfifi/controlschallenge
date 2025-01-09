
from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  
  def __init__(self, p=0.11421562702248518, i=0.14300035451862608, d=-0.04947878374407077 ):
    self.p = p
    self.i = i
    self.d = d
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      error = target_lataccel - current_lataccel
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      return self.p * error + self.i * self.error_integral + self.d * error_diff
