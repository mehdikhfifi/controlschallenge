# controllers/pid.py
from . import BaseController
import numpy as np


  # def __init__(self, p=0.09433659312462063, i=0.16311895952264427, d=-0.024751499173566064):
#   def __init__(self, p=0.11421562702248518, i=0.14300035451862608, d=-0.04947878374407077 ):
    #   0.09833509028084614, 0.12733172285678593, 0.05784999576551308




class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self, p=0.09833509028084614, i=0.12733172285678593, d=-0.05784999576551308 ):
    self.p = p
    self.i = i
    self.d = d
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      error = (target_lataccel - current_lataccel)
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      return self.p * error + self.i * self.error_integral + self.d * error_diff
