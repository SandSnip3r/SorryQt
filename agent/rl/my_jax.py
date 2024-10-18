import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
from functools import partial

# # Example JAX neural network
# class MyModel(nn.Module):
#   @nn.compact
#   def __call__(self, x):
#     x = nn.Dense(128)(x)
#     x = nn.relu(x)
#     x = nn.Dense(1)(x)
#     return x

# model = MyModel()

# # Forward pass function
# def forward_pass(state):
#   params = ...  # Load or initialize your model parameters
#   return model.apply(params, state)

class TestClass:
  def __init__(self):
    print(f'Jax Devices: {jax.devices()}')
    print(f'Initializing my python class')
    self.model = nn.Dense(features=1)
    key1, key2 = random.split(random.key(0))
    dummyData = random.normal(key1, (1,), dtype=jnp.float32)
    self.parameters = self.model.init(key2, dummyData)
    print(f'Parameters: {self.parameters}')
  
  # @staticmethod
  # @jax.jit
  # def f(model, parameters, input):
  #   modelResult = model.apply(parameters, jnp.array([input]))
  #   return modelResult.item()
  
  @partial(jax.jit, static_argnums=(0,))
  def func(self, arg: float):
    modelResult = self.model.apply(self.parameters, jnp.array([arg]))
    return jnp.asarray(modelResult)

    # return self.f(self.model, self.parameters, arg)

  def getObs(self, data):
    print(f'Got an observation: {data}')