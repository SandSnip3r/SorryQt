import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import nnx
from jax import random
from functools import partial

class MyModel(nnx.Module):
  def __init__(self, rngs):
    self.linear1 = nnx.Linear(in_features=323, out_features=32, rngs=rngs)
    self.linear2 = nnx.Linear(in_features=32, out_features=16050, rngs=rngs)

  def __call__(self, x):
    x = self.linear1(x)
    x = jax.nn.relu(x)
    x = self.linear2(x)
    return x

class TestClass:
  def __init__(self):
    print(f'Jax Devices: {jax.devices()}')
    print(f'Initializing my python class')
    self.model = MyModel(rngs=nnx.Rngs(0))
    print(f'Model:')
    nnx.display(self.model)
  
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

  def getAction(self, data):
    print(f'Got an observation: {data}')
    return self.model(data)