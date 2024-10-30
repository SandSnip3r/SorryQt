import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import os
import pathlib
from flax import nnx
from functools import partial

class SorryDenseModel(nnx.Module):
  def __init__(self, rngs, actionSpaceSize):
    self.linear1 = nnx.Linear(in_features=327, out_features=32, rngs=rngs)
    self.linear2 = nnx.Linear(in_features=32, out_features=actionSpaceSize, rngs=rngs)

  def __call__(self, x):
    x = self.linear1(x)
    x = jax.nn.relu(x)
    # Directly return the logits for both sampling (jax.random.categorical takes logits) and calculating probabilities
    return self.linear2(x)

# ================================================================================================
# ================================================================================================
# ================================================================================================

@partial(nnx.jit)
def compiledUpdate(model, gradient, reward, learningRate):
  _, params, rest = nnx.split(model, nnx.Param, ...)
  params = jax.tree.map(lambda p, g: p + learningRate * reward * g, params, gradient)
  nnx.update(model, nnx.GraphState.merge(params, rest))

def getProbabilityAndIndex(key, model, data, mask):
  logits = model(data)
  maskedLogits = logits + mask
  selectedIndex = jax.random.categorical(key, maskedLogits)
  probabilities = jax.nn.softmax(maskedLogits)
  oneHotIndex = jax.nn.one_hot(selectedIndex, probabilities.shape[-1])
  selectedProbability = jnp.sum(jax.lax.stop_gradient(oneHotIndex) * probabilities)
  return jnp.log(selectedProbability), selectedIndex

def getProbabilitiesAndIndex(model, data, mask):
  logits = model(data)
  maskedLogits = logits + mask
  probabilities = jax.nn.softmax(maskedLogits)
  return probabilities, jax.numpy.argmax(probabilities)

def loadModelFromCheckpoint(checkpointPath, actionSpaceSize):
  abstractModel = nnx.eval_shape(lambda: SorryDenseModel(rngs=nnx.Rngs(0), actionSpaceSize=actionSpaceSize))
  graphdef, abstractState = nnx.split(abstractModel)
  checkpointer = ocp.StandardCheckpointer()
  stateRestored = checkpointer.restore(checkpointPath, abstractState)
  return nnx.merge(graphdef, stateRestored)
# ================================================================================================
# ================================================================================================
# ================================================================================================

class InferenceClass:
  def __init__(self, actionSpaceSize):
    checkpointPath = pathlib.Path(os.path.join(os.getcwd(), 'checkpoints')) / 'reinforce_1p_any'
    print(f'Loading model from {checkpointPath}')
    self.model = loadModelFromCheckpoint(checkpointPath, actionSpaceSize)
    self.getProbabilityIndex = nnx.jit(getProbabilityAndIndex)
    self.getProbabilitiesAndIndex = nnx.jit(getProbabilitiesAndIndex)
  
  def getProbabilitiesAndSelectedIndex(self, data, mask):
    modelClone = nnx.clone(self.model)
    return self.getProbabilitiesAndIndex(modelClone, data, mask)

# ================================================================================================
# ================================================================================================
# ================================================================================================

class TrainingUtilClass:
  def __init__(self, actionSpaceSize, checkpointName=None):
    self.rngs = nnx.Rngs(0, sampleStream=1)
    self.checkpointPath = pathlib.Path(os.path.join(os.getcwd(), 'checkpoints')) / 'latest'
    if checkpointName is not None:
      print(f'Loading model from {self.checkpointPath}')
      self.model = loadModelFromCheckpoint(self.checkpointPath, actionSpaceSize)
    else:
      self.model = SorryDenseModel(rngs=self.rngs, actionSpaceSize=actionSpaceSize)
    self.checkpointer = ocp.StandardCheckpointer()

    self.getProbabilityIndexAndGradient = nnx.jit(nnx.value_and_grad(getProbabilityAndIndex, argnums=1, has_aux=True))
  
  def setSeed(self, seed):
    self.rngs = nnx.Rngs(0, sampleStream=seed)

  def getGradientAndIndex(self, data, mask):
    ((logProbability, index), gradient) = self.getProbabilityIndexAndGradient(self.rngs.sampleStream(), self.model, data, mask)
    return gradient, index

  def update(self, gradient, reward, learningRate):
    compiledUpdate(self.model, gradient, reward, learningRate)
  
  def saveCheckpoint(self):
    _, state = nnx.split(self.model)
    self.checkpointer.save(self.checkpointPath, state, force=True)
    print(f'Saved checkpoint at {self.checkpointPath}')
