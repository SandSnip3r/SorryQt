import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import math
import optax
import os
import sys
import pathlib
from flax import nnx
from functools import partial

class PolicyNetwork(nnx.Module):
  def __init__(self, rngs, actionSpaceSize):
    self.linear1 = nnx.Linear(in_features=327, out_features=32, rngs=rngs)
    self.linear2 = nnx.Linear(in_features=32, out_features=actionSpaceSize, rngs=rngs)

  def __call__(self, x):
    x = self.linear1(x)
    # assert not jnp.isnan(x).any(), "NaN detected after linear1"
    x = jax.nn.relu(x)
    # assert not jnp.isnan(x).any(), "NaN detected after ReLU"
    # Directly return the logits for
    #  1. Sampling (jax.random.categorical takes logits)
    #  2. Calculating probabilities
    x = self.linear2(x)
    # assert not jnp.isnan(x).any(), "NaN detected after linear2"
    return x

class ValueNetwork(nnx.Module):
  def __init__(self, rngs):
    self.linear1 = nnx.Linear(in_features=327, out_features=128, rngs=rngs)
    self.linear2 = nnx.Linear(in_features=128, out_features=1, rngs=rngs)

  def __call__(self, x):
    x = self.linear1(x)
    x = jax.nn.relu(x)
    x = self.linear2(x)
    return x.reshape()

# ================================================================================================
# ================================================================================================
# ================================================================================================

def printGradientInfo(gradients):
  flatGrads, _ = jax.tree_util.tree_flatten(gradients)
  flatGradsArray = jnp.concatenate([jnp.ravel(g) for g in flatGrads])
  mean_grad = jnp.mean(flatGradsArray)
  std_grad = jnp.std(flatGradsArray)
  max_grad = jnp.max(flatGradsArray)
  min_grad = jnp.min(flatGradsArray)
  max_idx = jnp.unravel_index(jnp.argmax(flatGradsArray), flatGradsArray.shape)
  min_idx = jnp.unravel_index(jnp.argmin(flatGradsArray), flatGradsArray.shape)

  # Print gradients and statistics
  jax.debug.print("Gradients:\n{flatGradsArray}", flatGradsArray=flatGradsArray)
  jax.debug.print("Mean of gradients: {mean_grad}", mean_grad=mean_grad)
  jax.debug.print("Std dev of gradients: {std_grad}", std_grad=std_grad)
  jax.debug.print("Max gradient value: {max_grad} at index {max_idx}", max_grad=max_grad, max_idx=max_idx)
  jax.debug.print("Min gradient value: {min_grad} at index {min_idx}", min_grad=min_grad, min_idx=min_idx)

@partial(nnx.jit)
def compiledUpdate(model, gradient, tdError, learningRate, l2Regularization):
  _, params, rest = nnx.split(model, nnx.Param, ...)
  # params = jax.tree.map(lambda p, g: p + learningRate * tdError * g, params, gradient)
  params = jax.tree.map(lambda p, g: p + learningRate * tdError * g - l2Regularization * p, params, gradient)
  nnx.update(model, nnx.GraphState.merge(params, rest))

def getProbabilityAndIndex(rngKey, policyNetwork, input, mask):
  # assert not jnp.isnan(input).any(), "NaN detected in input"
  logits = policyNetwork(input)
  # `mask` is a 1D array of size `actionSpaceSize`, where the index of the action to be masked is set to -inf, and all other indices are set to 0
  maskedLogits = logits + mask
  selectedIndex = jax.random.categorical(rngKey, maskedLogits)
  probabilities = jax.nn.softmax(maskedLogits)
  oneHotIndex = jax.nn.one_hot(selectedIndex, probabilities.shape[-1])
  selectedProbability = jnp.sum(jax.lax.stop_gradient(oneHotIndex) * probabilities)
  # if jnp.isnan(jax.lax.stop_gradient(selectedProbability)).any():
  #   jnp.set_printoptions(threshold=sys.maxsize)
  #   jax.debug.print(f"Logits: {jax.lax.stop_gradient(logits)}")
  #   assert not jnp.isnan(logits).any(), "NaN detected in logits"
  #   jax.debug.print(f"Masked Logits: {jax.lax.stop_gradient(maskedLogits)}")
  #   assert not jnp.isnan(maskedLogits).any(), "NaN detected in maskedLogits"
  #   jax.debug.print(f"Selected Index: {jax.lax.stop_gradient(selectedIndex)}")
  #   jax.debug.print(f"Probabilities: {jax.lax.stop_gradient(probabilities)}")
  #   assert not jnp.isnan(probabilities).any(), "NaN detected in probabilities"
  #   jax.debug.print(f'Selected probability: {jax.lax.stop_gradient(selectedProbability)}')
  return -jnp.log(selectedProbability), selectedIndex

def getProbabilitiesAndIndex(policyNetwork, input, mask):
  logits = policyNetwork(input)
  maskedLogits = logits + mask
  probabilities = jax.nn.softmax(maskedLogits)
  return probabilities, jax.numpy.argmax(probabilities)

def getValue(valueNetwork, input):
  return valueNetwork(input)

def loadPolicyNetworkFromCheckpoint(checkpointPath, actionSpaceSize):
  abstractModel = nnx.eval_shape(lambda: PolicyNetwork(rngs=nnx.Rngs(0), actionSpaceSize=actionSpaceSize))
  graphdef, abstractState = nnx.split(abstractModel)
  checkpointer = ocp.StandardCheckpointer()
  stateRestored = checkpointer.restore(checkpointPath, abstractState)
  return nnx.merge(graphdef, stateRestored)

def loadValueNetworkFromCheckpoint(checkpointPath):
  abstractModel = nnx.eval_shape(lambda: ValueNetwork(rngs=nnx.Rngs(0)))
  graphdef, abstractState = nnx.split(abstractModel)
  checkpointer = ocp.StandardCheckpointer()
  stateRestored = checkpointer.restore(checkpointPath, abstractState)
  return nnx.merge(graphdef, stateRestored)

def createNewPolicyNetwork(actionSpaceSize, rngs):
  return PolicyNetwork(rngs=rngs, actionSpaceSize=actionSpaceSize)

def createNewValueNetwork(rngs):
  return ValueNetwork(rngs=rngs)

# ================================================================================================
# ================================================================================================
# ================================================================================================

class InferenceClass:
  def __init__(self, actionSpaceSize):
    # Save the checkpoint path
    checkpointPath = pathlib.Path(os.path.join(os.getcwd(), 'checkpoints')) / 'reinforce_wbaseline_1p'
    print(f'Loading model from {checkpointPath}')

    # Load the model from checkpoint
    self.policyNetwork = loadPolicyNetworkFromCheckpoint(checkpointPath / 'policy', actionSpaceSize)
    self.valueNetwork = loadValueNetworkFromCheckpoint(checkpointPath / 'value')

    # Compile the inference functions
    # self.getProbabilityIndex = nnx.jit(getProbabilityAndIndex)
    self.getProbabilitiesAndIndex = nnx.jit(getProbabilitiesAndIndex)
  
  def getProbabilitiesAndSelectedIndex(self, data, mask):
    # We're usually calling this function from another thread from C++. Due to JAX's trace contexts, we need to clone the model
    # TODO: Manage the model at the C++ level
    modelClone = nnx.clone(self.policyNetwork)

    return self.getProbabilitiesAndIndex(modelClone, data, mask)

# ================================================================================================
# ================================================================================================
# ================================================================================================

@nnx.jit
def updateModels(policyGradients, valueGradients, rewards, values, masks, gamma, policyOptimizer, valueOptimizer):
  def calculateReturns(rewards, gamma):
    # Calculate returns for all timesteps at once using JAX operations
    length = len(rewards)
    indices = jnp.arange(length)[:, None]
    timesteps = jnp.arange(length)[None, :]
    mask = timesteps >= indices
    powers = gamma ** (timesteps - indices)
    masked_powers = powers * mask
    masked_rewards = rewards[None, :] * mask
    returns = jnp.sum(masked_powers * masked_rewards, axis=1)
    return returns

  def scaleByRank(x, scales):
    if x.ndim == 1:
      return x * scales
    elif x.ndim == 2:
      return x * scales[:, None]
    elif x.ndim == 3:
      return x * scales[:, None, None]

  vectorizedCalculateReturns = jax.vmap(calculateReturns, in_axes=(0, None))
  vectorizedDiscountCalculation = jax.vmap(lambda x, g: g**jnp.arange(len(x)), in_axes=(0, None))

  returns = vectorizedCalculateReturns(rewards, gamma)
  tdErrors = returns - values
  discounts = vectorizedDiscountCalculation(rewards, gamma)
  policyScale = tdErrors * discounts

  # Scale all gradients at once
  vectorizedScale = jax.vmap(scaleByRank, in_axes=(0, 0))
  scaledPolicyGradients = jax.tree.map(lambda x: vectorizedScale(x, policyScale), policyGradients)
  scaledValueGradients = jax.tree.map(lambda x: vectorizedScale(x, tdErrors), valueGradients)
  # Negate value gradients to perform gradient descent
  scaledValueGradients = jax.tree.map(lambda x: -x, scaledValueGradients)

  # Sum gradients across all timesteps
  finalPolicyGradient = jax.tree.map(lambda x: jnp.mean(x, axis=0), jax.tree.map(lambda x: jnp.sum(x, axis=1), scaledPolicyGradients))
  finalValueGradient = jax.tree.map(lambda x: jnp.mean(x, axis=0), jax.tree.map(lambda x: jnp.sum(x, axis=1), scaledValueGradients))

  # Update the models
  policyOptimizer.update(finalPolicyGradient)
  valueOptimizer.update(finalValueGradient)

  def calculateMeanAndStdDev(vals, mask):
    # Compute the mean
    sum_vals = jnp.sum(vals * mask)  # Sum only masked elements
    count = jnp.sum(mask)            # Count of unmasked elements
    mean = sum_vals / count

    # Compute the standard deviation
    squared_diffs = (vals - mean) ** 2
    stddev = jnp.sqrt(jnp.sum(squared_diffs * mask) / count)

    return mean, stddev

  vectorizedMeanAndStdDev = jax.vmap(calculateMeanAndStdDev, in_axes=(0, 0))
  means, stdDevs = vectorizedMeanAndStdDev(tdErrors, masks)

  return jnp.mean(means), jnp.mean(stdDevs)

class TrainingUtilClass:
  def __init__(self, actionSpaceSize, summaryWriter, checkpointName=None):
    self.summaryWriter = summaryWriter
    # Initialize RNG
    # TODO: Find some way to seed the RNG before creating the models
    self.rngs = nnx.Rngs(0, myAdditionalStream=1)

    # Save the checkpoint path
    self.checkpointPath = pathlib.Path(os.path.join(os.getcwd(), 'checkpoints')) / 'latest'

    # Create the model, either from checkpoint or from scratch
    if checkpointName is not None:
      print(f'Loading model from {self.checkpointPath}')
      self.policyNetwork = loadPolicyNetworkFromCheckpoint(self.checkpointPath/'policy', actionSpaceSize)
      self.valueNetwork = loadValueNetworkFromCheckpoint(self.checkpointPath/'value')
    else:
      self.policyNetwork = createNewPolicyNetwork(actionSpaceSize, self.rngs)
      self.valueNetwork = createNewValueNetwork(self.rngs)

    # Initialize the checkpointer
    self.checkpointer = ocp.StandardCheckpointer()

    # Compile the policy network inference function
    self.getProbabilityIndexAndGradient = nnx.jit(nnx.value_and_grad(getProbabilityAndIndex, argnums=1, has_aux=True))

    # Compile the value network inference function
    self.getValueAndValueGradient = nnx.jit(nnx.value_and_grad(getValue))
  
  def setSeed(self, seed):
    self.rngs = nnx.Rngs(0, myAdditionalStream=seed)

  def logLogitStatistics(self, input, episodeIndex):
    logits = self.policyNetwork(input)
    self.summaryWriter.add_histogram('logits', logits, episodeIndex)

  def getPolicyGradientAndIndex(self, data, mask):
    ((logProbability, index), gradient) = self.getProbabilityIndexAndGradient(self.rngs.myAdditionalStream(), self.policyNetwork, data, mask)
    return gradient, index

  def getValueGradientAndValue(self, data):
    (value, gradient) = self.getValueAndValueGradient(self.valueNetwork, data)
    return gradient, value

  def updatePolicyNetwork(self, gradient, tdError, learningRate, l2Regularization):
    compiledUpdate(self.policyNetwork, gradient, tdError, learningRate, l2Regularization)

  def updateValueNetwork(self, gradient, tdError, learningRate, l2Regularization):
    compiledUpdate(self.valueNetwork, gradient, tdError, learningRate, l2Regularization)

  def initializePolicyOptimizer(self, learningRate):
    learningRate = optax.linear_schedule(init_value=learningRate, end_value=learningRate/10, transition_steps=1000, transition_begin=7000)
    tx = optax.adam(learning_rate=learningRate)
    self.policyNetworkOptimizer = nnx.Optimizer(self.policyNetwork, tx)

  def initializeValueOptimizer(self, learningRate):
    tx = optax.adam(learning_rate=learningRate)
    self.valueNetworkOptimizer = nnx.Optimizer(self.valueNetwork, tx)

  def train(self, policyGradientsForTrajectories, valueGradientsForTrajectories, rewardsForTrajectories, valuesForTrajectories, gamma, episodeIndex):
    # Pad up to the nearest power of 2
    maxLength = max([len(x) for x in policyGradientsForTrajectories])
    newLength = int(2**math.ceil(math.log2(maxLength)))
    # print(f'Lengths: {[len(x) for x in policyGradientsForTrajectories]}')
    # print(f'Max length is {maxLength}, new length will be {newLength}')

    # Define a padding function
    def leftPadToMaxSizeWithZeros(x):
      pad_width = [(newLength - x.shape[0], 0)] + [(0, 0)] * (x.ndim - 1)
      return jnp.pad(x, pad_width, mode='constant', constant_values=0)

    # Stack and pad policy gradients
    stackedAndPaddedPolicyGradientsForTrajectories = [jax.tree.map(leftPadToMaxSizeWithZeros, jax.tree.map(lambda *x: jnp.stack(x), *policyGradients)) for policyGradients in policyGradientsForTrajectories]
    stackedAndPaddedPolicyGradients = jax.tree.map(lambda *x: jnp.stack(x), *stackedAndPaddedPolicyGradientsForTrajectories)

    # Stack and pad value gradients
    stackedAndPaddedValueGradientsForTrajectories = [jax.tree.map(leftPadToMaxSizeWithZeros, jax.tree.map(lambda *x: jnp.stack(x), *valueGradients)) for valueGradients in valueGradientsForTrajectories]
    stackedAndPaddedValueGradients = jax.tree.map(lambda *x: jnp.stack(x), *stackedAndPaddedValueGradientsForTrajectories)

    # Stack and pad rewards
    paddedRewardsForTrajectories = [leftPadToMaxSizeWithZeros(jnp.asarray(x)) for x in rewardsForTrajectories]
    paddedRewards = jnp.stack(paddedRewardsForTrajectories)

    # Stack and pad values
    paddedValuesForTrajectories = [leftPadToMaxSizeWithZeros(jnp.asarray(x)) for x in valuesForTrajectories]
    paddedValues = jnp.stack(paddedValuesForTrajectories)

    # Create a mask
    masks = jnp.concatenate([jnp.concatenate([jnp.zeros(newLength-len(x)), jnp.ones(len(x))]) for x in policyGradientsForTrajectories])[None, :]

    tdMean, tdStddev = updateModels(stackedAndPaddedPolicyGradients, stackedAndPaddedValueGradients, paddedRewards, paddedValues, masks, gamma, self.policyNetworkOptimizer, self.valueNetworkOptimizer)
    self.summaryWriter.add_scalar("episode/tdErrorMean", tdMean, episodeIndex)
    self.summaryWriter.add_scalar("episode/tdErrorStdDev", tdStddev, episodeIndex)
  
  def saveCheckpoint(self):
    policyNetworkPath = self.checkpointPath / 'policy'
    valueNetworkPath = self.checkpointPath / 'value'
    _, policyNetworkState = nnx.split(self.policyNetwork)
    self.checkpointer.save(policyNetworkPath, policyNetworkState, force=True)
    _, valueNetworkState = nnx.split(self.valueNetwork)
    self.checkpointer.save(valueNetworkPath, valueNetworkState, force=True)
    print(f'Saved checkpoint. Policy network at {policyNetworkPath} and value network at {valueNetworkPath}')
