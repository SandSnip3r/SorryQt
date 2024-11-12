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
  def __init__(self, rngs):
    inFeatureSize = 4+11*5+2*4*67
    stateLinearOutputSize = 512
    actionTypeCount = 5
    cardCount = 11
    positionCount = 67
    self.stateLinear = nnx.Linear(in_features=inFeatureSize, out_features=stateLinearOutputSize, rngs=rngs)

    self.actionTypeLinear = nnx.Linear(in_features=stateLinearOutputSize, out_features=actionTypeCount, rngs=rngs)
    self.cardLinear = nnx.Linear(in_features=stateLinearOutputSize, out_features=cardCount, rngs=rngs)
    self.move1SourceLinear = nnx.Linear(in_features=stateLinearOutputSize, out_features=positionCount, rngs=rngs)
    self.move1DestLinear = nnx.Linear(in_features=stateLinearOutputSize, out_features=positionCount, rngs=rngs)
    self.move2SourceLinear = nnx.Linear(in_features=stateLinearOutputSize, out_features=positionCount, rngs=rngs)
    self.move2DestLinear = nnx.Linear(in_features=stateLinearOutputSize, out_features=positionCount, rngs=rngs)

  def __call__(self, x):
    x = self.stateLinear(x)
    x = jax.nn.relu(x)
    return {
      'actionTypeLogits': self.actionTypeLinear(x),
      'cardLogits': self.cardLinear(x),
      'move1SourceLogits': self.move1SourceLinear(x),
      'move1DestLogits': self.move1DestLinear(x),
      'move2SourceLogits': self.move2SourceLinear(x),
      'move2DestLogits': self.move2DestLinear(x)
    }

class ValueNetwork(nnx.Module):
  def __init__(self, rngs):
    inFeatureSize = 4+11*5+2*4*67
    self.linear1 = nnx.Linear(in_features=inFeatureSize, out_features=128, rngs=rngs)
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

def getProbabilityAndActionTuple(rngKey, policyNetwork, input, actions, stillValidActionMask):
  # These are the concrete sizes of the different one-hot vectors for the different parts of the action
  sizes = (5,11,67,67,67,67)

  # These are the one-hot masks for each part of all of the given actions
  masks = {
    'actionTypeMask': jax.nn.one_hot(actions[:,0], sizes[0]),
    'cardMask': jax.nn.one_hot(actions[:,1], sizes[1]),
    'move1SourceMask': jax.nn.one_hot(actions[:,2], sizes[2]),
    'move1DestMask': jax.nn.one_hot(actions[:,3], sizes[3]),
    'move2SourceMask': jax.nn.one_hot(actions[:,4], sizes[4]),
    'move2DestMask': jax.nn.one_hot(actions[:,5], sizes[5])
  }

  # Invoke the model to get the logits for every part of the action
  logits = policyNetwork(input)

  # Filter out invalid action types via the masks computed on the given available actions
  actionTypeMask = masks['actionTypeMask']
  stillValidCardMasks = jnp.where(stillValidActionMask, actionTypeMask, jnp.zeros_like(actionTypeMask))
  stillValidActionTypeMask = jnp.where(jnp.any(stillValidCardMasks, axis=0), 0, -jnp.inf)
  actionTypeLogits = logits['actionTypeLogits']
  maskedActionTypeLogits = actionTypeLogits + stillValidActionTypeMask

  # Select an action type given the logits and the mask
  selectedActionType = jax.random.categorical(rngKey, maskedActionTypeLogits)

  # Given the selected action type, narrow down the actions used for creating the next mask, for the card type
  selectedActionOneHot = jax.nn.one_hot(selectedActionType, sizes[0])
  stillValidActionMask = jnp.logical_and(stillValidActionMask, jnp.any(jnp.logical_and(masks['actionTypeMask'], selectedActionOneHot), axis=1, keepdims=True))
  cardMask = masks["cardMask"]
  stillValidCardMasks = jnp.where(stillValidActionMask, cardMask, jnp.zeros_like(cardMask))
  finalCardMask = jnp.where(jnp.any(stillValidCardMasks, axis=0), 0, -jnp.inf)
  cardLogits = logits['cardLogits']
  maskedCardLogits = cardLogits + finalCardMask

  # Select a card given the logits and the mask
  selectedCard = jax.random.categorical(rngKey, maskedCardLogits)

  # Given the selected card type, narrow down the actions used for creating the next mask, for the first move source
  selectedCardOneHot = jax.nn.one_hot(selectedCard, sizes[1])
  stillValidActionMask = jnp.logical_and(stillValidActionMask, jnp.any(jnp.logical_and(masks['cardMask'], selectedCardOneHot), axis=1, keepdims=True))
  move1SourceMask = masks["move1SourceMask"]
  stillValidMove1SourceMasks = jnp.where(stillValidActionMask, move1SourceMask, jnp.zeros_like(move1SourceMask))
  finalMove1SourceMask = jnp.where(jnp.any(stillValidMove1SourceMasks, axis=0), 0, -jnp.inf)
  move1SourceLogits = logits['move1SourceLogits']
  maskedMove1SourceLogits = move1SourceLogits + finalMove1SourceMask

  # Select a first move source given the logits and the mask
  selectedMove1Source = jax.random.categorical(rngKey, maskedMove1SourceLogits)

  # Given the selected first move source, narrow down the actions used for creating the next mask, for the first move destination
  selectedMove1SourceOneHot = jax.nn.one_hot(selectedMove1Source, sizes[2])
  stillValidActionMask = jnp.logical_and(stillValidActionMask, jnp.any(jnp.logical_and(masks['move1SourceMask'], selectedMove1SourceOneHot), axis=1, keepdims=True))
  move1DestMask = masks["move1DestMask"]
  stillValidMove1DestMasks = jnp.where(stillValidActionMask, move1DestMask, jnp.zeros_like(move1DestMask))
  finalMove1DestMask = jnp.where(jnp.any(stillValidMove1DestMasks, axis=0), 0, -jnp.inf)
  move1DestLogits = logits['move1DestLogits']
  maskedMove1DestLogits = move1DestLogits + finalMove1DestMask

  # Select a first move destination given the logits and the mask
  selectedMove1Dest = jax.random.categorical(rngKey, maskedMove1DestLogits)

  # Given the selected first move destination, narrow down the actions used for creating the next mask, for the second move source
  selectedMove1DestOneHot = jax.nn.one_hot(selectedMove1Dest, sizes[3])
  stillValidActionMask = jnp.logical_and(stillValidActionMask, jnp.any(jnp.logical_and(masks['move1DestMask'], selectedMove1DestOneHot), axis=1, keepdims=True))
  move2SourceMask = masks["move2SourceMask"]
  stillValidMove2SourceMasks = jnp.where(stillValidActionMask, move2SourceMask, jnp.zeros_like(move2SourceMask))
  finalMove2SourceMask = jnp.where(jnp.any(stillValidMove2SourceMasks, axis=0), 0, -jnp.inf)
  move2SourceLogits = logits['move2SourceLogits']
  maskedMove2SourceLogits = move2SourceLogits + finalMove2SourceMask

  # Select a second move source given the logits and the mask
  selectedMove2Source = jax.random.categorical(rngKey, maskedMove2SourceLogits)

  # Given the selected second move source, narrow down the actions used for creating the next mask, for the second move destination
  selectedMove2SourceOneHot = jax.nn.one_hot(selectedMove2Source, sizes[4])
  stillValidActionMask = jnp.logical_and(stillValidActionMask, jnp.any(jnp.logical_and(masks['move2SourceMask'], selectedMove2SourceOneHot), axis=1, keepdims=True))
  move2DestMask = masks["move2DestMask"]
  stillValidMove2DestMasks = jnp.where(stillValidActionMask, move2DestMask, jnp.zeros_like(move2DestMask))
  finalMove2DestMask = jnp.where(jnp.any(stillValidMove2DestMasks, axis=0), 0, -jnp.inf)
  move2DestLogits = logits['move2DestLogits']
  maskedMove2DestLogits = move2DestLogits + finalMove2DestMask

  # Select a second move destination given the logits and the mask
  selectedMove2Dest = jax.random.categorical(rngKey, maskedMove2DestLogits)

  # Calculate the overall probability of the selected action by multiplying the probabilities of each part
  actionTypeProbabilities = jax.nn.softmax(maskedActionTypeLogits)
  actionTypeProbability = jnp.sum(jax.lax.stop_gradient(selectedActionOneHot) * actionTypeProbabilities)
  cardProbabilities = jax.nn.softmax(maskedCardLogits)
  cardProbability = jnp.sum(jax.lax.stop_gradient(selectedCardOneHot) * cardProbabilities)
  move1SourceProbabilities = jax.nn.softmax(maskedMove1SourceLogits)
  move1SourceProbability = jnp.sum(jax.lax.stop_gradient(selectedMove1SourceOneHot) * move1SourceProbabilities)
  move1DestProbabilities = jax.nn.softmax(maskedMove1DestLogits)
  move1DestProbability = jnp.sum(jax.lax.stop_gradient(selectedMove1DestOneHot) * move1DestProbabilities)
  move2SourceProbabilities = jax.nn.softmax(maskedMove2SourceLogits)
  move2SourceProbability = jnp.sum(jax.lax.stop_gradient(selectedMove2SourceOneHot) * move2SourceProbabilities)
  move2DestProbabilities = jax.nn.softmax(maskedMove2DestLogits)
  selectedMove2DestOneHot = jax.nn.one_hot(selectedMove2Dest, sizes[5])
  move2DestProbability = jnp.sum(jax.lax.stop_gradient(selectedMove2DestOneHot) * move2DestProbabilities)

  # TODO: At any point, if stillValidActionMask only has 1 True, we can simply return that action
  overallProbability = actionTypeProbability*cardProbability*move1SourceProbability*move1DestProbability*move2SourceProbability*move2DestProbability
  actionTuple = (selectedActionType.astype(int), selectedCard.astype(int), selectedMove1Source.astype(int), selectedMove1Dest.astype(int), selectedMove2Source.astype(int), selectedMove2Dest.astype(int))
  return (-jnp.log(overallProbability), actionTuple)

def getProbabilitiesAndIndex(policyNetwork, input, mask):
  logits = policyNetwork(input)
  maskedLogits = logits + mask
  probabilities = jax.nn.softmax(maskedLogits)
  return probabilities, jax.numpy.argmax(probabilities)

def getValue(valueNetwork, input):
  return valueNetwork(input)

def loadPolicyNetworkFromCheckpoint(checkpointPath):
  abstractModel = nnx.eval_shape(lambda: PolicyNetwork(rngs=nnx.Rngs(0)))
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

def createNewPolicyNetwork(rngs):
  return PolicyNetwork(rngs=rngs)

def createNewValueNetwork(rngs):
  return ValueNetwork(rngs=rngs)

# ================================================================================================
# ================================================================================================
# ================================================================================================

class InferenceClass:
  def __init__(self):
    # Save the checkpoint path
    checkpointPath = pathlib.Path(os.path.join(os.getcwd(), 'checkpoints')) / 'reinforce_smaller_model_1p'
    print(f'Loading model from {checkpointPath}')

    # Load the model from checkpoint
    self.policyNetwork = loadPolicyNetworkFromCheckpoint(checkpointPath / 'policy')
    self.valueNetwork = loadValueNetworkFromCheckpoint(checkpointPath / 'value')

    # Compile the inference functions
    # self.getProbabilityIndex = nnx.jit(getProbabilityAndActionTuple)
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
  def __init__(self, summaryWriter, checkpointName=None):
    self.summaryWriter = summaryWriter
    # Initialize RNG
    # TODO: Find some way to seed the RNG before creating the models
    self.rngs = nnx.Rngs(0, myAdditionalStream=1)

    # Save the checkpoint path
    self.checkpointPath = pathlib.Path(os.path.join(os.getcwd(), 'checkpoints')) / 'latest'

    # Create the model, either from checkpoint or from scratch
    if checkpointName is not None:
      print(f'Loading model from {self.checkpointPath}')
      self.policyNetwork = loadPolicyNetworkFromCheckpoint(self.checkpointPath/'policy')
      self.valueNetwork = loadValueNetworkFromCheckpoint(self.checkpointPath/'value')
    else:
      self.policyNetwork = createNewPolicyNetwork(self.rngs)
      self.valueNetwork = createNewValueNetwork(self.rngs)

    # Initialize the checkpointer
    self.checkpointer = ocp.StandardCheckpointer()

    # Compile the policy network inference function
    # self.getProbabilityActionTupleAndGradient = nnx.value_and_grad(getProbabilityAndActionTuple, argnums=1, has_aux=True)
    self.getProbabilityActionTupleAndGradient = nnx.jit(nnx.value_and_grad(getProbabilityAndActionTuple, argnums=1, has_aux=True))

    # Compile the value network inference function
    self.getValueAndValueGradient = nnx.jit(nnx.value_and_grad(getValue))
  
  def setSeed(self, seed):
    self.rngs = nnx.Rngs(0, myAdditionalStream=seed)

  def logLogitStatistics(self, input, episodeIndex):
    logits = self.policyNetwork(input)
    self.summaryWriter.add_histogram('logits', logits, episodeIndex)

  def getPolicyGradientAndActionTuple(self, data, actions):
    # Pad actions up to the nearest power of 2
    newActionsLength = int(2**math.ceil(math.log2(len(actions))))
    originalSize = len(actions)
    padSize = newActionsLength - len(actions)
    actions = jnp.asarray(actions, dtype=jnp.int32)
    actions = jnp.pad(actions, ((0,padSize),(0,0)), mode='constant', constant_values=0)
    validActions = jnp.concat([jnp.ones(originalSize), jnp.zeros(padSize)], axis=0)
    validActions = validActions[:, None]
    ((logProbability, actionTuple), gradient) = self.getProbabilityActionTupleAndGradient(self.rngs.myAdditionalStream(), self.policyNetwork, data, actions, validActions)
    return gradient, actionTuple

  def getValueGradientAndValue(self, data):
    (value, gradient) = self.getValueAndValueGradient(self.valueNetwork, data)
    return gradient, value

  def initializePolicyOptimizer(self, learningRate):
    # learningRate = optax.linear_schedule(init_value=learningRate, end_value=learningRate/10, transition_steps=1000, transition_begin=7000)
    tx = optax.adam(learning_rate=learningRate)
    self.policyNetworkOptimizer = nnx.Optimizer(self.policyNetwork, tx)

  def initializeValueOptimizer(self, learningRate):
    tx = optax.adam(learning_rate=learningRate)
    self.valueNetworkOptimizer = nnx.Optimizer(self.valueNetwork, tx)

  def loadPolicyOptimizerCheckpoint(self, learningRate, checkpointName):
    tx = optax.adam(learning_rate=learningRate)
    self.policyNetworkOptimizer = nnx.Optimizer(self.policyNetwork, tx)
    abstractOptStateTree = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, self.policyNetworkOptimizer.opt_state)
    checkpointer = ocp.StandardCheckpointer()
    self.policyNetworkOptimizer.opt_state = checkpointer.restore(checkpointName/'policy_optimizer', abstractOptStateTree)
    print('loaded PolicyOptimizerCheckpoint')

  def loadValueOptimizerCheckpoint(self, learningRate, checkpointName):
    tx = optax.adam(learning_rate=learningRate)
    self.valueNetworkOptimizer = nnx.Optimizer(self.valueNetwork, tx)
    abstractOptStateTree = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, self.valueNetworkOptimizer.opt_state)
    checkpointer = ocp.StandardCheckpointer()
    self.valueNetworkOptimizer.opt_state = checkpointer.restore(checkpointName/'value_optimizer', abstractOptStateTree)
    print('loaded ValueOptimizerCheckpoint')

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
    stackedAndPaddedRewards = jnp.stack(paddedRewardsForTrajectories)

    # Stack and pad values
    paddedValuesForTrajectories = [leftPadToMaxSizeWithZeros(jnp.asarray(x)) for x in valuesForTrajectories]
    stackedAndPaddedValues = jnp.stack(paddedValuesForTrajectories)

    # Create a mask
    masks = jnp.asarray([jnp.concatenate([jnp.zeros(newLength-len(x)), jnp.ones(len(x))]) for x in policyGradientsForTrajectories])

    tdMean, tdStddev = updateModels(stackedAndPaddedPolicyGradients, stackedAndPaddedValueGradients, stackedAndPaddedRewards, stackedAndPaddedValues, masks, gamma, self.policyNetworkOptimizer, self.valueNetworkOptimizer)
    self.summaryWriter.add_scalar("episode/tdErrorMean", tdMean, episodeIndex)
    self.summaryWriter.add_scalar("episode/tdErrorStdDev", tdStddev, episodeIndex)
  
  def saveCheckpoint(self):
    policyNetworkPath = self.checkpointPath / 'policy'
    _, policyNetworkState = nnx.split(self.policyNetwork)
    self.checkpointer.save(policyNetworkPath, policyNetworkState, force=True)

    valueNetworkPath = self.checkpointPath / 'value'
    _, valueNetworkState = nnx.split(self.valueNetwork)
    self.checkpointer.save(valueNetworkPath, valueNetworkState, force=True)

    policyOptimizerStatePath = self.checkpointPath / 'policy_optimizer'
    self.checkpointer.save(policyOptimizerStatePath, self.policyNetworkOptimizer.opt_state, force=True)

    valueOptimizerStatePath = self.checkpointPath / 'value_optimizer'
    self.checkpointer.save(valueOptimizerStatePath, self.valueNetworkOptimizer.opt_state, force=True)
    print(f'Saved checkpoints at {self.checkpointPath}')
