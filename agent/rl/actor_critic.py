import os

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import math
import optax
import sys
import pathlib
from flax import nnx
from functools import partial

# ==========================================================================================

class PolicyNetwork(nnx.Module):
  def __init__(self, rngs):
    inFeatureSize = 1+11*5+2*4*67
    stateLinearOutputSize = 512
    actionTypeCount = 5
    cardCount = 11
    positionCount = 67
    self.stateLinear = nnx.Linear(in_features=inFeatureSize, out_features=stateLinearOutputSize, rngs=rngs)

    self.actionTypeLinear = nnx.Linear(
        in_features=stateLinearOutputSize,
        out_features=actionTypeCount,
        rngs=rngs)
    self.cardLinear = nnx.Linear(
        in_features=stateLinearOutputSize + actionTypeCount,
        out_features=cardCount,
        rngs=rngs)
    self.move1SourceLinear = nnx.Linear(
        in_features=stateLinearOutputSize + actionTypeCount + cardCount,
        out_features=positionCount,
        rngs=rngs)
    self.move1DestLinear = nnx.Linear(
        in_features=stateLinearOutputSize + actionTypeCount + cardCount + positionCount,
        out_features=positionCount,
        rngs=rngs)
    self.move2SourceLinear = nnx.Linear(
        in_features=stateLinearOutputSize + actionTypeCount + cardCount + 2*positionCount,
        out_features=positionCount,
        rngs=rngs)
    self.move2DestLinear = nnx.Linear(
        in_features=stateLinearOutputSize + actionTypeCount + cardCount + 3*positionCount,
        out_features=positionCount,
        rngs=rngs)

  def __call__(self, x):
    x = self.stateLinear(x)
    return jax.nn.relu(x)

def createNewPolicyNetwork(rngs):
  return PolicyNetwork(rngs=rngs)

def loadPolicyNetworkFromCheckpoint(checkpointPath):
  abstractModel = nnx.eval_shape(lambda: PolicyNetwork(rngs=nnx.Rngs(0)))
  graphdef, abstractState = nnx.split(abstractModel)
  checkpointer = ocp.StandardCheckpointer()
  stateRestored = checkpointer.restore(checkpointPath, abstractState)
  return nnx.merge(graphdef, stateRestored)

# ==========================================================================================

class ValueNetwork(nnx.Module):
  def __init__(self, rngs):
    inFeatureSize = 1+11*5+2*4*67
    self.linear1 = nnx.Linear(in_features=inFeatureSize, out_features=128, rngs=rngs)
    self.linear2 = nnx.Linear(in_features=128, out_features=1, rngs=rngs)

  def __call__(self, x):
    x = self.linear1(x)
    x = jax.nn.relu(x)
    x = self.linear2(x)
    return x.reshape()

def createNewValueNetwork(rngs):
  return ValueNetwork(rngs=rngs)

def loadValueNetworkFromCheckpoint(checkpointPath):
  abstractModel = nnx.eval_shape(lambda: ValueNetwork(rngs=nnx.Rngs(0)))
  graphdef, abstractState = nnx.split(abstractModel)
  checkpointer = ocp.StandardCheckpointer()
  stateRestored = checkpointer.restore(checkpointPath, abstractState)
  return nnx.merge(graphdef, stateRestored)

# ==========================================================================================

def getProbabilityAndActionTuple(rngKey, policyNetwork, observation, actions, stillValidActionMask):
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
  stateEmbedding = policyNetwork(observation)

  # Filter out invalid action types via the masks computed on the given available actions
  actionTypeMask = masks['actionTypeMask']
  stillValidCardMasks = jnp.where(stillValidActionMask, actionTypeMask, jnp.zeros_like(actionTypeMask))
  stillValidActionTypeMask = jnp.where(jnp.any(stillValidCardMasks, axis=0), 0, -jnp.inf)
  actionTypeLogits = policyNetwork.actionTypeLinear(stateEmbedding)
  maskedActionTypeLogits = actionTypeLogits + stillValidActionTypeMask

  # Select an action type given the logits and the mask
  selectedActionType = jax.random.categorical(rngKey, maskedActionTypeLogits)

  # Given the selected action type, narrow down the actions used for creating the next mask, for the card type
  selectedActionOneHot = jax.nn.one_hot(selectedActionType, sizes[0])
  stillValidActionMask = jnp.logical_and(stillValidActionMask, jnp.any(jnp.logical_and(masks['actionTypeMask'], selectedActionOneHot), axis=1, keepdims=True))
  cardMask = masks["cardMask"]
  stillValidCardMasks = jnp.where(stillValidActionMask, cardMask, jnp.zeros_like(cardMask))
  finalCardMask = jnp.where(jnp.any(stillValidCardMasks, axis=0), 0, -jnp.inf)
  stateEmbeddingAndAction = jnp.concatenate([stateEmbedding, selectedActionOneHot])
  cardLogits = policyNetwork.cardLinear(stateEmbeddingAndAction)
  maskedCardLogits = cardLogits + finalCardMask

  # Select a card given the logits and the mask
  selectedCard = jax.random.categorical(rngKey, maskedCardLogits)

  # Given the selected card type, narrow down the actions used for creating the next mask, for the first move source
  selectedCardOneHot = jax.nn.one_hot(selectedCard, sizes[1])
  stillValidActionMask = jnp.logical_and(stillValidActionMask, jnp.any(jnp.logical_and(masks['cardMask'], selectedCardOneHot), axis=1, keepdims=True))
  move1SourceMask = masks["move1SourceMask"]
  stillValidMove1SourceMasks = jnp.where(stillValidActionMask, move1SourceMask, jnp.zeros_like(move1SourceMask))
  finalMove1SourceMask = jnp.where(jnp.any(stillValidMove1SourceMasks, axis=0), 0, -jnp.inf)
  stateEmbeddingAndAction = jnp.concatenate([stateEmbeddingAndAction, selectedCardOneHot])
  move1SourceLogits = policyNetwork.move1SourceLinear(stateEmbeddingAndAction)
  maskedMove1SourceLogits = move1SourceLogits + finalMove1SourceMask

  # Select a first move source given the logits and the mask
  selectedMove1Source = jax.random.categorical(rngKey, maskedMove1SourceLogits)

  # Given the selected first move source, narrow down the actions used for creating the next mask, for the first move destination
  selectedMove1SourceOneHot = jax.nn.one_hot(selectedMove1Source, sizes[2])
  stillValidActionMask = jnp.logical_and(stillValidActionMask, jnp.any(jnp.logical_and(masks['move1SourceMask'], selectedMove1SourceOneHot), axis=1, keepdims=True))
  move1DestMask = masks["move1DestMask"]
  stillValidMove1DestMasks = jnp.where(stillValidActionMask, move1DestMask, jnp.zeros_like(move1DestMask))
  finalMove1DestMask = jnp.where(jnp.any(stillValidMove1DestMasks, axis=0), 0, -jnp.inf)
  stateEmbeddingAndAction = jnp.concatenate([stateEmbeddingAndAction, selectedMove1SourceOneHot])
  move1DestLogits = policyNetwork.move1DestLinear(stateEmbeddingAndAction)
  maskedMove1DestLogits = move1DestLogits + finalMove1DestMask

  # Select a first move destination given the logits and the mask
  selectedMove1Dest = jax.random.categorical(rngKey, maskedMove1DestLogits)

  # Given the selected first move destination, narrow down the actions used for creating the next mask, for the second move source
  selectedMove1DestOneHot = jax.nn.one_hot(selectedMove1Dest, sizes[3])
  stillValidActionMask = jnp.logical_and(stillValidActionMask, jnp.any(jnp.logical_and(masks['move1DestMask'], selectedMove1DestOneHot), axis=1, keepdims=True))
  move2SourceMask = masks["move2SourceMask"]
  stillValidMove2SourceMasks = jnp.where(stillValidActionMask, move2SourceMask, jnp.zeros_like(move2SourceMask))
  finalMove2SourceMask = jnp.where(jnp.any(stillValidMove2SourceMasks, axis=0), 0, -jnp.inf)
  stateEmbeddingAndAction = jnp.concatenate([stateEmbeddingAndAction, selectedMove1DestOneHot])
  move2SourceLogits = policyNetwork.move2SourceLinear(stateEmbeddingAndAction)
  maskedMove2SourceLogits = move2SourceLogits + finalMove2SourceMask

  # Select a second move source given the logits and the mask
  selectedMove2Source = jax.random.categorical(rngKey, maskedMove2SourceLogits)

  # Given the selected second move source, narrow down the actions used for creating the next mask, for the second move destination
  selectedMove2SourceOneHot = jax.nn.one_hot(selectedMove2Source, sizes[4])
  stillValidActionMask = jnp.logical_and(stillValidActionMask, jnp.any(jnp.logical_and(masks['move2SourceMask'], selectedMove2SourceOneHot), axis=1, keepdims=True))
  move2DestMask = masks["move2DestMask"]
  stillValidMove2DestMasks = jnp.where(stillValidActionMask, move2DestMask, jnp.zeros_like(move2DestMask))
  finalMove2DestMask = jnp.where(jnp.any(stillValidMove2DestMasks, axis=0), 0, -jnp.inf)
  stateEmbeddingAndAction = jnp.concatenate([stateEmbeddingAndAction, selectedMove2SourceOneHot])
  move2DestLogits = policyNetwork.move2DestLinear(stateEmbeddingAndAction)
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

# ==========================================================================================

@jax.jit
def createObservationForModel(observation):
  cardCount = 11
  positionCount = 67
  haveOpponentAcrossBoard = observation[0].reshape(1)

  # Represent the cards
  #  This representation is a one-hot encoding for each card, a 5x11 matrix.
  # cardRepresentation = jnp.concat(jax.nn.one_hot(observation[1:1+5], cardCount))
  #  This representation is kind of a N-hot, representing the number of cards of each type, where N is the number of cards of the given type. This is an 11x5 matrix.
  cardRepresentation = jnp.concat((jnp.bincount(observation[1:1+5], length=cardCount) > jnp.arange(5).reshape(5,1)).astype(jnp.float32).transpose())

  selfPositions = jnp.concat(jax.nn.one_hot(observation[1+5:1+5+4], positionCount))
  opponentAcrossBoardPositions = jnp.concat(jax.nn.one_hot(observation[1+5+4:1+5+4+4], positionCount))
  return jnp.concat([haveOpponentAcrossBoard, cardRepresentation, selfPositions, opponentAcrossBoardPositions])

def padActionsAndGetMask(validActions):
  # Pad actions up to the nearest power of 2
  newActionsLength = int(2**math.ceil(math.log2(len(validActions))))
  originalSize = len(validActions)
  padSize = newActionsLength - len(validActions)
  validActions = jnp.asarray(validActions, dtype=jnp.int32)
  paddedActions = jnp.pad(validActions, ((0,padSize),(0,0)), mode='constant', constant_values=0)
  validActionMask = jnp.concat([jnp.ones(originalSize), jnp.zeros(padSize)], axis=0)
  validActionMask = validActionMask[:, None]
  return paddedActions, validActionMask

def updateAndReturnLoss(policyNetwork, valueNetwork, policyOptimizer, valueOptimizer, lastObservation, reward, currentObservation, rngKey, paddedActions, validActionMask, gamma):
  getProbabilityActionTupleAndGradient = nnx.jit(nnx.value_and_grad(getProbabilityAndActionTuple, argnums=1, has_aux=True))
  (probability, actionTuple), policyGradient = getProbabilityActionTupleAndGradient(rngKey, policyNetwork, lastObservation, paddedActions, validActionMask)

  # Calculate the advantage
  lastValue = valueNetwork(lastObservation)
  currentValue = valueNetwork(currentObservation)
  advantage = reward + gamma * currentValue - lastValue

  # Scale the policy gradient by the advantage
  policyGradient = jax.tree.map(lambda g: g * advantage, policyGradient)

  # Update policy
  #  We want to do gradient ascent on the log probability.
  #  The function which we computed the gradient from already
  #  returns the negative log probability, so using the optimizer
  #  for gradient descent ends up doing gradient ascent.
  policyOptimizer.update(policyGradient)

  def valueLoss(valueNetwork, lastObservation, currentObservation, reward):
    lastValue = valueNetwork(lastObservation)
    # jax.debug.print(f'Last value: {lastValue}')
    value = valueNetwork(currentObservation)
    # jax.debug.print(f'Value: {value}')
    target = reward + gamma * jax.lax.stop_gradient(value)
    return jnp.mean((lastValue - target)**2) / 2.0

  # Take the gradient of and update the value network
  loss, valueGradient = nnx.value_and_grad(valueLoss)(valueNetwork, lastObservation, currentObservation, reward)
  valueOptimizer.update(valueGradient)
  return loss

# ==========================================================================================

class InferenceClass:
  def __init__(self, trainingUtil=None, checkpointName=None):
    self.rngs = nnx.Rngs(0, myAdditionalStream=1)

    if trainingUtil is not None:
      # Copy model from the training util
      self.policyNetwork = nnx.clone(trainingUtil.policyNetwork)
    elif checkpointName is not None:
      checkpointPath = pathlib.Path(os.path.join(os.getcwd(), 'checkpoints')) / checkpointName
      print(f'Loading model from {checkpointPath}')
      self.policyNetwork = loadPolicyNetworkFromCheckpoint(checkpointPath/'policy')

    # Compile the inference functions
    self.getStochasticActionTuple = nnx.jit(getProbabilityAndActionTuple)

  def setSeed(self, seed):
    self.rngs = nnx.Rngs(0, myAdditionalStream=seed)

  def getBestAction(self, observation, actions):
    paddedActions, validActionMask = padActionsAndGetMask(actions)
    observation = createObservationForModel(jnp.asarray(observation))
    _, action = self.getStochasticActionTuple(self.rngs.myAdditionalStream(), self.policyNetwork, observation, paddedActions, validActionMask)
    return action

# ==========================================================================================

class TrainingUtilClass:
  def __init__(self, summaryWriter, policyNetworkLearningRate, valueNetworkLearningRate, checkpointDirectoryName, restoreFromCheckpoint):
    self.summaryWriter = summaryWriter
    # Initialize RNG
    # TODO: Find some way to seed the RNG before creating the models
    self.rngs = nnx.Rngs(0, myAdditionalStream=1)

    # Save the checkpoint path
    self.checkpointBasePath = pathlib.Path(os.path.join(os.getcwd(), 'checkpoints')) / checkpointDirectoryName

    # Initialize the checkpointer
    self.checkpointer = ocp.StandardCheckpointer()

    if restoreFromCheckpoint:
      self.policyNetwork = loadPolicyNetworkFromCheckpoint(self.getCheckpointPath()/'policy')
      self.valueNetwork = loadValueNetworkFromCheckpoint(self.getCheckpointPath()/'value')
      self.loadPolicyOptimizerCheckpoint(policyNetworkLearningRate)
      self.loadValueOptimizerCheckpoint(valueNetworkLearningRate)
    else:
      # Create the model
      self.policyNetwork = createNewPolicyNetwork(self.rngs)
      self.valueNetwork = createNewValueNetwork(self.rngs)

      # Create the policy optimizer
      # policyNetworkLearningRate = optax.linear_schedule(init_value=policyNetworkLearningRate, end_value=policyNetworkLearningRate/10, transition_steps=1000, transition_begin=7000)
      tx = optax.adam(learning_rate=policyNetworkLearningRate)
      self.policyNetworkOptimizer = nnx.Optimizer(self.policyNetwork, tx)

      # Create the value optimizer
      tx = optax.adam(learning_rate=valueNetworkLearningRate)
      self.valueNetworkOptimizer = nnx.Optimizer(self.valueNetwork, tx)

    # Compile the policy network inference functions
    self.getProbabilityActionTupleAndGradient = nnx.jit(nnx.value_and_grad(getProbabilityAndActionTuple, argnums=1, has_aux=True))
    self.getProbabilityAndActionTuple = nnx.jit(getProbabilityAndActionTuple)
    self.jittedUpdate = nnx.jit(updateAndReturnLoss)

  def setSeed(self, seed):
    self.rngs = nnx.Rngs(0, myAdditionalStream=seed)

  def getCheckpointPath(self):
    return self.checkpointBasePath

  def saveCheckpoint(self):
    checkpointPath = self.getCheckpointPath()
    policyNetworkPath = checkpointPath / 'policy'
    _, policyNetworkState = nnx.split(self.policyNetwork)
    self.checkpointer.save(policyNetworkPath, policyNetworkState, force=True)

    valueNetworkPath = checkpointPath / 'value'
    _, valueNetworkState = nnx.split(self.valueNetwork)
    self.checkpointer.save(valueNetworkPath, valueNetworkState, force=True)

    policyOptimizerStatePath = checkpointPath / 'policy_optimizer'
    self.checkpointer.save(policyOptimizerStatePath, nnx.state(self.policyNetworkOptimizer), force=True)

    valueOptimizerStatePath = checkpointPath / 'value_optimizer'
    self.checkpointer.save(valueOptimizerStatePath, nnx.state(self.valueNetworkOptimizer), force=True)
    print(f'Saved checkpoints at {checkpointPath}')

  def loadPolicyOptimizerCheckpoint(self, learningRate):
    tx = optax.adam(learning_rate=learningRate)
    self.policyNetworkOptimizer = nnx.Optimizer(self.policyNetwork, tx)
    abstractOptStateTree = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, nnx.state(self.policyNetworkOptimizer))
    checkpointer = ocp.StandardCheckpointer()
    optimizerState = checkpointer.restore(self.getCheckpointPath()/'policy_optimizer', abstractOptStateTree)
    nnx.update(self.policyNetworkOptimizer, optimizerState)

  def loadValueOptimizerCheckpoint(self, learningRate):
    tx = optax.adam(learning_rate=learningRate)
    self.valueNetworkOptimizer = nnx.Optimizer(self.valueNetwork, tx)
    abstractOptStateTree = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, nnx.state(self.valueNetworkOptimizer))
    checkpointer = ocp.StandardCheckpointer()
    optimizerState = checkpointer.restore(self.getCheckpointPath()/'value_optimizer', abstractOptStateTree)
    nnx.update(self.valueNetworkOptimizer, optimizerState)

  def getActionTupleAndKeyUsed(self, observation, actions):
    paddedActions, validActionMask = padActionsAndGetMask(actions)
    observation = createObservationForModel(jnp.asarray(observation))
    rngKey = self.rngs.myAdditionalStream()
    (logProbability, actionTuple) = self.getProbabilityAndActionTuple(rngKey, self.policyNetwork, observation, paddedActions, validActionMask)
    return actionTuple, rngKey

  def train(self, lastObservation, reward, currentObservation, rngKey, lastValidActionsArray, gamma):
    # Convert observations to a format ready for the networks
    lastObservation = createObservationForModel(jnp.asarray(lastObservation))
    currentObservation = createObservationForModel(jnp.asarray(currentObservation))

    # Take the gradient of the policy, at the last observation
    paddedActions, validActionMask = padActionsAndGetMask(lastValidActionsArray)

    # Call jitted update function
    loss = self.jittedUpdate(self.policyNetwork, self.valueNetwork, self.policyNetworkOptimizer, self.valueNetworkOptimizer, lastObservation, reward, currentObservation, rngKey, paddedActions, validActionMask, gamma)

    return loss
