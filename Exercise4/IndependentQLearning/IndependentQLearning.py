#!/usr/bin/env python3
# encoding utf-8

import random
import argparse
from DiscreteMARLUtils.Environment import DiscreteMARLEnvironment
from DiscreteMARLUtils.Agent import Agent
from copy import deepcopy
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class IndependentQLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(IndependentQLearningAgent, self).__init__()
		self.Q = dict()
		self.Actions = self.possibleActions
		self.gamma = discountFactor
		self.epsilon = epsilon
		self.alpha = learningRate
		self.initial = initVals
		#Initialize states and rewards
		self.current_state = None
		self.selected_action = None
		self.next_state = None
		self.reward = None
		self.status = 'IN_GAME'

	def setExperience(self, state, action, reward, status, nextState):
		if nextState not in self.Q.keys():
			self.Q[nextState]=dict()
			for act in self.Actions:
				self.Q[nextState][act]=self.initial
		self.next_state = nextState
		self.selected_action = action
		self.reward=reward
		self.status = status
	
	def learn(self):
		current_state,current_action = (self.current_state,self.selected_action)
		reward, nextState = self.reward, self.next_state
		prior = self.Q[current_state][current_action]
		if self.status=='IN_GAME':
			action_distribution = self.Q[nextState]
			best_action = max(action_distribution.keys(), key=(lambda key: action_distribution[key]))
			td_target = reward + self.gamma * action_distribution[best_action]
		else:
			td_target = reward
		td_delta = td_target - self.Q[current_state][current_action]
		self.Q[current_state][current_action] += self.alpha * td_delta
		return self.Q[current_state][current_action]-prior

	def act(self):
		action_distribution = self.Q[self.current_state]
		maxValue = max(action_distribution.values())
		action_to_take = np.random.choice([k for k,v in action_distribution.items() if v == maxValue])

		if random.random() < self.epsilon:      # e-greedy action
			action_to_take = self.Actions[random.randint(0,len(self.Actions)-1)]
		return action_to_take

	def toStateRepresentation(self, State):
		if type(State)== str:
			return State
		attacker1 = (State[0][0][0],State[0][0][1])
		attacker2 = (State[0][1][0],State[0][1][1])
		ball = (State[2][0][0],State[2][0][1])
		defender = (State[1][0][0],State[1][0][1])
		return (attacker1,attacker2,ball,defender)

	def setState(self, state):
		if state not in self.Q.keys():
			self.Q[state]=dict()
			for action in self.Actions:
				self.Q[state][action]=self.initial
		self.current_state = state

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon
		
	def setLearningRate(self, learningRate):
		self.alpha = learningRate 
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self.alpha = max((1-episodeNumber/50000),1e-5)
		self.epsilon = max((1-episodeNumber/50000),1e-6)	
		return self.alpha,self.epsilon

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--numOpponents', type=int, default=1)
	parser.add_argument('--numAgents', type=int, default=2)
	parser.add_argument('--numEpisodes', type=int, default=50000)

	args=parser.parse_args()

	MARLEnv = DiscreteMARLEnvironment(numOpponents = args.numOpponents, numAgents = args.numAgents)
	agents = []
	goal_per_1000 = [0]
	episode_1000 = [0]
	goals=0
	total_goals=0
	for i in range(args.numAgents):
		agent = IndependentQLearningAgent(learningRate = 0.1, discountFactor = 0.9, epsilon = 1.0)
		agents.append(agent)

	numEpisodes = args.numEpisodes
	numTakenActions = 0
	
	for episode in range(numEpisodes):	
		status = ["IN_GAME","IN_GAME","IN_GAME"]
		observation = MARLEnv.reset()
		totalReward = 0.0
		timeSteps = 0
		while status[0]=="IN_GAME":
			for agent in agents:
				learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
				agent.setEpsilon(epsilon)
				agent.setLearningRate(learningRate)
			actions = []
			stateCopies = []
			for agentIdx in range(args.numAgents):
				obsCopy = deepcopy(observation[agentIdx])
				stateCopies.append(obsCopy)
				agents[agentIdx].setState(agent.toStateRepresentation(obsCopy))
				actions.append(agents[agentIdx].act())
			numTakenActions += 1
			nextObservation, reward, done, status = MARLEnv.step(actions)

			for agentIdx in range(args.numAgents):
				agents[agentIdx].setExperience(agent.toStateRepresentation(stateCopies[agentIdx]), actions[agentIdx], reward[agentIdx], 
					status[agentIdx], agent.toStateRepresentation(nextObservation[agentIdx]))
				agents[agentIdx].learn()
				
			observation = nextObservation
		if reward[0]==1: 
			goals+=1
			total_goals+=1
		if ((episode+1) % 1000)==0:
			goal_per_1000.append(goals)
			episode_1000.append(episode)
			goals=0
		print(episode,' : ',total_goals)
	plt.plot(episode_1000,goal_per_1000)
	plt.xlabel('Episodes')
	plt.ylabel('Goals per 1000 episodes')
	plt.title('IDG')
	plt.show()
