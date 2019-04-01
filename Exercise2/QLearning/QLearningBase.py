#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt

class QLearningAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(QLearningAgent, self).__init__()
		self.gamma = discountFactor
		self.epsilon = epsilon
		self.alpha = learningRate
		self.st_act_rew_triples=[]
		self.current_state = None
		self.goals = 0
		self.possibleActions = ['DRIBBLE_UP', 'DRIBBLE_DOWN', 'DRIBBLE_LEFT', 'DRIBBLE_RIGHT', 'KICK']
		self.possible_states = [(i,j) for i in range(5) for j in range(6)]
		self.Q = {(state,action):initVals for state in self.possible_states for action in self.possibleActions}
		self.status = 0 

		

	def learn(self):
		current_state_action , reward_next_state = self.st_act_rew_triples[0], self.st_act_rew_triples[1]
		prior = self.Q[current_state_action]
		if self.status==0:
			action_distribution = {action:self.Q[(reward_next_state[1],action)] for action in self.possibleActions}
			best_action = max(action_distribution.keys(), key=(lambda key: action_distribution[key]))
			td_target = reward_next_state[0] + self.gamma * action_distribution[best_action]
		else:
			td_target = reward_next_state[0]
		td_delta = td_target - self.Q[current_state_action]
		self.Q[current_state_action] += self.alpha * td_delta
		return self.Q[current_state_action]-prior
		

	def act(self):
		action_distribution={}
		for action in self.possibleActions:
			action_distribution[action] = self.Q[(self.current_state,action)]
		#Greedy policy
		maxValue = max(action_distribution.values())
		action_to_take = np.random.choice([k for k,v in action_distribution.items() if v == maxValue])
		#e-greedy policy
		if random.random() < self.epsilon:      
			action_to_take = self.possibleActions[random.randint(0,len(self.possibleActions)-1)]
		return action_to_take

	def toStateRepresentation(self, state):
		return state[0]

	def setState(self, state):
		self.current_state = state

	def setExperience(self, state, action, reward, status, nextState):
		self.st_act_rew_triples=[(state,action),(reward,nextState)]
		self.status = status
		if status == 1: self.goals +=1

	def setLearningRate(self, learningRate):
		self.alpha = learningRate

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def reset(self):
		self.st_act_rew_triples = []
		
	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self.epsilon = max(1 - episodeNumber/4750,1e-4) 
		if episodeNumber==0:
			self.alpha=0.1
		elif episodeNumber>4499:
			self.alpha=0.01
		return self.alpha,self.epsilon	


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=5000)

	args=parser.parse_args()

	# Initialize connection with the HFO server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Q-Learning Agent
	agent = QLearningAgent(learningRate = 0.1, discountFactor = 0.99, epsilon = 1.0)
	numEpisodes = args.numEpisodes

	# Run training using Q-Learning
	numTakenActions = 0 
	goals=0
	final_goal=0
	goals_to_plot = 0
	goal_per_100 = []
	episode_100 = []
	for episode in range(numEpisodes):
		status = 0
		observation = hfoEnv.reset()
		
		while status==0:
			learningRate, epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			agent.setLearningRate(learningRate)
			
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			update = agent.learn()
			if status==1:
				goals+=1
				if episode>4499:
					final_goal+=1
			observation = nextObservation
		print(episode,':',goals)
		if status==1: 
			goals_to_plot+=1
		if (episode+1) % 100==0:
			goal_per_100.append(goals_to_plot)
			episode_100.append(episode)
			goals_to_plot=0
	print('Scored :',final_goal,'in the last 500 episodes')
	plt.plot(episode_100,goal_per_100)
	plt.xlabel('Episodes')
	plt.ylabel('Goals per 100 episodes')
	plt.ylim(0.920)
	plt.title('QLearning')
	plt.show()
		
	