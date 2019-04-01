#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt

class SARSAAgent(Agent):
	def __init__(self, learningRate, discountFactor, epsilon, initVals=0.0):
		super(SARSAAgent, self).__init__()
		self.gamma = discountFactor
		self.epsilon = 1
		self.alpha = 1
		self.st_act_rew_triples= []
		self.current_state = None
		self.possibleActions = ['DRIBBLE_UP', 'DRIBBLE_DOWN', 'DRIBBLE_LEFT', 'DRIBBLE_RIGHT', 'KICK']
		self.possible_states = [(i,j) for i in range(5) for j in range(6)]
		self.Q = {(state,action):initVals for state in self.possible_states for action in self.possibleActions}

	def learn(self):
		#States are defined as ((sa_pair),reward)
		current_state , next_state = self.st_act_rew_triples[0], self.st_act_rew_triples[1]
		prior = self.Q[current_state[0]]
		if next_state[0][1]==None:
			td_target = current_state[1] 
		else:
			td_target = current_state[1] + self.gamma * self.Q[next_state[0]]
		td_delta = td_target - self.Q[current_state[0]]
		self.Q[current_state[0]] += self.alpha * td_delta
		del self.st_act_rew_triples[0]
		return self.Q[current_state[0]]-prior

	def act(self):
		action_distribution={}
		for action in self.possibleActions:
			action_distribution[action] = self.Q[(self.current_state,action)]
		maxValue = max(action_distribution.values())
		action_to_take = np.random.choice([k for k,v in action_distribution.items() if v == maxValue])

		if random.random() < self.epsilon:     
			action_to_take = self.possibleActions[random.randint(0,len(self.possibleActions)-1)]
		return action_to_take


	def setState(self, state):
		self.current_state = state

	def setExperience(self, state, action, reward, status, nextState):
		self.st_act_rew_triples.append(((state,action),reward))

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self.epsilon = max(1 - episodeNumber/4750,1e-4)
		self.alpha = max(1 - episodeNumber/4750,1e-4) 
		return self.alpha, self.epsilon	

	def toStateRepresentation(self, state):
		return state[0]

	def reset(self):
		self.st_act_rew_triples=[]

	def setLearningRate(self, learningRate):
		self.alpha=learningRate

	def setEpsilon(self, epsilon):
		self.epsilon=epsilon

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()
	
	numEpisodes = args.numEpisodes
	# Initialize connection to the HFO environment using HFOAttackingPlayer
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a SARSA Agent
	agent = SARSAAgent(1, 0.99,0.1)

	# Run training using SARSA
	numTakenActions = 0 
	goals=0
	final_goal=0
	goals_to_plot=0
	goal_per_100 = [0]
	episode_100 = [0]
	for episode in range(numEpisodes):	
		agent.reset()
		status = 0

		observation = hfoEnv.reset()
		nextObservation = None
		epsStart = True

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
			
			if not epsStart :
				agent.learn()
			else:
				epsStart = False
			if status==1:
				goals+=1
				if episode>4499:
					final_goal+=1
			observation = nextObservation

		agent.setExperience(agent.toStateRepresentation(nextObservation), None, None, None, None)
		agent.learn()
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
	plt.title('SARSA')
	plt.show()
		
	
