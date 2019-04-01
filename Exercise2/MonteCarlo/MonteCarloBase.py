#!/usr/bin/env python3
# encoding utf-8

from DiscreteHFO.HFOAttackingPlayer import HFOAttackingPlayer
from DiscreteHFO.Agent import Agent
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt

class MonteCarloAgent(Agent):
	def __init__(self, discountFactor, epsilon, initVals=0.0):
		super(MonteCarloAgent, self).__init__()
		self.gamma = discountFactor
		self.epsilon = 1
		self.episode = []
		self.goals=0
		self.G = 0
		self.current_state = None
		self.possibleActions = ['DRIBBLE_UP', 'DRIBBLE_DOWN', 'DRIBBLE_LEFT', 'DRIBBLE_RIGHT', 'KICK']
		self.possible_states = [(i,j) for i in range(5) for j in range(6)]
		self.Q = {(state,action):initVals for state in self.possible_states for action in self.possibleActions}
		self.returns_Sum = {(state,action):initVals for state in self.possible_states for action in self.possibleActions}
		self.returns_Count = {(state,action):initVals for state in self.possible_states for action in self.possibleActions}	

		
	def learn(self):
		#State_action_pairs = set([(x[0],x[1]) for x in self.episode])
		output = [(x[0],x[1]) for x in self.episode]
		State_action_pairs = list(dict.fromkeys(output))
		output=[]
		for state,action in State_action_pairs:
			sa_pair = (state,action)
			first_occurence_idx = next(i for i,x in enumerate(self.episode) if x[0] == state and x[1] == action)
			G = sum([x[2]*(self.gamma**i) for i,x in enumerate(self.episode[first_occurence_idx:])])
			self.returns_Sum[sa_pair]+=G
			self.returns_Count[sa_pair]+=1
			self.Q[sa_pair] = self.returns_Sum[sa_pair] / self.returns_Count[sa_pair]
			output.append(self.Q[sa_pair])
		#	self.policy[state] = self.create_e_greedy_policy(state,self.epsilon)
		return self.Q,output

	def toStateRepresentation(self, state):
		return state[0]

	def setExperience(self, state, action, reward, status, nextState):			
		self.episode.append((state,action,reward))
		if status == 1:
			self.goals += 1

	def setState(self, state): 
		self.current_state = state

	def reset(self):
		self.episode = []

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

	def setEpsilon(self, epsilon):
		self.epsilon = epsilon

	def computeHyperparameters(self, numTakenActions, episodeNumber):
		self.epsilon =max(1-episodeNumber/4750,1e-5)
		return self.epsilon


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=int, default=0)
	parser.add_argument('--numOpponents', type=int, default=0)
	parser.add_argument('--numTeammates', type=int, default=0)
	parser.add_argument('--numEpisodes', type=int, default=500)

	args=parser.parse_args()

	#Init Connections to HFO Server
	hfoEnv = HFOAttackingPlayer(numOpponents = args.numOpponents, numTeammates = args.numTeammates, agentId = args.id)
	hfoEnv.connectToServer()

	# Initialize a Monte-Carlo Agent
	agent = MonteCarloAgent(discountFactor = 0.99, epsilon = 1)
	numEpisodes = args.numEpisodes
	numTakenActions = 0
	goals=0
	final_goals=0
	goal_per_100 = [0]
	episode_100= [0]
	goals_to_plot = 0
	# Run training Monte Carlo Method
	for episode in range(numEpisodes):	
		agent.reset()
		observation = hfoEnv.reset()
		status = 0

		while status==0:
			epsilon = agent.computeHyperparameters(numTakenActions, episode)
			agent.setEpsilon(epsilon)
			obsCopy = observation.copy()
			agent.setState(agent.toStateRepresentation(obsCopy))
			action = agent.act()
			numTakenActions += 1
			nextObservation, reward, done, status = hfoEnv.step(action)
			agent.setExperience(agent.toStateRepresentation(obsCopy), action, reward, status, agent.toStateRepresentation(nextObservation))
			if status==1:
				goals+=1
				if episode>4499:
					final_goals+=1
			observation = nextObservation
		Q_function,_=agent.learn()
		print(episode,':',goals)
		if status==1: 
			goals_to_plot+=1
		if (episode+1) % 100==0:
			goal_per_100.append(goals_to_plot)
			episode_100.append(episode)
			goals_to_plot=0
	print('Scored :',final_goals,'in the last 500 episodes')
	plt.plot(episode_100,goal_per_100)
	plt.xlabel('Episodes')
	plt.ylabel('Goals per 100 episodes')
	plt.ylim(0.920)
	plt.title('Monte Carlo')
	plt.show()
		
	

		
