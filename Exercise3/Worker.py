import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sys import maxsize
from Networks import ValueNetwork
from torch.autograd import Variable
from Environment import HFOEnv
import random
import numpy as np 
import os
import copy


def train(idx, val_network, target_value_network, optimizer, lock, counter,timesteps_per_process,results_dir):
	
	# This runs a random agent

	episodeNumber = 0
	discountFactor = 0.99
	f = open(os.path.join(results_dir, 'worker_%d.out'%idx), 'w')
	#Each worker writes to a file to check if my network runs correctly
	columns = '{0:<10} {1:<8} {2:<10} {3:<12} {4:<20} {5:<15} {6:<15}\n'
	f.write(columns.format('Episode','Status','Steps','Total steps','Avg steps to goal','Total Goals','Counter'))
	
	hfoEnv = HFOEnv(numTeammates=0, numOpponents=1, port=6000+(idx+5)*10, seed=idx)
	hfoEnv.connectToServer()
	asyc_update = 10
	copy_freq = 10000
	total_goals = 0
	total_steps = 0
	totalEpisodes = timesteps_per_process // 500
	save_model = False
	copy_model = False
	for episodeNumber in range(totalEpisodes):
		done = False
		epsilon = epsilon_annealing(idx,episodeNumber,totalEpisodes)
		state = hfoEnv.reset()
		timesteps = 0 
		while timesteps<500:
			lock.acquire()
			counter.value += 1
			locked_counter = counter.value
			if (locked_counter % 1e6) == 0:
				save_model = True
			if (locked_counter % copy_freq) == 0:
				copy_model = True
			lock.release()
			timesteps+=1	

			obs_tensor = torch.Tensor(state).unsqueeze(0)
			Q, act = compute_val(val_network, obs_tensor,idx,epsilon)
			act = hfoEnv.possibleActions[act]
			newObservation, reward, done, status, info = hfoEnv.step(act)

			Q_target = computeTargets(reward,torch.Tensor(newObservation).unsqueeze(0),discountFactor,done,target_value_network)

			loss_function = nn.MSELoss()
			loss = loss_function(Q_target,Q)
			loss.backward()
			
			if timesteps % asyc_update ==0 or done:
				with lock:
					optimizer.step()
					optimizer.zero_grad()		

			if copy_model:
				hard_copy(target_value_network, val_network)
				copy_model = False
			if save_model:
				updating_learning_rate(optimizer,locked_counter)
				saveModelNetwork(target_value_network,os.path.join(results_dir, 'params_%d' % int(locked_counter / 1e6)))
				save_model = False
			state = newObservation
			if done:
				break
		if status==1:
			total_steps += timesteps
			total_goals += 1
		else:
			total_steps += 500
		f.write(columns.format(episodeNumber, status, timesteps, total_steps, '%.1f'%(total_steps/(episodeNumber+1)), total_goals, locked_counter))
		f.flush()

		

def computeTargets(reward, nextObservation, discountFactor, done, targetNetwork):
	if not done:
		output_qs = targetNetwork(nextObservation)
		target = reward + discountFactor*torch.max(output_qs[0],dim=0)[0]
	else:
		target = torch.Tensor([reward])
	return target

def computePrediction(state, action, valueNetwork):
	output_qs = valueNetwork(state)
	return output_qs[0][action]

#Fuction used to linearly decrease epsilon 
def epsilon_annealing(idx,current,total):
	i = min(idx,10)
	init_value = 1-0.1*i
	if current==0:
		return init_value
	else:
		denom = total // init_value
		epsilon = max(init_value-current/denom,1e-4)
	return epsilon
	
def compute_val(valueNetwork,state,idx,epsilon):
	
	output_qs = valueNetwork(state)
	_,act =torch.max(output_qs[0],dim=0)
	act = act.item()

	if random.random() < epsilon:
		act = random.randint(0,3)
	return output_qs[0][act],act

# Function to save parameters of a neural network in pytorch.
def saveModelNetwork(model, strDirectory):
        torch.save(model.state_dict(), strDirectory)

# Set the target parameters to be exactly the same as the source
def hard_copy(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(param.data)

#Function to decrease the learning rate 
def updating_learning_rate(optimizer, counter):
	if counter>=6e6:
		learningRate = 1e-9
	elif counter>=5e6:
		learningRate = 1e-8
	elif counter >= 4e6:
		learningRate= 1e-7
	elif counter >= 3e6:
		learningRate = 1e-6
	elif counter >= 2e6:
		learningRate = 0.5e-5
	elif counter>=1e6:
		learningRate = 1e-5
	for param_group in optimizer.param_groups:
		param_group['lr'] = learningRate

