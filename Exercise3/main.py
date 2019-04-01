#!/usr/bin/env python3
# encoding utf-8


from Environment import HFOEnv
import torch
import torch.multiprocessing as mp
from Networks import ValueNetwork
from Worker import train,hard_copy,saveModelNetwork
import argparse
from SharedAdam import SharedAdam
import copy
import os

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train ')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (better to use annealing)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='Batches before logging training status')
parser.add_argument('--num_processes', type=int, default=8, metavar='N',
                    help='Parallel processes to run (default: 8)')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='enables CUDA training')


# Use this script to handle arguments and 
# initialize important components of your experiment.
# These might include important parameters for your experiment,
# your models, torch's multiprocessing methods, etc.

if __name__ == "__main__" :

	# Example on how to initialize global locks for processes
	# and counters.

	numOpponents = 1
	args = parser.parse_args()	
	results_dir = './Network_parameters' 
	if not os.path.exists(results_dir):
		os.makedirs(results_dir)

	counter = mp.Value('i', 0)

	lock = mp.Lock()
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	val_network = ValueNetwork()
	val_network.share_memory()

	target_value_network = ValueNetwork()
	target_value_network.share_memory()

	hard_copy(val_network,target_value_network)


	# Shared optimizer ->  Share the gradients between the processes. Lazy allocation so gradients are not shared here
	optimizer = SharedAdam(params=val_network.parameters(), lr=args.lr)
	optimizer.share_memory()
	optimizer.zero_grad()
	timesteps_per_process = 32*(10**6) // args.num_processes

	processes = []
	for idx in range(0, args.num_processes):
			trainingArgs = (idx, val_network, target_value_network, optimizer, lock, counter,timesteps_per_process,results_dir)
			p = mp.Process(target=train, args=trainingArgs)
			p.start()
			processes.append(p)
	for p in processes:
			p.join()
	saveModelNetwork(target_value_network,os.path.join(results_dir, 'params_last' ))



