from MDP import MDP

class BellmanDPSolver(object):
	def __init__(self,discountRate):
		self.MDP = MDP()
		self.states = self.MDP.S
		self.Actions = self.MDP.A
		self.state_values = self.initVs()
		self.policy = {}
		self.gamma = discountRate

	def initVs(self):
		state_values = {}
		for state in self.states:
			state_values[state]=0.0
		return state_values

	def one_step_ahead(self,state):
		"""
		Function that calculates the value for all actions in a given state
		Args: state to be considered
		Returns: A dictionary with keys the actions that can be taken and as values the expected value of each action 
		"""
		action_values = {}
		for action in self.Actions:
			transition_prob = self.MDP.probNextStates(state,action)
			total = 0.0
			for next_state, probability in transition_prob.items():
				reward = self.MDP.getRewards(state,action,next_state)
				total+= probability*(reward + self.gamma*self.state_values[next_state])
			action_values[action] = total
		return action_values


	def BellmanUpdate(self):
		for state in self.states:
			Action_values = self.one_step_ahead(state)
			max_value = max(Action_values.values())
			self.state_values[state] = max_value
			actions=[]
			for action in self.Actions:
				if Action_values[action]==max_value:
					actions.append(action)
			self.policy[state] = actions
		return (self.state_values, self.policy)
	
		
		
if __name__ == '__main__':
	solution = BellmanDPSolver(0.9)
	for i in range(20000):
		values, policy = solution.BellmanUpdate()
	print("Values : ", values)
	print("Policy : ", policy)


