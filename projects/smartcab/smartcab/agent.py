import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        # All possible actions
        actions = self.env.valid_actions   # [None, 'forward', 'left', 'right']
        # Traffic light state choices
        traffic_light = ['red', 'green']
        # All other states have the same choices
        waypoint, oncoming, left = actions, actions, actions
        # Initialize a dictionary to store the Q-values, intialized with all zeros
        self.q_table = {}
        for li in traffic_light:
            for pt in waypoint:
                for on in oncoming:
                    for lf in left:
                        self.q_table[(li, pt, on, lf)] = {None: 0, 'forward': 0, 'left': 0, 'right': 0}

        self.success_num = 0


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)  #{'light': light, 'oncoming': oncoming, 'left': left, 'right': right}
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        # Use traffic light, next waypoint, oncoming car and left car as a state. Global location is not appropriate for status because this cab cannot get that information, though it's better to have it.
        self.state = (inputs['light'],
                      self.next_waypoint,
                      inputs['oncoming'],
                      inputs['left'])
        # print ("The current state is: {}".format(self.state))
        # print ("t:{}".format(t))

        # TODO: Select action according to your policy
        # action = None
        # action = random.choice(Environment.valid_actions)
        epsilon = 0.1  # e-greedy method
        # Comment: Answering to the question, if I set big number as epsilon, success rate becomes bad.
        action = max(self.q_table[self.state],
                     key=self.q_table[self.state].get) if random.random() < (1 - epsilon) else random.choice(Environment.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        # Set the tuning parameters
        alpha = 0.1  # learning rate
        gamma = 0.9  # discount factor
        # Get the new state after the above action
        inputs_new = self.env.sense(self)
        state_new = (inputs_new['light'],
                     self.planner.next_waypoint(),
                     inputs_new['oncoming'],
                     inputs_new['left'])
        print ("The new state is: {}".format(state_new))
        print ("t:{}".format(t))
        print ("\n")

        # Calculate the Q_value
        # By using q value, smartcab continues to improve
        q_value = (1 - alpha) * self.q_table[self.state][action] + \
                  alpha * (reward + gamma * max(self.q_table[state_new].values()))
        # Update the Q_table
        self.q_table[self.state][action] = q_value
        # Set current state and action as previous state and action
        #self.state_prev = self.state
        #self.action_prev = action

        if self.env.done == True:
            self.success_num += 1
            # print(self.success_num)


        print ("LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward) ) # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.001, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    trial_num = 100
    sim.run(n_trials=trial_num)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    print("Success rate: {}".format(a.success_num/trial_num))


if __name__ == '__main__':
    run()



# Reference: https://discussions.udacity.com/t/why-state-changes-within-the-self-update-run/164707/3
