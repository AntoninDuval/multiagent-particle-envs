#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import numpy as np

from multiagent.environment import MultiAgentEnv
from multiagent.policy import RandomPolicy
import multiagent.scenarios as scenarios
import time

if __name__ == '__main__':

    # Load arguments

    parser = argparse.ArgumentParser()
    parser.add_argument("agent_view_radius")
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load('simple_tag_coop_partial_obs_v5.py').Scenario()
    # create world
    world = scenario.make_world(args)
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = True)
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [RandomPolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    while True:
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(env.world.agents[3].state.p_pos/8)
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        #print(obs_n[0])
        # render all agent views
        env.render()
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
