#!/usr/bin/python3
# coding: utf-8
'''
 @Time    : 2021/4/15 0:54
 @Author  : Shulu Chen
 @FileName: test_scenarios.py
 @Software: PyCharm
'''
import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):

    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 1
        num_agents = 2
        world.num_agents = num_agents
        num_adversaries = 1
        num_landmarks = num_agents - 1
         # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.adversary = True if i < num_adversaries else False
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.size = 0.08
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        goal = np.random.choice(world.landmarks)
        for agent in world.agents:
            agent.goal_a = goal

        world.agents[0].state.p_pos=np.array([0,0])
        world.agents[0].state.p_vel=np.zeros(world.dim_p)
        world.agents[0].state.c = np.zeros(world.dim_c)
        world.agents[1].state.p_pos=np.array([0,0])
        world.agents[1].state.p_vel=np.zeros(world.dim_p)
        world.agents[1].state.c = np.zeros(world.dim_c)

        world.landmarks[0].state.p_pos=np.array([0.8,0.8])
        world.landmarks[0].state.p_vel = np.zeros(world.dim_p)

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark
        return self.adversary_reward(agent, world) if agent.adversary else self.agent_reward(agent, world)

    def agent_reward(self, agent, world):

        # Rewarded based on how close any good agent is to the goal landmark, and how far the adversary is from it
        shaped_reward = False
        shaped_adv_reward = False

        # Calculate negative reward for adversary
        adversary_agents = self.adversaries(world)
        if shaped_adv_reward:  # distance-based adversary reward
            adv_rew = sum([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in adversary_agents])
        else:  # proximity-based adversary reward (binary)
            adv_rew = 0
            for a in adversary_agents:
                if np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) < 2 * a.goal_a.size:
                    adv_rew -= 5

        # Calculate positive reward for agents
        good_agents = self.good_agents(world)
        if shaped_reward:  # distance-based agent reward
            pos_rew = -min(
                [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        else:  # proximity-based agent reward (binary)
            pos_rew = 0
            # if min([np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents]) \
            #         < 2 * agent.goal_a.size:
            # if min([abs(a.state.p_pos[0]-0.8) for a in good_agents])<0.1:
            pos_rew += 1-(min([abs(a.state.p_pos[0]) for a in good_agents]))
            # pos_rew -= min(
            #     [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal_a.state.p_pos))) for a in good_agents])
        # return pos_rew + adv_rew
        return pos_rew

    def adversary_reward(self, agent, world):
        # Rewarded based on proximity to the goal landmark

        shaped_reward = False
        if shaped_reward:  # distance-based reward
            return -np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))
        else:  # proximity-based reward (binary)
            adv_rew = 0
            # if np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos))) < 5 * agent.goal_a.size:
            # if abs(agent.state.p_pos[0]-0.8)<0.1:
            adv_rew += 1-abs(agent.state.p_pos[0])

            return adv_rew


    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:
            entity_color.append(entity.color)
        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        print(np.concatenate(entity_pos + other_pos))
        if not agent.adversary:
            return np.concatenate([agent.goal_a.state.p_pos - agent.state.p_pos] + entity_pos + other_pos)
        else:
            return np.concatenate(entity_pos + other_pos)

    def fulled(self,agent,world):
        # print(agent.state.p_pos)
        if agent.state.p_pos[1]>=8000000:
            return True
        else:
            return False
        return False