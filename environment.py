import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from multiagent.multi_discrete import MultiDiscrete
import random
from PaxBehavior import Settlement
import scipy.stats as st
import matplotlib.pyplot as plt
# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!


# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        self.world = world
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback 
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        self.low_count=[0,0]
        self.high_count=[0,0]
        self.EMRSb=40+st.norm.ppf(1-187.5/337.5)*5
        self.high_seats0=[]
        self.high_seats1=[]
        self.low_seats0=[]
        self.low_seats1=[]
        self.reward0=0
        self.reward1=0
        self.reward_list=[[],[]]
        self.price=[[],[],[],[]]
        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # class 0 action space
            u_action_space = spaces.Discrete(8)
            total_action_space.append(u_action_space)

            # class 1 action space
            c_action_space = spaces.Discrete(8)
            total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space
            # obs_dim = len(observation_callback(agent, self.world))
            # self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)
        self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32))
        self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(6,), dtype=np.float32))
        print(self.observation_space)

    def step(self, action_n,pax,left_Seats,day):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        agent_action_h=[]
        agent_action_l=[]
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
            agent_action_h.append(agent.action.u[0])
            agent_action_l.append(agent.action.u[1])
        done_list=[]
        done_list=[False,False]
        if left_Seats[0]<=0:
            done_list[0]=True
        if left_Seats[1]<=0:
            done_list[1]=True
        # print(agent_action,"agent_action")
        class_high=[[0,0],[0,0]]
        class_low=[[0,0],[0,0]]
        if done_list[0]: #Agent1 monopoly
            class_high[1]=Settlement(agent_action_h[1],pax[0],0,day)
            class_low[1]=Settlement(agent_action_l[1],pax[1],1,day)
        elif done_list[1]: #Agent2 monopoly
            class_high[0]=Settlement(agent_action_h[0],pax[0],0,day)
            class_low[0]=Settlement(agent_action_l[0],pax[1],1,day)
        else: #Compete
            ##high class compete
            if agent_action_h[0]==agent_action_h[1]:  #same price, share the market
                pax_=random.randint(0,pax[0])
                class_high[0]=Settlement(agent_action_h[1],pax_,0,day)
                class_high[1]=Settlement(agent_action_h[1],pax[0]-pax_,0,day)
            else:                                     #lower price wins all
                price=min(agent_action_h)
                ind=agent_action_h.index(min(agent_action_h))
                class_high[ind]=Settlement(price,pax[0],0,day)

            ##low class compete
            if agent_action_l[0]==agent_action_l[1]:  #same price, share the market
                pax_=random.randint(0,pax[1])
                class_low[0]=Settlement(agent_action_l[1],pax_,1,day)
                class_low[1]=Settlement(agent_action_l[1],pax[1]-pax_,1,day)
            else:                                     #lower price wins all
                price=min(agent_action_l)
                ind=agent_action_l.index(min(agent_action_l))
                class_low[ind]=Settlement(price,pax[1],1,day)
        sold_tickets=[class_low[0][1],class_high[0][1],class_low[1][1],class_high[1][1]]
        # self.world.step()
        # print(pax,"pax")
        # record observation for each agent
        obs_n=[np.array([agent_action_h[1],agent_action_l[1],0,0,0,0
                         ]),
               np.array([agent_action_h[0],agent_action_l[0],0,0,0,0])]
        for i,agent in enumerate(self.agents):
            # obs_n.append(self._get_obs(agent))
            reward_n.append(class_high[i][0]+class_low[i][0])
            done_n.append(done_list[i])
            info_n['n'].append(self._get_info(agent))

        # print(obs_n,"obs")
        self.reward=reward_n
        self.class_low=class_low
        self.class_high=class_high
        self.price_h=agent_action_h
        self.price_l=agent_action_l
        return obs_n, reward_n, done_n, info_n,sold_tickets

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        # for agent in self.agents:
        #     obs_n.append(self._get_obs(agent))
        #TODO: size of obs
        obs_n.append(np.array([0,0,0,0,0,0]))
        obs_n.append(np.array([0,0,0,0,0,0]))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(abs(action[index:(index+s)]))
                index += s
            action = act
        else:
            action = [abs(action)]

        action_choose_h=[250,275,300,325,350,375,400,425]
        action_choose_l=[100,125,150,175,200,225,250,275]
        action_h=action[0].tolist()
        action_l=action[1].tolist()
        #TODO: max of action output?
        agent.action.u[0] = action_choose_h[action_h.index(max(action_h))]
        agent.action.u[1] = action_choose_l[action_l.index(max(action_l))]

    # test model
    def Test_model(self,episode_step,plot_graph,price_curve):
        if episode_step==0:
            self.high_seats0.append(self.high_count[0])
            self.high_seats1.append(self.high_count[1])
            self.low_seats0.append(self.low_count[0])
            self.low_seats1.append(self.low_count[1])
            self.reward_list[0].append(self.reward0)
            self.reward_list[1].append(self.reward1)
            self.reward0=0
            self.reward1=0
            self.low_count=[0,0]
            self.high_count=[0,0]
            # if not price_curve:
            #     self.price=[[],[],[],[]]
        if plot_graph:
            curve=1
            curve_high_seat0=[]
            curve_high_seat1=[]
            curve_low_seat0=[]
            curve_low_seat1=[]
            curve_reward0=[]
            curve_reward1=[]
            for i in range(len(self.high_seats0)-curve):
                curve_high_seat0.append(sum(self.high_seats0[i:i+curve])/curve)
                curve_high_seat1.append(sum(self.high_seats1[i:i+curve])/curve)
                curve_low_seat0.append(sum(self.low_seats0[i:i+curve])/curve)
                curve_low_seat1.append(sum(self.low_seats1[i:i+curve])/curve)
                curve_reward0.append(sum(self.reward_list[0][i:i+curve])/curve)
                curve_reward1.append(sum(self.reward_list[1][i:i+curve])/curve)
            plt.plot(curve_high_seat0,label="Agent 0 High",color='blue',linestyle='-')
            plt.plot(curve_high_seat1,label="Agent 1 High",color='orange',linestyle='-')
            plt.plot(curve_low_seat0,label="Agent 0 low",color='blue',linestyle='-.')
            plt.plot(curve_low_seat1,label="Agent 1 low",color='orange',linestyle='-.')
            plt.axhline(y=self.EMRSb, label="EMSRb-high",color="green", linestyle='-')
            plt.axhline(y=100-self.EMRSb, label="EMSRb-low",color="green", linestyle='-.')
            plt.title("Sold seats")
            plt.xlabel("eposides")
            plt.ylabel("number of tickets")
            plt.ylim(0, 120)
            plt.legend()
            plt.show()
            plt.plot(curve_reward0,label="Agent 0")
            plt.plot(curve_reward1,label="Agent 1")
            plt.title("Rewards")
            plt.xlabel("eposides")
            plt.ylabel("total rewards")
            plt.legend()
            plt.show()
        if price_curve:
            ind=self.reward_list[0].index(max(self.reward_list[0]))
            plt.plot(self.price[0][ind*50:(ind+1)*50],label="Agent0 High Class",color='blue',linestyle='-')
            plt.plot(self.price[1][ind*50:(ind+1)*50],label="Agent1 High Class",color='orange',linestyle='-')
            plt.plot(self.price[2][ind*50:(ind+1)*50],label="Agent0 Low Class",color='blue',linestyle='-.')
            plt.plot(self.price[3][ind*50:(ind+1)*50],label="Agent1 Low Class",color='orange',linestyle='-.')
            plt.title("Ticket price")
            plt.xlabel("booking horizon")
            plt.ylabel("price")
            plt.legend()
            plt.show()
            print(self.price[0])
            print(self.price[1])
            print(self.price[2])
            print(self.price[3])
        self.low_count[0]+=self.class_low[0][1]
        self.low_count[1]+=self.class_low[1][1]
        self.high_count[0]+=self.class_high[0][1]
        self.high_count[1]+=self.class_high[1][1]
        self.reward0+=self.reward[0]
        self.reward1+=self.reward[1]
        self.price[0].append(self.price_h[0])
        self.price[1].append(self.price_h[1])
        self.price[2].append(self.price_l[0])
        self.price[3].append(self.price_l[1])


