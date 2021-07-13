#!/usr/bin/python3
# coding: utf-8
'''
 @Time    : 2021/4/14 23:16
 @Author  : Shulu Chen
 @FileName: mainProcess.py
 @Software: PyCharm
'''
import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import matplotlib.pyplot as plt
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from PaxBehavior import generate_pax,get_pax
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="test_scenarios", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=50, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=100000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="numbe  r of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for Adam optimizer")#try higher and lower,
    parser.add_argument("--gamma", type=float, default=1, help="discount factor")#0.9999
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")#three layer, 256
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="compete_price_pax", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="D:\maddpg_det_manual\policy", help="directory in which training state and model should be saved")
    parser.add_argument("--save-dir2", type=str, default="D:\maddpg_det_manual2\policy", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="D:\maddpg_det_manual\policy", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--test", action="store_true", default= False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=10000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from environment_manual import MultiAgentEnv
    # import multiagent.scenarios as scenarios
    import scenarios
    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()

    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,done_callback=scenario.fulled)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]

        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.test or arglist.restore:
            print('Loading previous state...')
            print(arglist.load_dir)
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        left_seats=[50,50]
        load_factor_0=[]
        load_factor_1=[]
        t_start = time.time()
        addition_rew_count=0
        print('Starting iterations...')
        sold_seats_h=[0,0]
        sold_seats_l=[0,0]
        arrival_data=generate_pax()
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # print(action_n)
            # action_n=[1,2,3,4,5]
            # print(action_n)

            # print(episode_step)
            # print(left_seats)
            # environment step
            new_obs_n, rew_n, done_n, info_n,sold_seats = env.step(action_n,get_pax(arrival_data,episode_step),left_seats,episode_step)
            episode_step += 1
            sold_seats_h[0]+=sold_seats[1]
            sold_seats_h[1]+=sold_seats[3]
            sold_seats_l[0]+=sold_seats[0]
            sold_seats_l[1]+=sold_seats[2]
            left_seats=[left_seats[0]-sold_seats[0]-sold_seats[1],left_seats[1]-sold_seats[2]-sold_seats[3]]
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            new_obs_n[0][0]=sold_seats_h[0]
            new_obs_n[0][1]=sold_seats_l[0]
            new_obs_n[0][2]=left_seats[0]
            new_obs_n[0][3]=episode_step
            new_obs_n[1][0]=sold_seats_h[1]
            new_obs_n[1][1]=sold_seats_l[1]
            new_obs_n[1][2]=left_seats[1]
            new_obs_n[1][3]=episode_step
            # print(new_obs_n)
            # #Give additional rewards for agent 0 if over agent 1
            # if rew_n[1]>rew_n[0]:
            #     rew_n[1]+=50
            #     addition_rew_count+=1
            # collect experience
            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew
            for i, agent in enumerate(trainers):
                # print(done_n[i],i)
                # print(agent_rewards[i][-1])
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n




            # print(rew_n[0])
            if done or terminal or episode_step>=50:
                # agent_rewards[1][-1]-=addition_rew_count*50
                # print(episode_rewards[-1],"eps_rew")
                # print(agent_rewards[0][-1],"agt0")
                # print(agent_rewards[1][-1],"agt1")
                arrival_data=generate_pax()
                sold_seats_h=[0,0]
                sold_seats_l=[0,0]
                obs_n = env.reset()
                episode_step = 0
                addition_rew_count=0
                load_factor=[50-left_seats[0],50-left_seats[1]]
                left_seats=[50,50]
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                load_factor_0.append(load_factor[0])
                load_factor_1.append(load_factor[1])
                agent_info.append([[]])
                print(len(episode_rewards)*100/arglist.num_episodes,"%")



        # increment global step counter
            train_step += 1

            if arglist.test:
                if len(episode_rewards) > arglist.num_episodes:
                    env.Test_model(episode_step,plot_graph=True,price_curve=True)
                    break
                else:
                    env.Test_model(episode_step,plot_graph=False,price_curve=False)
                    continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)

            # save model, display training output
            # if terminal and (len(episode_rewards) % arglist.save_rate == 0):
            if agent_rewards[0][-1]==max(agent_rewards[0]):
                print("save")
                U.save_state(arglist.save_dir2, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
                rew_file_name = arglist.save_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.save_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:

                episode_rewards.pop(-1)
                agent_rewards[0].pop(-1)
                agent_rewards[1].pop(-1)
                curve_rew=[]
                curve_rew_a1=[]
                curve_rew_a2=[]
                load_0=[]
                load_1=[]
                curve=40
                for i in range(len(episode_rewards)-curve):
                    curve_rew.append(sum(episode_rewards[i:i+curve])/curve)
                    curve_rew_a1.append(sum(agent_rewards[0][i:i+curve])/curve)
                    curve_rew_a2.append(sum(agent_rewards[1][i:i+curve])/curve)
                    load_0.append(sum(load_factor_0[i:i+curve])/curve)
                    load_1.append(sum(load_factor_1[i:i+curve])/curve)
                plt.plot(curve_rew,label="episode rewards",color="green")
                plt.plot(curve_rew_a1,label="agent 0",color="orange")
                plt.plot(curve_rew_a2,label="agent 1",color="blue")
                plt.title("Total rewards, deterministic lr="+str(arglist.lr)+" gamma="+str(arglist.gamma))

                plt.legend()
                plt.show()
                plt.plot(load_0,label="agent 0",color="orange")
                plt.plot(load_1,label="agent 1",color='blue')
                plt.title("Load Factor, deterministic lr="+str(arglist.lr)+" gamma="+str(arglist.gamma))
                plt.legend()
                plt.show()
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                # print("save")
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))
                rew_file_name = arglist.save_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.save_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)


                break
start=time.time()
if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
    print(time.time()-start,"runnig time")