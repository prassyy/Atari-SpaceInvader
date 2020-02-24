import gym
import cv2
import time
import numpy as np
import argparse

from DQN import DQN
from ReplayBuffer import ReplayBuffer

class Game(object):

    def __init__(self, env_name):
        self.episilon = 1.0
        self.episilon_min = 0.1
        self.episilon_decay = 0.000003
        self.num_of_epochs = 1000
        self.batch_size = 32
        self.num_of_frames = 3

        #Initialize network with parameters
        self.env_name = env_name
        self.env = gym.make(env_name)
        number_of_actions = self.env.action_space.n
        self.deep_q = DQN(self.num_of_frames, number_of_actions)
        self.replay_buffer = ReplayBuffer(100000)

        # Initialize the frame_buffer
        self.frame_buffer = []
        self.env.reset()
        s1, r1, _, _ = self.env.step(0)
        s2, r2, _, _ = self.env.step(0)
        s3, r3, _, _ = self.env.step(0)
        self.frame_buffer = [s1, s2, s3]

    def load_network(self):
        self.deep_q.load_network("save/{}/saved.h5".format(self.env_name))

    def process_frame_buffer(self):
        grayscale_buffer = [cv2.resize(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), (84, 90)) for x in self.frame_buffer]
        grayscale_buffer = [x[1:85, :, np.newaxis] for x in grayscale_buffer]
        return np.concatenate(grayscale_buffer, axis=2)

    def replay(self):
        if self.replay_buffer.size() > self.batch_size:
            s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample(self.batch_size)
            self.deep_q.train(s_batch, a_batch, r_batch, d_batch, s2_batch)
            self.deep_q.target_train()

    def train(self):
        epoch = 0
        while epoch < self.num_of_epochs:
            self.env.reset()
            alive_frame = 0
            total_reward = 0
            done = False

            while done == False:
                if self.episilon > self.episilon_min:
                    self.episilon -= self.episilon_decay

                current_state = self.process_frame_buffer()
                self.frame_buffer = []
                action, _ = self.deep_q.predict(current_state, self.episilon)

                reward, done = 0, False
                for i in range(self.num_of_frames):
                    if epoch % 10 == 0:
                        self.env.render()
                    temp_state, temp_reward, temp_done, _ = self.env.step(action)
                    reward += temp_reward
                    self.frame_buffer.append(temp_state)
                    done = done | temp_done

                self.replay()
                new_state = self.process_frame_buffer()
                self.replay_buffer.add(current_state, action, reward, done, new_state)

                alive_frame += 1
                total_reward += reward

            if epoch % 10 == 0:
                self.deep_q.save_network("save/{}/saved.h5".format(self.env_name))

            print("Alive time: {}, Total reward: {}, Epoch: {}".format(alive_frame, total_reward, epoch))
            epoch += 1

    def play(self):
        done = False
        tot_reward = 0
        self.env.reset()
        
        while not done:
            state = self.process_frame_buffer()
            action = self.deep_q.predict(state, 0)[0]
            self.env.render()
            time.sleep(0.01)
            state, reward, done, _ = self.env.step(action)
            tot_reward += reward
            self.frame_buffer.append(state)
            self.frame_buffer = self.frame_buffer[1:]
        print('Total score: {}'.format(tot_reward))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test different Atari Games")
    parser.add_argument("-m", "--mode", type=str, action='store', help="Please specify the mode you wish to run, either train or test", required=True)
    args = parser.parse_args()

    game = Game('SpaceInvaders-v0')
    if args.mode == "train":
        game.train()
    else:
        game.load_network()
        while True:
            game.play()
