import gym
from gym.utils import play

if __name__ == "__main__":
	play.play(gym.make('SpaceInvaders-v0'), zoom=3)