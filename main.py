import pygame
import pygame.surfarray as surfarray
import sys
from environment import Environment
from snake_agent import Snake
from DQN import DQN
from policy import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import datetime
import random

def play_one(
  env,
  total_t,
  experience_replay_buffer,
  model,
  target_model,
  gamma,
  batch_size,
  epsilon,
  epsilon_change,
  epsilon_min):

  t0 = datetime.now()

  # Reset the environment
  state = env.reset()
  #assert(state.shape == (4, 80, 80))
  loss = None


  total_time_training = 0
  num_steps_in_episode = 0
  episode_reward = 0

  done = False
  while not done:
    
    # Update target network
    if total_t % TARGET_UPDATE_PERIOD == 0:
      target_model.copy_from(model)
      print("Copied model parameters to target network. total_t = %s, period = %s" % (total_t, TARGET_UPDATE_PERIOD))


    # Take action
    action = model.sample_action(state, epsilon)
    obs, reward, done = env.step(action)
    next_state = env.update_state(state.copy(), obs)
    # assert(state.shape == (4, 80, 80))


    episode_reward += reward

    # Remove oldest experience if replay buffer is full
    if len(experience_replay_buffer) == MAX_EXPERIENCES:
      experience_replay_buffer.pop(0)

    # Save the latest experience
    state_shaped = np.reshape(state, (4,80,80))
    next_state_shaped = np.reshape(next_state, (4,80,80))
    experience_replay_buffer.append((state, action, reward, next_state, done))

    # Train the model, keep track of time
    t0_2 = datetime.now()
    dt = datetime.now() - t0_2

    total_time_training += dt.total_seconds()
    num_steps_in_episode += 1

    loss = learn(model, target_model, experience_replay_buffer, gamma, batch_size)

    state = next_state
    total_t += 1

    epsilon = max(epsilon - epsilon_change, epsilon_min)

  return total_t, episode_reward, (datetime.now() - t0), num_steps_in_episode, total_time_training/num_steps_in_episode, epsilon

def learn(model, target_model, experience_replay_buffer, gamma, batch_size):
  # Sample experiences
  samples = random.sample(experience_replay_buffer, batch_size)
  states, actions, rewards, next_states, dones = map(np.array, zip(*samples))

  # Calculate targets
  next_Qs = target_model.predict(next_states)
  next_Q = np.amax(next_Qs, axis=1)
  targets = rewards + np.invert(dones).astype(np.float32) * gamma * next_Q

  # Update model
  loss = model.update(states, actions, targets)
  return loss

def run_game():
	state = env.reset()
	
	t = 0
	done = False
	while True:

		#env.render()
		
		
		if len(state) < 4:
			action = np.random.choice(list(env.actions.keys()))
		else:
			action = agent.act(state)

		
		
		observation, reward, done = env.step(action)

		next_state = env.update_state(state.copy(), observation)

		agent.remember(state, action, reward, next_state, done)
		print(env.food.x,env.food.y)
		
		if done:
			break

		state = next_state
		t+=1
		

	if len(agent.memory) >= BATCH_SIZE:
		agent.replay(BATCH_SIZE)


if __name__ == "__main__":

	
	#Initialize game
	pygame.init()
	width, height = (240,240)

	screen = pygame.display.set_mode((width,height))
	caption = pygame.display.set_caption('Snake')

	clock = pygame.time.Clock()

	env = Environment(screen, clock, width, height)
	agent = DQNAgent(env.action_space)

	BATCH_SIZE = 32
	EPISODE = 10

	#while True:
	#	run_game()



	# hyperparams and initialize stuff
	conv_layer_sizes = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
	hidden_layer_sizes = [512]
	gamma = 0.99
	batch_sz = 32
	num_episodes = 10000
	total_t = 0
	experience_replay_buffer = []
	episode_rewards = np.zeros(num_episodes) 

	MAX_EXPERIENCES = 500000
	MIN_EXPERIENCES = 10000
	TARGET_UPDATE_PERIOD = 10000



	# epsilon
	# decays linearly until 0.1
	epsilon = 1.0
	epsilon_min = 0.1
	epsilon_change = (epsilon - epsilon_min) / 500000


	# Create models
	model = DQN(
	K=env.action_space,
	conv_layer_sizes=conv_layer_sizes,
	hidden_layer_sizes=hidden_layer_sizes,
	gamma=gamma,
	scope="model")
	target_model = DQN(
	K=env.action_space,
	conv_layer_sizes=conv_layer_sizes,
	hidden_layer_sizes=hidden_layer_sizes,
	gamma=gamma,
	scope="target_model"
	)



	with tf.Session() as sess:
		model.set_session(sess)
		target_model.set_session(sess)
		sess.run(tf.global_variables_initializer())


		print("Populating experience replay buffer...")
		state = env.reset()
		
		# assert(state.shape == (4, 80, 80))
		for i in range(MIN_EXPERIENCES):
			print(i)
			
			action = np.random.choice(env.action_space)
			obs, reward, done = env.step(action)
			next_state = env.update_state(state.copy(), obs)
			
			state_shaped = np.reshape(state, (4,80,80))
			next_state_shaped = np.reshape(next_state, (4,80,80))
			
			assert(state_shaped.shape == (4, 80, 80))
			experience_replay_buffer.append((state, action, reward, next_state, done))

			if done:
				state = env.reset()
				
				# assert(state.shape == (4, 80, 80))
			else:
				state = next_state


		# Play a number of episodes and learn!
		while True:

			total_t, episode_reward, duration, num_steps_in_episode, time_per_step, epsilon = play_one(
			env,
			total_t,
			experience_replay_buffer,
			model,
			target_model,
			gamma,
			batch_sz,
			epsilon,
			epsilon_change,
			epsilon_min,
			)
			episode_rewards[i] = episode_reward

			last_100_avg = episode_rewards[max(0, i - 100):i + 1].mean()
			print("Episode:", i,
			"Duration:", duration,
			"Num steps:", num_steps_in_episode,
			"Reward:", episode_reward,
			"Training time per step:", "%.3f" % time_per_step,
			"Avg Reward (Last 100):", "%.3f" % last_100_avg,
			"Epsilon:", "%.3f" % epsilon
			)
			sys.stdout.flush()