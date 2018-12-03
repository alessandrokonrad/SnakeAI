import pygame
import sys
from random import randint
import random
from scipy.spatial.distance import euclidean
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Environment():

	def __init__(self,screen,clock, width, height):
		self.screen = screen
		self.width = width
		self.height = height
		self.screen_rect = self.screen.get_rect()
		self.clock = clock
		self.blocksize = 20
		self.range = self.width/self.blocksize
		#food
		self.food = pygame.Rect(randint(0,self.range-1)*self.blocksize,randint(0,self.range-1)*self.blocksize,self.blocksize, self.blocksize)
		self.food_color = (124,252,0)
		#game
		self.FPS = 1000
		self.has_food = False
		self.game_over = False
		#env
		self.action_space = 4
		self.actions = {0:'L', 1:'R', 2:'U', 3:'D'}
		self.observation_space = (80,80)
		self.stack_number = 4
		self.closest_distance = 1000000

		#movement
		self.left = False
		self.right = True
		self.up = False
		self.down = False
		self.speed = 20

		#snake init
		self.screen = screen
		self.snake = pygame.Rect(randint(3,self.range/2)*self.blocksize,randint(self.range/2,self.range-1)*20,self.blocksize, self.blocksize)
		self.snake_color = (255,69,0)
		self.length = 3
		self.snake_list = []
		


	def reward(self):
		s = (self.snake.x, self.snake.y)
		f = (self.food.x, self.food.y)
		distance_to_food = euclidean(s,f)

		if self.has_food:
			self.has_food = False
			return 1
		if self.game_over:
			return -1
		#if distance_to_food <= self.closest_distance:
		#	self.closest_distance = distance_to_food
		#	return 0.5
		return 0


	def step(self, action):

		action = self.actions[action]

		#check if move possible
		if self.right == False:
			if action == 'L':
				self.left = True
				self.right = False
				self.up = False
				self.down = False


	
		if self.left == False:
			if action == 'R':
				self.right = True
				self.left = False
				self.up = False
				self.down = False


		if self.down == False:
			if action == 'U':
				self.up = True
				self.left = False
				self.right = False
				self.down = False


		if self.up == False:
			if action == 'D':
				self.down = True
				self.left = False
				self.right = False
				self.up = False
		


		#take action update screen
		self.update_screen()
		
		#reward
		reward = self.reward()

		#done
		done = False
		if self.game_over == True:
			done = True
			self.game_over = False
		
		#observation
		observation = pygame.surfarray.array3d(pygame.display.get_surface())

		return observation, reward, done

	def update_screen(self):
		#take move
		if self.left:
			self.snake.x -= self.speed


		if self.right:
			self.snake.x += self.speed


		if self.up:
			self.snake.y -= self.speed


		if self.down:
			self.snake.y += self.speed

		self.snakeHead = []	

		self.snakeHead.append(self.snake.x)
		self.snakeHead.append(self.snake.y)
		self.snake_list.append(self.snakeHead)

		if len(self.snake_list) > self.length:
			del self.snake_list[0]

		self.update_food_location()
		self.check_game_status()
		self.render()
		self.clock.tick(self.FPS)
		pygame.display.flip()
	

	def update_food_location(self):
		#update food location
		if self.snake.center == self.food.center:
			self.has_food = True
			self.closest_distance = 1000000
			while True:
				self.food = pygame.Rect(randint(0,self.range-1)*self.blocksize,randint(0,self.range-1)*self.blocksize,self.blocksize, self.blocksize)
				if self.food not in self.snake_list:
					break
			self.length += 1
		
		
		
	def check_game_status(self):	
		#game over
		for segment in self.snake_list[:-1]:
			if segment == self.snakeHead:
				self.gameover()
		
		#boarder
		if self.snake.y <= self.screen_rect.top - 1:
			self.gameover()
			
		elif self.snake.y >= self.screen_rect.bottom :
			self.gameover()
			
		if self.snake.x <= self.screen_rect.left - 1:
			self.gameover()
			
		elif self.snake.x >= self.screen_rect.right:
			self.gameover()
		
	def render(self):
		self.screen.fill((255,255,255))
		for tail in self.snake_list:
			pygame.draw.rect(self.screen, self.snake_color, (tail[0],tail[1], self.blocksize, self.blocksize))
		pygame.draw.rect(self.screen, self.food_color, self.food)
	
	
	def gameover(self):
		self.game_over = True
		

	def reset(self):
		self.closest_distance = 1000000
		self.length = 3
		self.snakeHead = []
		self.snake = pygame.Rect(randint(3,self.range/2)*self.blocksize,randint(self.range/2,self.range-1)*20,self.blocksize, self.blocksize)
		self.snake_list = [[self.snake.x-2*20, self.snake.y],[self.snake.x-20, self.snake.y],[self.snake.x, self.snake.y]]
		self.food = pygame.Rect(randint(0,self.range-1)*self.blocksize,randint(0,self.range-1)*self.blocksize,self.blocksize, self.blocksize)
		self.left = False
		self.up = False
		self.down = False
		self.right = True
		self.has_food = False
		self.game_over = False
		while self.food.topleft == self.snake.topleft:
			self.food = pygame.Rect(randint(0,self.range-1)*self.blocksize,randint(0,self.range-1)*self.blocksize,self.blocksize, self.blocksize)

		self.clock.tick(self.FPS)
		pygame.display.flip()
		self.render()
		observation = pygame.surfarray.array3d(pygame.display.get_surface())
		obs_sampled = self.downsample_image(observation)
		state = [obs_sampled]*4
		return state

	def update_state(self, state, observation):
		obs_small = self.downsample_image(observation)
		state.append(obs_small)
		if len(state) > 4:
			state.pop(0)
		return state
		
	
	def downsample_image(self, observation):
		observation = observation.mean(axis=2)
		observation = cv2.resize(observation, dsize=self.observation_space, 
								 interpolation=cv2.INTER_NEAREST)
	
		return observation
		
		
		
	
	
		
