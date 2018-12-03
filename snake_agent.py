import pygame
import sys
from random import randint
import random
import numpy as np

class Snake():
	def __init__(self, screen):
		#snake init
		self.screen = screen
		self.blocksize = 20
		self.snake = pygame.Rect(0,randint(10,20)*20,self.blocksize, self.blocksize)
		self.snake_color = (randint(0,200),randint(0,190),randint(0,180))
		self.length = 3
		self.snake_list = [[self.snake.x, self.snake.y]]
		self.reward = 0
		#movement
		self.left = False
		self.right = True
		self.up = False
		self.down = False
		self.speed = 20

	def policy(self):
		pass

	
	def take_action(self, env):

		action = np.random.choice(['L', 'R', 'U', 'D'])

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


		#observation, reward, done
		reward = env.reward(self.snake)

		done = False
		if env.game_over == True:
			done = True
			env.game_over = False

		return reward, done
		


	def show(self):
		for tail in self.snake_list:
			pygame.draw.rect(self.screen, self.snake_color, (tail[0],tail[1], self.blocksize, self.blocksize))