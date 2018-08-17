import numpy as np
import time
import random
import matplotlib.pyplot as plt
from matplotlib import animation


class Maze:
    def __init__(self, name='5x5-v0'):
        self.name = name
        ### positions
        # [0  1  2  3  4]
        # [5  6  7  8  9]
        # [10 11 12 13 14]
        # [15 16 17 18 19]
        # [20 21 22 23 24]
        d = {
            '5x5-v0': np.array([['S', '.', '.', '.', '.'],
                                ['.', 'O', '.', 'O', '.'],
                                ['.', '.', '.', '.', '.'],
                                ['.', 'O', '.', 'O', '.'],
                                ['.', '.', '.', '.', 'E']]),
            '5x5-v1': np.array([['S', '.', 'O', '.', '.'],
                                ['.', 'O', '.', 'O', '.'],
                                ['.', '.', '.', '.', '.'],
                                ['.', 'O', '.', 'O', '.'],
                                ['.', '.', 'O', '.', 'E']]),
            '10x10': np.array([['.', 'O', '.', '.', 'O', '.', '.', '.', '.', '.'],
                               ['.', '.', 'O', '.', '.', '.', 'O', '.', 'S', '.'],
                               ['.', '.', '.', 'O', '.', '.', 'O', '.', '.', '.'],
                               ['.', '.', '.', '.', 'O', '.', 'O', 'O', 'O', '.'],
                               ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
                               ['.', '.', '.', '.', 'O', '.', 'O', '.', '.', 'O'],
                               ['.', 'O', 'O', 'O', '.', '.', '.', 'O', '.', '.'],
                               ['.', '.', 'E', 'O', '.', '.', '.', '.', 'O', '.'],
                               ['.', 'O', '.', 'O', '.', '.', '.', '.', '.', 'O'],
                               ['O', '.', '.', '.', '.', '.', '.', '.', '.', '.']])
        }

        self.maze = d[self.name]
        self.rows, self.columns = self.maze.shape
        self.reset()
        self.enlarge = 10
        self.size = (self.rows * self.enlarge, self.columns * self.enlarge)

    def show(self):
        old_sign = self.get_sign_at_current_position()
        self.maze[self.position // self.columns, self.position % self.columns] = '@'  # current position
        print(self.name)
        [print(row) for row in self.maze]
        print()
        self.maze[self.position // self.columns, self.position % self.columns] = old_sign

    def make_image(self, size):
        img = np.zeros((*size, 3), dtype='uint8')
        for i, row in enumerate(self.maze):
            for j, cell in enumerate(row):
                img[i*self.enlarge: (i+1)*self.enlarge-1, j*self.enlarge: (j+1)*self.enlarge-1] = self.coloured_part(cell)[-1, -1, :]

        i = self.position // self.columns
        j = self.position % self.columns
        img[i*self.enlarge: (i+1)*self.enlarge-1, j*self.enlarge: (j+1)*self.enlarge-1] = self.coloured_part('@')[-1, -1, :]
        img[0, :, :] = 0
        img[:, 0, :] = 0
        return img

    def coloured_part(self, cell):
        part = np.zeros((self.enlarge, self.enlarge, 3), dtype='uint8')
        if cell == 'S' or cell == '.':
            part[:, :, 1] = 200
        elif cell == 'O':
            part[:, :, 0] = 200
        elif cell == 'E':
            part[:, :, 0] = 250
            part[:, :, 1] = 200
        elif cell == '@':
            part[:, :, 2] = 200
        return part

    def reset(self):
        self.position = 0 if self.name.startswith('5') else 18

    def get_sign_at_current_position(self):
        return self.maze[self.position // self.columns, self.position % self.columns]

    def check_start(self):
        return self.get_sign_at_current_position() == 'S'

    def check_end(self):
        return self.get_sign_at_current_position() == 'E'

    def check_holes(self):
        return self.get_sign_at_current_position() == 'O'

    def check_path(self):
        return self.get_sign_at_current_position() == '.'

    def make_move(self, direction):
        if direction == 'up':
            if self.position >= self.columns:
                self.position -= self.columns
        elif direction == 'right':
            if self.position % self.columns != self.columns - 1:
                self.position += 1
        elif direction == 'down':
            if self.position < (self.rows * self.columns) - self.columns:
                self.position += self.columns
        elif direction == 'left':
            if self.position % self.columns != 0:
                self.position -= 1


class ReinforcementLearning:
    def __init__(self, name='5x5-v0', verbose=False):
        self.name = name
        self.verbose = verbose
        self.maze = Maze(self.name)
        self.set_parameters()

    def set_parameters(self):
        self.max_episodes = 1000
        self.learning_rate = 0.1
        self.max_steps = 100     # Max steps per episode
        self.discount_rate = 0.95

        # Exploration parameters
        self.epsilon = 1.0       # Exploration rate
        self.max_epsilon = 1.0   # Exploration probability at start
        self.min_epsilon = 0.01  # Minimum exploration probability
        self.decay_rate = 0.01   # Exponential decay rate for exploration prob

        self.map2str = {
            0: 'up',
            1: 'right',
            2: 'down',
            3: 'left'
        }

    def check_position(self):
        """
        :return: reward, done
        """
        if self.maze.check_end():
            return 1000, True
        elif self.maze.check_holes():
            return -100, True
        elif self.maze.check_path() or self.maze.check_start():
            return -1, False

    def reinforcement_learning(self):
        action_size = len(self.map2str)
        state_size = self.maze.rows * self.maze.columns

        Q_table = np.zeros((state_size, action_size))

        rewards = []
        for episode in range(self.max_episodes):
            done = False
            total_reward = 0
            self.maze.reset()

            if self.verbose:
                print("***********************************************")
                print("EPISODE ", episode, '\n')
                self.maze.show()
                time.sleep(2)

            for step in range(self.max_steps):
                r = random.uniform(0, 1)

                if r > self.epsilon:    # exploitation (biggest value of current state)
                    action = np.argmax(Q_table[self.maze.position, :])
                else:   # exploration (random choice)
                    action = random.randint(0, action_size - 1)

                previous_position = self.maze.position
                self.maze.make_move(self.map2str[action])

                reward, done = self.check_position()

                # update Q_table
                Q_table[previous_position, action] = Q_table[previous_position, action] + self.learning_rate * \
                                                (reward + self.discount_rate * np.max(Q_table[self.maze.position, :]) -
                                                 Q_table[previous_position, action])

                if self.verbose:
                    self.maze.show()
                    time.sleep(1.5)

                total_reward += reward
                if done:
                    break

            rewards.append(total_reward)

            # reduce epsilon (less exploration)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)

        if self.verbose:
            print('   up     right     down     left')
            for i, row in enumerate(Q_table):
                print(i, row)
            print('\n\n')

        self.Q = Q_table

    def play(self):
        images = []
        self.maze = Maze(self.name)
        images.append([plt.imshow(self.maze.make_image(self.maze.size), animated=True)])

        for step in range(self.max_steps):
            # time.sleep(1.5)     # for demo only

            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(self.Q[self.maze.position, :])

            self.maze.make_move(self.map2str[action])
            images.append([plt.imshow(self.maze.make_image(self.maze.size), animated=True)])

            _, done = self.check_position()
            if done:
                break

        anim = animation.ArtistAnimation(plt.figure(0), images, interval=1000, repeat=False, blit=True)
        plt.show()


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    name = '10x10'
    rl = ReinforcementLearning(name, verbose=False)
    rl.reinforcement_learning()
    rl.play()
