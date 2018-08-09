import numpy as np
import time
import random


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

    def show(self):
        old_sign = self.get_sign_at_current_position()
        self.maze[self.position // self.columns, self.position % self.columns] = '@'  # current position
        print(self.name)
        print(self.maze, '\n')
        self.maze[self.position // self.columns, self.position % self.columns] = old_sign

    def reset(self):
        self.position = 0 if self.name.startswith('5') else 18
        # if self.name.startswith('5'):
        #     self.position = 0
        # else:
        #     self.position = 18

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
        self.maze = Maze(self.name)
        self.maze.show()
        for step in range(self.max_steps):
            # time.sleep(1.5)     # for demo only

            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(self.Q[self.maze.position, :])

            self.maze.make_move(self.map2str[action])
            self.maze.show()

            _, done = self.check_position()
            if done:
                break


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    name = '5x5-v0'
    rl = ReinforcementLearning(name, verbose=False)
    rl.reinforcement_learning()
    rl.play()
