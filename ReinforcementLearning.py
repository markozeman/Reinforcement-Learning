import numpy as np
import gym
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
        print(self.rows, 'x', self.columns, 'maze')
        print(self.maze, '\n')
        self.maze[self.position // self.columns, self.position % self.columns] = old_sign

    def reset(self):
        if self.name.startswith('5'):
            self.position = 0
        else:
            self.position = 18

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
    def __init__(self, name='5x5-v0'):
        self.name = name
        self.maze = Maze(self.name)
        self.set_parameters()
        self.reinforcement_learning()
        self.play()

    def set_parameters(self):
        self.max_episodes = 1000
        self.learning_rate = 0.1
        self.max_steps = 100          # Max steps per iteration
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

    def check_position(self, previous_position):
        """
        :return: reward, done
        """
        # if self.maze.position == previous_position:
        #     return -1, False
        if self.maze.check_start():
            return -5, False
        elif self.maze.check_end():
            return 1000, True
        elif self.maze.check_holes():
            return -100, True
        elif self.maze.check_path():
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

            for step in range(self.max_steps):
                r = random.uniform(0, 1)

                if r > self.epsilon:    # exploitation (biggest value of current state)
                    action = np.argmax(Q_table[self.maze.position, :])
                else:   # exploration (random choice)
                    action = random.randint(0, action_size - 1)

                previous_position = self.maze.position
                self.maze.make_move(self.map2str[action])

                reward, done = self.check_position(previous_position)

                # update Q_table
                Q_table[previous_position, action] = Q_table[previous_position, action] + self.learning_rate * \
                                                (reward + self.discount_rate * np.max(Q_table[self.maze.position, :]) -
                                                 Q_table[previous_position, action])

                total_reward += reward
                if done:
                    break

            rewards.append(total_reward)

            # reduce epsilon (less exploration)
            self.epsilon = self.min_epsilon + (self.max_epsilon - self.min_epsilon) * np.exp(-self.decay_rate * episode)


        # print("Score over time: " + str(sum(rewards) / self.max_episodes))
        print('   up    right    down    left')
        for i, row in enumerate(Q_table):
            print(i, row)

        self.Q = Q_table

    def play(self):
        print('\n\n\n')
        self.maze = Maze(self.name)
        self.maze.show()
        for step in range(self.max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(self.Q[self.maze.position, :])

            self.maze.make_move(self.map2str[action])

            self.maze.show()

            _, done = self.check_position(99)
            if done:
                break


    # def frozen_lake(self):
    #     env = gym.make("FrozenLake-v0")
    #     env.render()
    #     action_size = env.action_space.n
    #     state_size = env.observation_space.n
    #
    #     print('action_size:', action_size, '   state_size', state_size)
    #
    #     qtable = np.zeros((state_size, action_size))
    #
    #     total_episodes = 10000  # Total episodes
    #     learning_rate = 0.8  # Learning rate
    #     max_steps = 99  # Max steps per episode
    #     gamma = 0.95  # Discounting rate
    #
    #     # Exploration parameters
    #     epsilon = 1.0  # Exploration rate
    #     max_epsilon = 1.0  # Exploration probability at start
    #     min_epsilon = 0.01  # Minimum exploration probability
    #     decay_rate = 0.01  # Exponential decay rate for exploration prob
    #
    #     # List of rewards
    #     rewards = []
    #
    #     # 2 For life or until learning is stopped
    #     for episode in range(total_episodes):
    #         # Reset the environment
    #         state = env.reset()
    #         step = 0
    #         done = False
    #         total_rewards = 0
    #
    #         for step in range(max_steps):
    #             # 3. Choose an action a in the current world state (s)
    #             ## First we randomize a number
    #             exp_exp_tradeoff = random.uniform(0, 1)
    #
    #             ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
    #             if exp_exp_tradeoff > epsilon:
    #                 action = np.argmax(qtable[state, :])
    #
    #             # Else doing a random choice --> exploration
    #             else:
    #                 action = env.action_space.sample()
    #
    #             # Take the action (a) and observe the outcome state(s') and reward (r)
    #             new_state, reward, done, info = env.step(action)
    #
    #             # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
    #             # qtable[new_state,:] : all the actions we can take from new state
    #             qtable[state, action] = qtable[state, action] + learning_rate * (
    #                 reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
    #
    #             total_rewards += reward
    #
    #             # Our new state is state
    #             state = new_state
    #
    #             # If done (if we're dead) : finish episode
    #             if done == True:
    #                 break
    #
    #         episode += 1
    #         # Reduce epsilon (because we need less and less exploration)
    #         epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    #         rewards.append(total_rewards)
    #
    #     print("Score over time: " + str(sum(rewards) / total_episodes))
    #     print(qtable)


if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)

    name = '10x10'
    rl = ReinforcementLearning(name)


