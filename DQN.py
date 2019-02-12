import gym
import tensorflow as tf
import numpy as np
import random
from collections import deque

# Hyper Parameters for DQN
GAMMA = 0.9 # discount factor for target Q
INITIAL_EPSILON = 0.5 # starting value of epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
REPLAY_SIZE = 10000 # experience replay buffer size
BATCH_SIZE = 32 # size of minibatch

class DQN():
  # DQN Agent
  def __init__(self, env):
    # init experience replay
    self.replay_buffer = deque()  #用deque定义一个双端队列
    # init some parameters
    self.time_step = 0
    self.epsilon = INITIAL_EPSILON
    self.state_dim = env.observation_space.shape[0]  #用gym定义状态向量的长度
    self.action_dim = env.action_space.n  #用ygm定义动作的数量

    self.create_Q_network()
    self.create_training_method()

    # Init session
    self.session = tf.InteractiveSession()
    self.session.run(tf.global_variables_initializer())

  def create_Q_network(self):
    # network weights
    W1 = self.weight_variable([self.state_dim,20])  #与状态变量相乘的权重
    b1 = self.bias_variable([20]) #各神经元的偏置
    W2 = self.weight_variable([20,self.action_dim]) #隐藏层之后，计算各动作的Q值（输出）所需的权重，所以顺序和上面相反
    b2 = self.bias_variable([self.action_dim])
    # input layer
    self.state_input = tf.placeholder("float",[None,self.state_dim])  #定义的输入状态矩阵，行数不定，列数为state_dim
    # hidden layers
    h_layer = tf.nn.relu(tf.matmul(self.state_input,W1) + b1) #隐藏层矩阵：样本数*20
    # Q Value layer
    self.Q_value = tf.matmul(h_layer,W2) + b2  #Q_value矩阵：样本数*动作数

  def create_training_method(self):
    self.action_input = tf.placeholder("float",[None,self.action_dim]) # one hot presentation
    self.y_input = tf.placeholder("float",[None])
    #tf.multiply后：样本数*动作数（action_input为one-hot格式，最后只有被选定的动作对应的值被保留，其余都为0），reduce_sum后：1*样本数（横向相加合并、转置）
    Q_action = tf.reduce_sum(tf.multiply(self.Q_value,self.action_input),reduction_indices = 1)
    self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))#计算所有样本的均方差作为损失
    self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

  def perceive(self,state,action,reward,next_state,done):
    one_hot_action = np.zeros(self.action_dim)
    one_hot_action[action] = 1
    self.replay_buffer.append((state,one_hot_action,reward,next_state,done))
    if len(self.replay_buffer) > REPLAY_SIZE:
      self.replay_buffer.popleft() #样本数大于最高，从左侧删除一个

    if len(self.replay_buffer) > BATCH_SIZE:
      self.train_Q_network() #样本数大于32，就开始训练

  def train_Q_network(self):
    self.time_step += 1
    # Step 1: obtain random minibatch from replay memory
    minibatch = random.sample(self.replay_buffer,BATCH_SIZE) #把抽取样本的各个属性分类汇总，待用
    state_batch = [data[0] for data in minibatch]
    action_batch = [data[1] for data in minibatch]
    reward_batch = [data[2] for data in minibatch]
    next_state_batch = [data[3] for data in minibatch]

    # Step 2: calculate y
    # 对每个样本，用下一状态的Q值计算当前的Q值，再和用Q网络直接计算出的当前Q值作比较，取得误差并反向训练网络
    y_batch = []
    Q_value_batch = self.Q_value.eval(feed_dict={self.state_input:next_state_batch})
    for i in range(0,BATCH_SIZE):
      done = minibatch[i][4]
      if done:
        y_batch.append(reward_batch[i])
      else :
        y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

    self.optimizer.run(feed_dict={
      self.y_input:y_batch,
      self.action_input:action_batch,
      self.state_input:state_batch
      })

  def egreedy_action(self,state): #ϵ−贪婪法求下一动作
    Q_value = self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0]
    if random.random() <= self.epsilon:
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        return random.randint(0,self.action_dim - 1)
    else:
        self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
        return np.argmax(Q_value)

  def action(self,state):  #贪婪法求下一动作
    return np.argmax(self.Q_value.eval(feed_dict = {
      self.state_input:[state]
      })[0])

  def weight_variable(self,shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)

  def bias_variable(self,shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)
# ---------------------------------------------------------
# Hyper Parameters
ENV_NAME = 'CartPole-v0'
EPISODE = 3000 # Episode limitation
STEP = 300 # Step limitation in an episode
TEST = 10 # The number of experiment test every 100 episode

def main():
  # initialize OpenAI Gym env and dqn agent
  env = gym.make(ENV_NAME)
  agent = DQN(env) #用env实例化DQN类

  for episode in range(EPISODE):
    # initialize task
    state = env.reset() #每个episode开始时，状态初始化
    # Train
    for step in range(STEP):
      action = agent.egreedy_action(state) 
      next_state,reward,done,_ = env.step(action) #用gym得到下一状态
      # Define reward for agent
      reward = -1 if done else 0.1
      agent.perceive(state,action,reward,next_state,done) #包含了所有存储和训练
      state = next_state
      if done:
        break
    # Test every 100 episodes
    if episode % 100 == 0:
      total_reward = 0
      for i in range(TEST):
        state = env.reset()
        for j in range(STEP):
          env.render() #将当前画面展示出来
          action = agent.action(state) # 测试无需探索，直接贪婪法看结果
          state,reward,done,_ = env.step(action)
          total_reward += reward
          if done:
            break
      ave_reward = total_reward/TEST
      print ('episode: ',episode,'Evaluation Average Reward:',ave_reward)

#if __name__ == '__main__':
main()