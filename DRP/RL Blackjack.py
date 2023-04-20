import random as rand


class Blackjack:
  def __init__(self):
    self.reset()

  def reset(self):
    self.player = []
    self.dealer = []
    #drawing to cards respectively 
    self.player.append(self.drawCard())
    self.player.append(self.drawCard())
    self.dealer.append(self.drawCard())
    self.dealer.append(self.drawCard())

  def drawCard(self):
    cardVal = rand.randint(1,13)
    if cardVal > 10:
      cardVal = 10
    return cardVal 

  def playerScore(self):
    score = sum(self.player)
    # player has usuable ace, and would bust, unless they converted it to a 1 
    if score > 21 and 11 in self.player:
      self.player[self.player.index(11)] = 1 
      score = sum(self.player)
    return score
 
  def dealerScore(self):
    score = sum(self.dealer)
    if score > 21 and 11 in self.dealer:
      self.dealer[self.dealer.index(11)] = 1 
      score = sum(self.dealer)
    return score


  def play(self, action):
    # when choosing to stand, dealer hits until they are at least at 17
    if action == "stand":
      while self.dealerScore() < 17: 
        self.dealer.append(self.drawCard())
      dealerScore = self.dealerScore()
      playerScore = self.playerScore()
      # reward distributions based on outcome of game
      if dealerScore > 21 or playerScore > dealerScore: 
        reward = 1
      elif dealerScore == playerScore:
        reward = 0
      else: 
        reward = -1
      term = True 

    # player chooses to hit  
    else:
      self.player.append(self.drawCard())
      currScore = self.playerScore()
      # if the player busts
      if currScore > 21:
        reward = -1 
        term = True
      # if the player does not bust 
      #no reward should be given for drawing new card
      else:
        reward = 0 
        term = False
    
    return reward, term

class MC_Sim:
  def __init__(self):
    self.rewards = {}
    self.policy = {}
    # list of estimated of the expected rewards of a given state-action pair
    self.Q = {}
    # the exploration probability 
    self.epsilon = 0.1

  def action(self, state):
    # if you haven't encounter this state-action -> make a random choice
    if state not in self.policy: 
      self.policy[state] = rand.choice(["hit", "stand"])
    # even if you know Q-value for this state, with probablity of Epsilon make 
    # a random choice between hit and stand (exploration)
    if rand.random() < self.epsilon:
      return rand.choice(["hit", "stand"])
    else:
      return self.policy[state]

  def updatePolicy(self):
    # based on esitmations (Q values) evaluate current action
     for state in self.Q:
      hitVal = self.Q[state]["hit"]
      standVal = self.Q[state]["stand"]
      if hitVal >= standVal:
        self.policy[state] = "hit"
      else:
        self.policy[state] = "stand"


  def playEpisodes(self, n):
    for episode in range(n):
      game = Blackjack()
      states = []
      actions = []
      rewards = []

      while True: 
        state = (game.playerScore(), game.dealer[0])
        action = self.action(state)
        reward, term = game.play(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        if term:
          break

      G = 0
      for i in range(len(states)-1, -1, -1):
        state = states[i]
        action = actions[i]
        reward = rewards[i]
        G += reward
        if state not in self.Q:
          self.Q[state] = {"hit": 0, "stand": 0}
        self.Q[state][action] += (G - self.Q[state][action]) / (i+1)
      self.updatePolicy()
  
  def printPolicy(self):
    print("Player Score\tDealer Score\t Policy")
    for playerScore in range(12,22):
      for dealerScore in range(1,11):
        state = (playerScore, dealerScore)
        if state in self.policy:
          policy = self.policy[state]
        else: 
          policy = "N/A"
        print(f"{playerScore}\t\t\t\t{dealerScore}\t\t\t\t {policy}")
          


sim = MC_Sim()
sim.playEpisodes(1000000)
print(sim.printPolicy())
