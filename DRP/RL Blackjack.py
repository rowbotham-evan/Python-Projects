import random
import numpy as np
from collections import defaultdict

class Environment:
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

class Blackjack:
    def __init__(self, num_decks=1, dealer_hits_on_soft_17=False):
        self.n_decks = num_decks
        self.dealer_hits_on_soft_17 = dealer_hits_on_soft_17
        self.reset()

    def reset(self):
        # Reset the environment to its initial state
        self.dealer_hand = self._deal_cards()
        self.player_hand = self._deal_cards()
        self.player_ace_count = self._count_aces(self.player_hand)
        self.player_sum = self._sum_hand(self.player_hand, self.player_ace_count)
        self.dealer_sum = self._sum_hand(self.dealer_hand, 0)
        self.done = False

    def _deal_cards(self):
        # Deal two cards from the deck
        deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4 * self.n_decks
        random.shuffle(deck)
        return [deck.pop(), deck.pop()]

    def _sum_hand(self, hand, ace_count):
        # Calculate the sum of the hand, treat aces AS 11 (granted the hand won't bust)
        hand_sum = sum(hand)
        for i in range(ace_count):
            if hand_sum > 21:
                hand_sum -= 10
        return hand_sum

    def _count_aces(self, hand):
        # Count the number of aces in the hand
        return hand.count(11)

    def step(self, action):
        assert not self.done

        if action == 0:
            self.done = True
            while self.dealer_sum < 17 or (self.dealer_sum == 17 and self.dealer_hits_on_soft_17 and 11 in self.dealer_hand):
                card = self._deal_cards()[0]
                self.dealer_hand.append(card)
                self.dealer_sum = self._sum_hand(self.dealer_hand, self._count_aces(self.dealer_hand))
            if self.dealer_sum > 21 or self.player_sum > self.dealer_sum:
                reward = 1
            elif self.player_sum == self.dealer_sum:
                reward = 0
            else:
                reward = -1
        else:  
            card = self._deal_cards()[0]
            self.player_hand.append(card)
            self.player_ace_count += self._count_aces([card])
            self.player_sum = self._sum_hand(self.player_hand, self.player_ace_count)
            if self.player_sum > 21:
                self.done = True
                reward = -1
            else:
                reward = 0

        return (self.player_sum, self.dealer_hand[0], int(self.player_ace_count > 0)), reward, self.done

def monte_carlo(env, episodes, gamma):
    V = np.zeros(env.n_states)

    # Initialize the sum of returns and count of visits
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    
    for i in range(episodes):
    # Generate an episode by playing a game of Blackjack
    episode = []
    state = env.reset()
    done = False
    while not done:
        action = np.random.randint(2)
        next_state, reward, done = env.step(action)
        episode.append((state, action, reward))
        state = next_state

    # Calculate the returns for each state in the episode
    G = 0
    for t in reversed(range(len(episode))):
        state, action, reward = episode[t]
        G = gamma * G + reward
        returns_sum[state] += G
        returns_count[state] += 1

   # Update the state-value function
    for state in returns_sum:
    V[state] = returns_sum[state] / returns_count[state]

    return V
