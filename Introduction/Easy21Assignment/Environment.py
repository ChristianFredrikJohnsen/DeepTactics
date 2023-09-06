import numpy as np

class Easy21Environment:
    
    def __init__(self):
        """
        Initializes the environment.
        """

        # We have two players:
        self.player = []
        self.dealer = []
        self.dealer_sum = 0

        # The dimension of the action space is 2 (using the gym-framework):
        self.action_space = self.ActionSpace(2)

    def reset(self):
        """
        Resets the game, returns the initial state.
        State is represented as a tuple: (dealer_sum, player_sum)
        """

        # We empty the hands of both player and dealer.
        self.player = []
        self.dealer = []

        # Two cards to the player and one card to the dealer.
        self.deal_card(self.player, "black")
        self.deal_card(self.dealer, "black")
        
        # Storing the card of the dealer
        self.dealer_sum = self.hand_eval(self.dealer)

        return (self.dealer_sum, self.hand_eval(self.player))




    def step(self, action):
        """
        Executes action (0 = stick, and 1 = hit) and returns the next state of the game
        and the reward. The method also alerts when the game is over, by setting done = 1.
        Returns a tuple (next_state, reward, done)
        """

        if action == 1:
            self.deal_card(self.player)
            value = self.hand_eval(self.player)
            if (not value in range (1, 22)):
                return ((self.dealer_sum, value), -1, 1)
            else:
                return ((self.dealer_sum, value), 0, 0)

        elif action == 0:
            
            # Evaluating the player and dealer sums.
            player_sum = self.hand_eval(self.player)
            dealer_sum = self.player_stick()
            
            # Storing state representation as a tuple.
            dp = (dealer_sum, player_sum)

            if (dealer_sum not in range(17, 22)):
                return (dp, 1, 1)
            
            elif (dealer_sum == player_sum):
                return (dp, 0, 1)
            
            elif (dealer_sum > player_sum):
                return (dp, -1, 1)
            
            else:
                return (dp, 1, 1)

    
    def player_stick(self):
        
        while True:

            self.deal_card(self.dealer)
            self.dealer_sum = self.hand_eval(self.dealer)
            
            # We continue to draw cards until we land at something above 17, or we go bust at 0 or a negative number.
            if self.dealer_sum in range (1, 17):
                continue

            else:
                return self.dealer_sum
            


    def deal_card(self, player, color = "rand"):
        player.append(Card(color))

    def hand_eval(self, player):
        return sum(card.value if card.color == "black" else -card.value for card in player)

    class ActionSpace:
        """
        Making an ActionSpace-class, such that i have the same methods as in the Gym environment.
        """
        def __init__(self, n):
            self.n = n


class Card:

    def __init__(self, color = "rand"):
        self.value = np.random.randint(1, 11)
        if color == "rand":
            self.color = "red" if np.random.random() < 1/3 else "black"
        elif color == "black":
            self.color = "black"
        elif color == "red":
            self.color = "red"
        else:
            raise ValueError("Invalid color argument")
            

    

