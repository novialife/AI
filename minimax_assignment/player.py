#!/usr/bin/env python3
import random
import numpy as np
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        nodes = initial_tree_node.compute_and_get_children()
        best_move = None
        best_score = -np.inf
        for node in nodes:
            score = self.minimax(node.state, 0)
            if score > best_score:
                best_score = score
                best_move = node.action
        
        return ACTION_TO_STR(best_move)
    
    def mu(player, state):
        if player == 0:
            return state.compute_and_get_children()
        else:
            return state.compute_and_get_children()
    
    def gamma(state, player):
        return state.compute_score(player)

    def minimax(state, player):
        if mu(state, player) == None:
            return gamma(state, player)

        else:
            if player == 0:
                best_possible = -np.inf 
                for child in mu(0, state):
                    v = minimax(child, 1)
                    best_possible = max(best_possible, v)
                return best_possible

            else:
                best_possible = np.inf 
                for child in mu(1, state):
                    v = minimax(child, 0)
                    best_possible = min(best_possible, v)
                return best_possible

            

