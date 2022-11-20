#!/usr/bin/env python3
import random
import numpy as np
import math
from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
import time

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
            self.root_node = Node(message=msg, player=0)
            self.Zobrist_table = self.init_table()
            self.Zobrist_table_hash = {}
            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=self.root_node)

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
        initial_time = time.time()
        nodes = initial_tree_node.compute_and_get_children()
        nodes.sort(key=self.compute_heuristic, reverse=True)
        alpha = -np.inf
        beta = np.inf
        highScore = -np.inf
        best_move = 0
        for node in nodes:
            player = node.state.get_player()
            score = self.minimax(node, 5, alpha, beta, player, initial_time)
            if (score > highScore):
                highScore = score
                best_move = node.move
        #print("Best move: ", ACTION_TO_STR[best_move])
        return ACTION_TO_STR[best_move]

    def init_table(self):
        # Initialize a Zobrist table
        # The table is a 2D array of size 20x20
        # Each cell contains a random 64-bit integer
        
        zobrist_table = np.zeros((20, 20), dtype=np.uint64)
        for i in range(20):
            for j in range(20):
                zobrist_table[i][j] = random.getrandbits(64)
        return zobrist_table
    
    def compute_zobrist_hash(self, node):
        # Compute the Zobrist hash of a node
        # The hash is the XOR of the hashes of the positions of the boats
        # and the hash of the position of the fish

        hash_value = 0
        for hook in node.state.get_hook_positions().values():
            hash_value ^= int(self.Zobrist_table[hook[0]][hook[1]])
        for fish in node.state.get_fish_positions().values():
            hash_value ^= int(self.Zobrist_table[fish[0]][fish[1]])

        return hash_value

    def minimax(self, node, depth, alpha, beta, player, initial_time):

        zobrist_hash = self.compute_zobrist_hash(node)
        if zobrist_hash in self.Zobrist_table_hash:
            return self.Zobrist_table_hash[zobrist_hash]
        
        if time.time() - initial_time > 0.055:
            return self.compute_heuristic(node)

        if depth == 0 or node.compute_and_get_children() == []:
            return self.compute_heuristic(node)

        nodes = node.compute_and_get_children()
        nodes.sort(key=self.compute_heuristic, reverse=True) if player == 0 else nodes.sort(key=self.compute_heuristic, reverse=False)

        if player == 0:
            value = -np.inf
            for child in nodes:
                value = max(value, self.minimax(child, depth-1, alpha, beta, 1, initial_time))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            self.Zobrist_table_hash[zobrist_hash] = value
            return value
        else:
            value = np.inf
            for child in nodes:
                value = min(value, self.minimax(child, depth-1, alpha, beta, 0, initial_time))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            self.Zobrist_table_hash[zobrist_hash] = value
            return value

    def compute_heuristic(self, node):
        player_pos = node.state.get_hook_positions()    # dict of x, y tuple
        fish_pos = node.state.get_fish_positions()    # dict of x, y tuple
        fish_scores = node.state.get_fish_scores()      # dict of scores
        score_diff = node.state.player_scores[0] - node.state.player_scores[1]
        
        h = 0
        for fish in fish_pos:
            dx_0 = min(abs(player_pos[0][0] - fish_pos[fish][0]), 20 - abs(player_pos[0][0] - fish_pos[fish][0]))
            dx_1 = min(abs(player_pos[1][0] - fish_pos[fish][0]), 20 - abs(player_pos[1][0] - fish_pos[fish][0]))

            dy_0 = abs(player_pos[0][1] - fish_pos[fish][1])
            dy_1 = abs(player_pos[1][1] - fish_pos[fish][1])

            d_max = dx_0 + dy_0 + 1
            d_min = dx_1 + dy_1 + 1

            h += fish_scores[fish] * (1/d_max - 1/ d_min)
        
        return h + score_diff