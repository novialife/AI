#!/usr/bin/env python3

from player_controller_hmm import PlayerControllerHMMAbstract
from constants import *
import random
import numpy as np
from HMM import *
import sys


class PlayerControllerHMM(PlayerControllerHMMAbstract):
    def init_parameters(self):
        """
        In this function you should initialize the parameters you will need,
        such as the initialization of models, or fishes, among others.
        """        

        self.models = {}
        for fish in range(N_SPECIES):
            self.models[fish] = HMM()
            self.models[fish].init_parameters(1, N_EMISSIONS)
        
        self.fishes = [(i, []) for i in range(N_FISH)]
        self.curr_fish_id = 0

    def guess(self, step, observations):
            """
            This method gets called on every iteration, providing observations.
            Here the player should process and store this information,
            and optionally make a guess by returning a tuple containing the fish index and the guess.
            :param step: iteration number
            :param observations: a list of N_FISH observations, encoded as integers
            :return: None or a tuple (fish_id, fish_type)
            """

            for i in range(len(self.fishes)):
                self.fishes[i][1].append(observations[i])

            if step < 110:      # 110 = 180 timesteps - 70 guesses
                return None
            else:
                fish_id, obs = self.fishes.pop()
                fish_type = 0
                max = 0
                for model, j in zip(list(self.models.values()), range(N_SPECIES)):
                    model.T = len(obs)
                    model.O = obs
                    m = model.guess(obs)
                    if m > max:
                        max = m
                        fish_type = j
                self.obs = obs
                return fish_id, fish_type

    def reveal(self, correct, fish_id, true_type):
        """
        This methods gets called whenever a guess was made.
        It informs the player about the guess result
        and reveals the correct type of that fish.
        :param correct: tells if the guess was correct
        :param fish_id: fish's index
        :param true_type: the correct type of the fish
        :return:
        """

        if not correct:
            self.models[true_type].baum_welch(50)

