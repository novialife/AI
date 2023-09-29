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

            if step >= 110:
                fish_id, obs = self.fishes.pop()
                guesses = []
                for i in self.models:
                    model = self.models[i]
                    model.T = len(obs)
                    model.O = obs
                    guesses.append(model.guess(obs))
                
                guess = guesses.index(max(guesses))
                return fish_id, guess

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
        

