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
            self.models[fish].init_parameters(N_SPECIES, N_EMISSIONS)
        
        self.opps = {}
        self.seen = {}
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

        guesses = []
        for fish in range(N_FISH):
            try:
                self.opps[fish].append(observations[fish])
            except:
                self.opps[fish] = [observations[fish]]
        
        if step > 10:
            for fish_species in range(N_SPECIES):
                self.models[fish_species].T = len(self.opps[self.curr_fish_id])
                self.models[fish_species].O = self.opps[self.curr_fish_id]
                alpha = self.models[fish_species].efficient_forward()[0]
                guesses.append(np.sum(alpha[-1]))
            
            guess = np.argmax(guesses)
            return(self.curr_fish_id, guess)
        else:
            return None

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
        if true_type not in self.seen:
            self.seen[true_type] = 1
            self.models[true_type].baum_welch(200) 

        self.curr_fish_id += 1
