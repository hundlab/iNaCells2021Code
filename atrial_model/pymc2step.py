#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 13:46:15 2020

@author: grat05
"""

import numpy as np
from numpy import ndim, ones, zeros, log, shape, cov, ndarray, inner, reshape, sqrt, any, array, all, abs, exp, where, isscalar, iterable, multiply, transpose, tri, pi
from numpy.random import normal as rnormal
from numpy.random import random

from pymc import StepMethod
from pymc.six import print_
from pymc.Node import ZeroProbability

class AdaptiveSDMetropolis(StepMethod):

    """
    The default StepMethod, which Model uses to handle singleton, continuous variables.
    Applies the one-at-a-time Metropolis-Hastings algorithm to the Stochastic over which self has jurisdiction.
    To instantiate a Metropolis called M with jurisdiction over a Stochastic P:
      >>> M = Metropolis(P, scale=1, proposal_sd=None, dist=None)
    :Arguments:
    - s : Stochastic
            The variable over which self has jurisdiction.
    - scale (optional) : number
            The proposal jump width is set to scale * variable.value.
    - proposal_sd (optional) : number or vector
            The proposal jump width is set to proposal_sd.
    - proposal_distribution (optional) : string
            The proposal distribution. May be 'normal' (default) or
            'prior'.
    - verbose (optional) : integer
            Level of output verbosity: 0=none, 1=low, 2=medium, 3=high. Setting to -1 (default) allows verbosity to be turned on by sampler.
    :SeeAlso: StepMethod, Sampler.
    """

    def __init__(self, stochastic, scale=1., proposal_sd=None,
                 proposal_distribution='normal', verbose=-1, tally=True, check_before_accepting=True,
                 delay=1000, interval=200, greedy=True):
        # Metropolis class initialization

        # Initialize superclass
        StepMethod.__init__(self, [stochastic], tally=tally)
        
        self.stochastic = stochastic

        # Initialize hidden attributes
        self.proposal_sd = proposal_sd
        self._trace = []
        self._current_iter = 0
        
        # Number of successful steps before the empirical covariance is
        # computed
        self.delay = delay
        # Interval between covariance updates
        self.interval = interval
        # Flag for tallying only accepted jumps until delay reached
        self.greedy = greedy

        self.adaptive_scale_factor = np.ones(self.stochastic.value.shape[0])
        self.accepted = np.zeros(self.stochastic.value.shape[0])
        self.rejected = np.zeros(self.stochastic.value.shape[0])
        self._state = [
            'rejected',
            'accepted',
            'adaptive_scale_factor',
            'proposal_sd',
            'proposal_distribution',
            'check_before_accepting',
            '_trace']
        self._tuning_info = ['adaptive_scale_factor', 'proposal_sd']
        self.check_before_accepting = check_before_accepting

        # Set public attributes
        if verbose > -1:
            self.verbose = verbose
        else:
            self.verbose = stochastic.verbose
           
        prop_dist = proposal_distribution.lower() 
        if prop_dist in self.valid_proposals:
            self.proposal_distribution = proposal_distribution
        else:
            raise ValueError(
                "Invalid proposal distribution '%s' specified for Metropolis sampler." %
                proposal_distribution)

        if prop_dist != "prior":
            # Avoid zeros when setting proposal variance
            if proposal_sd is None:
                if all(self.stochastic.value != 0.):
                    self.proposal_sd = ones(
                        shape(
                            self.stochastic.value)) * abs(
                                self.stochastic.value) * scale
                else:
                    self.proposal_sd = ones(shape(
                        self.stochastic.value)) * scale

            # Initialize proposal deviate with array of zeros
            self.proposal_deviate = zeros(
                shape(self.stochastic.value),
                dtype=float)

            # Determine size of stochastic
            if isinstance(self.stochastic.value, ndarray):
                self._len = len(self.stochastic.value.ravel())
            else:
                self._len = 1

    valid_proposals = property(lambda self: ('normal', 'prior'))

    @staticmethod
    def competence(s):
        return 0

    def hastings_factor(self):
        """
        If this is a Metropolis-Hastings method (proposal is not symmetric random walk),
        this method should return log(back_proposal) - log(forward_proposal).
        """
        return 0.

    def step(self):
        """
        The default step method applies if the variable is floating-point
        valued, and is not being proposed from its prior.
        """

        # Probability and likelihood for s's current value:

        if self.verbose > 2:
            print_()
            print_(self._id + ' getting initial logp.')

        if self.proposal_distribution == "prior":
            logp = self.loglike
        else:
            logp = self.logp_plus_loglike

        if self.verbose > 2:
            print_(self._id + ' proposing.')

        old_value = self.stochastic.value.copy()
        # Sample a candidate value
        self.propose()
        new_value = self.stochastic.value.copy()

        # Probability and likelihood for s's proposed value:
        try:
            if self.proposal_distribution == "prior":
                logp_p = self.loglike
                # Check for weirdness before accepting jump
                if self.check_before_accepting:
                    self.stochastic.logp
            else:
                logp_p = self.logp_plus_loglike

        except ZeroProbability:

            # Reject proposal
            if self.verbose > 2:
                print_(self._id + ' rejecting due to ZeroProbability.')
            self.reject()

            # Increment rejected count
            self.rejected += 1

            if self.verbose > 2:
                print_(self._id + ' returning.')
            return

        if self.verbose > 2:
            print_('logp_p - logp: ', logp_p - logp)

        self.stochastic.revert()
        
        rejected_all = True
        accepted_proposal = old_value.copy()
        for i in range(self.stochastic.value.shape[0]):
            new_single = old_value.copy()
            new_single[i] = new_value[i]
            self.stochastic.value = new_single
            logp_p = self.logp_plus_loglike
            
            # Evaluate acceptance ratio
            if log(random()) <= logp_p - logp:
                accepted_proposal[i] = new_value[i]
                # Increment accepted count
                self.accepted[i] += 1
                rejected_all = False

                if self.verbose > 2:
                    print_(self._id + ' accepting' + str(i))
            else:
                self.rejected[i] += 1
                if self.verbose > 2:
                    print_(self._id + ' rejecting' + str(i))
            self.reject()
        if not rejected_all:
            self.stochastic.value = accepted_proposal

        if self.verbose > 2:
            print_(self._id + ' returning.')
        
        if self._current_iter > self.delay:
            self.internal_tally()
        
        if self._current_iter > self.delay and\
            (self._current_iter - self.delay) % self.interval == 0:
            self.updateproposal_sd()
            
        self._current_iter += 1
        
                #         # Revert s if fail
                # self.reject()
                # if not self.greedy:
                #     self.internal_tally()
    
                # # Increment rejected count
                # self.rejected += 1
                # if self.verbose > 2:
                #     print_(self._id + ' rejecting')

    def reject(self):
        # Sets current s value to the last accepted value
        # self.stochastic.value = self.stochastic.last_value
        self.stochastic.revert()

    def propose(self):
        """
        This method is called by step() to generate proposed values
        if self.proposal_distribution is "Normal" (i.e. no proposal specified).
        """

        prop_dist = self.proposal_distribution.lower()
        if  prop_dist == "normal":
            self.stochastic.value = rnormal(
                self.stochastic.value,
                self.adaptive_scale_factor[...,None] *
                self.proposal_sd,
                size=self.stochastic.value.shape)
        elif prop_dist == "prior":
            self.stochastic.random()

    def tune(self, divergence_threshold=1e10, verbose=0):
        return False
        """
        Tunes the scaling parameter for the proposal distribution
        according to the acceptance rate of the last k proposals:
        Rate    Variance adaptation
        ----    -------------------
        <0.001        x 0.1
        <0.05         x 0.5
        <0.2          x 0.9
        >0.5          x 1.1
        >0.75         x 2
        >0.95         x 10
        This method is called exclusively during the burn-in period of the
        sampling algorithm.
        May be overridden in subclasses.
        """

        if self.verbose > -1:
            verbose = self.verbose

        # Verbose feedback
        if verbose > 0:
            print_('\t%s tuning:' % self._id)

        # Flag for tuning state
        tuning = True

        # Calculate recent acceptance rate
        if not (self.accepted + self.rejected):
            return tuning
        acc_rate = self.accepted / (self.accepted + self.rejected)
        
        current_factor = self.adaptive_scale_factor

        # Switch statement
        if acc_rate < 0.001:
            # reduce by 90 percent
            self.adaptive_scale_factor *= 0.1
        elif acc_rate < 0.05:
            # reduce by 50 percent
            self.adaptive_scale_factor *= 0.5
        elif acc_rate < 0.2:
            # reduce by ten percent
            self.adaptive_scale_factor *= 0.9
        elif acc_rate > 0.95:
            # increase by factor of ten
            self.adaptive_scale_factor *= 10.0
        elif acc_rate > 0.75:
            # increase by double
            self.adaptive_scale_factor *= 2.0
        elif acc_rate > 0.5:
            # increase by ten percent
            self.adaptive_scale_factor *= 1.1
        else:
            tuning = False

        # Re-initialize rejection count
        self.rejected = 0.
        self.accepted = 0.
        
        # Prevent from tuning to zero
        if not self.adaptive_scale_factor:
            self.adaptive_scale_factor = current_factor
            return False

        # More verbose feedback, if requested
        if verbose > 0:
            if hasattr(self, 'stochastic'):
                print_('\t\tvalue:', self.stochastic.value)
            print_('\t\tacceptance rate:', acc_rate)
            print_('\t\tadaptive scale factor:', self.adaptive_scale_factor)
            print_()

        return tuning


    def updateproposal_sd(self):
        chain = np.asarray(self._trace)
        pre_proposal_sd = self.proposal_sd
        self.proposal_sd = 1/4*np.std(chain, axis=0)+ 3/4*pre_proposal_sd
        zero_mask = self.proposal_sd == 0
        self.proposal_sd[zero_mask] = 0.1*pre_proposal_sd[zero_mask]
        
        # avg_change = np.mean(self.proposal_sd/pre_proposal_sd)
        # avg_change /= 2
        
        # self.adaptive_scale_factor /= avg_change
        
        acc_rate = self.accepted / (self.accepted + self.rejected)
        
        scale = np.ones_like(acc_rate)
        
        # reduce by ten percent
        scale[acc_rate < 0.2] = 0.9
        # reduce by 50 percent
        scale[acc_rate < 0.05] = 0.5
        # reduce by 90 percent
        scale[acc_rate < 0.001] = 0.1
        # increase by ten percent
        scale[acc_rate > 0.5] = 1.1
        # increase by double
        scale[acc_rate > 0.75] = 2.0
        # increase by factor of ten
        scale[acc_rate > 0.95] = 10.0
        
        self.adaptive_scale_factor *= scale


        # Re-initialize rejection count
        self.rejected = np.zeros_like(self.rejected)
        self.accepted = np.zeros_like(self.accepted)
        
    def internal_tally(self):
        """Store the trace of stochastics for the computation of the covariance.
        This trace is completely independent from the backend used by the
        sampler to store the samples."""
        self._trace.append(self.stochastic.value)
        
