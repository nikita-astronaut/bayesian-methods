# bayesian-methods
This directory contains codes of EM-algorithm and variational EM-algorithm.

## Villain face search
Villain's face is a noisy picture that is located at a random place of a noisy background. In total, one has ~300 such images with the same face and background, but with various noises and face locations. The task is to find face picture using Bayesian inference and EM--algorithm to find the minimum of the loss functional.

## Digits images clustering 
Given a lot of 8x8 images of digits, one should cluster them (like unsupervised machine learning). This can be done by introducing pi_ik --- probabilities for image i belong to class k and make variational bayesian inference. The inference is parametrized by the Dirichlet random process, with stick-breaking probabilities initialization.
