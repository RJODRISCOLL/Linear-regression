#!/usr/bin/env python
# coding: utf-8

# **Linear regression**
# </br>
# 
# We are manually writing the functions which will help us run a linear regression, the functions caluclate gradient descent. 
# 
# *y = mx + b* 
# 
# where m is the slope of the line, and b is the y-intercept 

# In[30]:


import os
import numpy as np 
import pandas as pd
import math, time, random, datetime


# In[32]:


os.chdir("/Users/Ruairi/Desktop")


# In[33]:


start_time = time.time() #for checking run time


def compute_error_for_line_given_points(b, m, points):
    totalError = 0 #inital starting points 
    for i in range(0, len(points)): #interate over every point in the data 
        x = points[i, 0] #get the pairs 
        y = points[i, 1] #get the pairs 
        totalError += (y - (m * x + b)) ** 2 # here we are adding to total error, the y value -the predicted (m *x+b)^2
    return totalError / float(len(points)) # return the average 

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0 #inital values 
    m_gradient = 0 #initial values 
    N = float(len(points)) #  N is the floated length of the data 
    for i in range(0, len(points)): # from 0 to last point 
        x = points[i, 0] # pairs 
        y = points[i, 1] # pairs 
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient) 
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
    return [b, m]

def run():
    points = genfromtxt("data.csv", delimiter=",") # points is our data 
    learning_rate = 0.0001 # learning rate, stops us overfitting data  
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000 # how many repeats 
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))

if __name__ == '__main__':
    run()
    log_time = (time.time() - start_time)
    print("Running Time: %s" % datetime.timedelta(seconds=log_time)) # how long does it take 


# In[ ]:




