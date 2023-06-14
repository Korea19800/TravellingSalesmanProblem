# TravellingSalesmanProblem

Python code to solve TSP using an genetic algorithm

Description:
The Traveling Salesman problem (TSP) is a famous problem in Computer Science, both because
it is a very hard problem to solve (one of a class called NP-complete), and because there are many
approximation algorithms to find a decent solution to it. TSP can be quickly summarized by
assuming that there are N cities, with a cost of travel from every city to every other city defined.
TSP asks us to find the lowest cost path from a given city back to that city that passes through
every other city once and only once.
One way of obtaining a solution to a TSP problem is to use a genetic algorithm. We will be
implementing a genetic algorithm to find a solution to TSP problems. 

The Program:

 Prompt the user to enter a file name.


 Open the file and read in the information about the TSP. If the file doesn’t exist, it should
print an appropriate error message and exit the program. If it does, print the information from
the file (the list of names and grid of costs, see below).


 Run a genetic algorithm to find the best path for the TSP. While running, the program
should print the cost of the best tour it has found during each generation.


 When the program is finished, print the best path found, and the resulting path cost
