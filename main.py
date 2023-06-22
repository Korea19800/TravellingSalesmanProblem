import numpy as np, random, operator, pandas as pd
import sys

class City:
    #index is used instead of city name. 
    #ex.in tsp.txt index 2 indicates Chicgao  
    def __init__(self, index, distList):
        self.index = index 
        self.distList = distList
   
      
    #finding distance using the input matrix 
    def distance(self, toCity):
        distance = self.distList[self.index][toCity.index]    #distList[fromCity][toCity]
        return distance
      
class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness= 0.0
    #finding distance of route using distances between cities in the route
    def routeDistance(self):
        if self.distance ==0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance
      
    #shorter route has better fitness
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

# creating route by randomly choosing n cities(n is the number of cities in input)
def createRoute(cityList):
    route = random.sample(cityList, len(cityList))

    return route # route = [city1, city2, ...]

def initialPopulation(popSize, cityList):
    population = []
    # population consists of routes. routes consists of cities
    for i in range(0, popSize):
        population.append(createRoute(cityList))
    return population

def rankRoutes(population):
    # a dictionary containing route id and its fitness. ex. route1 : 24.5
    fitnessResults = {} 

    for i in range(0,len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    
    # sort each route's fitess value in the fitnessResults in decreasing order
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)

#select based on probability
def selection(popRanked, eliteSize):
    selectionResults = []
    # make roulette whell by calculating relative fitness weigh of each individual
    df = pd.DataFrame(np.array(popRanked), columns=["Index","Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100*df.cum_sum/df.Fitness.sum()

    #use elitisim: best performing individuals from the population will carry over to the new generation
    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100*random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i,3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults # list of routeID, which will be used to create mating pool

# a collection of parents used to create new population 
def matingPool(population, selectionResults):
    #matingpool is a list containing n(popsize) individual(route)
    matingpool = []
    
    #extracting the selected individuals from the population.
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool

#crossover
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    #randomly select a subset of the first parent string, and then fill the remainder  with the genes from the second parent 
    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))
    
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child

# create a children population
def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))
    
    # use elitism to get best routes from current population
    for i in range(0,eliteSize):
        children.append(matingpool[i])
    
    for i in range(0, length):
        #fill out rest of the new population
        child = breed(pool[i], pool[len(matingpool)-i-1])
        children.append(child)
    return children

# mutation using swap 
def mutate(individual, mutationRate):
    #swap 2 random cities
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))
            
            city1 = individual[swapped]
            city2 = individual[swapWith]
            
            individual[swapped] = city2
            individual[swapWith] = city1
    return individual

# extend mutate function to run through new population
def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

# make a new population
def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations,cityName):
    # creating population
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / rankRoutes(pop)[0][1]))

    # printing result of each generation
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        print("generation" + str(i+1) + ":" + str(1 / rankRoutes(pop)[0][1]))
    # printing final distance 
    print("Final distance: " + str(1 / rankRoutes(pop)[0][1]))
    
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]

    # printing the name of cities in the best route
    print("Best route: ")
    for city in bestRoute:  
      print(cityName[city.index], end = ' ')
      
    return bestRoute

# main function
def main():
  
    # getting name of a file
    print("Enter a file name: ")
    fname = input()  
  
    # catching error when user writes wrong file name
    try:
        file = open(fname, 'r')    
        s = file.read()
        file.close()   
        fileList = s.split()  #splice text in file before ' '
    
    except FileNotFoundError:
          print("Cannot find the file. Ending the program")
          sys.exit(1)
    else:
        print("Successfuly load the file\n")
      
    #number of cities
    num = int(fileList[0])
  
    #name of cities
    cityName = []
    for i in range(1,num+1):
        cityName.append(fileList[i])

    #dist between cites : 1D
    distList = []
    for i in range(num+1,len(fileList)):
        distList.append(fileList[i])
    distList = list(map(int, distList))
    
    #dist between cites : 2D
    distList2 = [[0 for j in range(num)] for i in range(num)]
    for i in range(len(distList)):
        row = (int)(i / num)
        col = (int)(i % num)
        distList2[row][col] = distList[i]
    
    # Priting input data
    print("Initial Information :\n")
    print(num)
    for i in cityName:
      print(i)
    for i in range(num):
        for j in range(num):
            print(distList2[i][j], end = ' ')
        print()
    
    print("\nCalculating...\n")
    
    #creating cityList which is used for population
    cityList = []  
    for i in range(0,num):
        cityList.append(City(i,distList2))
      
    #selecting population and generation size
    geneticAlgorithm(population=cityList, popSize=1000, eliteSize=10, mutationRate=0.01, generations=50, cityName = cityName)

if __name__ == "__main__":
    main()





