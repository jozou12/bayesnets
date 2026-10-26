# factorOperations.py
# -------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from typing import List
from bayesNet import Factor
import functools
from util import raiseNotDefined

def joinFactorsByVariableWithCallTracking(callTrackingList=None):


    def joinFactorsByVariable(factors: List[Factor], joinVariable: str):
        """
        Input factors is a list of factors.
        Input joinVariable is the variable to join on.

        This function performs a check that the variable that is being joined on 
        appears as an unconditioned variable in only one of the input factors.

        Then, it calls your joinFactors on all of the factors in factors that 
        contain that variable.

        Returns a tuple of 
        (factors not joined, resulting factor from joinFactors)
        """

        if not (callTrackingList is None):
            callTrackingList.append(('join', joinVariable))

        currentFactorsToJoin =    [factor for factor in factors if joinVariable in factor.variablesSet()]
        currentFactorsNotToJoin = [factor for factor in factors if joinVariable not in factor.variablesSet()]

        # typecheck portion
        numVariableOnLeft = len([factor for factor in currentFactorsToJoin if joinVariable in factor.unconditionedVariables()])
        if numVariableOnLeft > 1:
            print("Factors failed joinFactorsByVariable typecheck: ", currentFactorsToJoin)
            raise ValueError("The joinBy variable can only appear in one factor as an \nunconditioned variable. \n" +  
                               "joinVariable: " + str(joinVariable) + "\n" +
                               ", ".join(map(str, [factor.unconditionedVariables() for factor in currentFactorsToJoin])))
        
        joinedFactor = joinFactors(currentFactorsToJoin)
        return currentFactorsNotToJoin, joinedFactor

    return joinFactorsByVariable

joinFactorsByVariable = joinFactorsByVariableWithCallTracking()

########### ########### ###########
########### QUESTION 2  ###########
########### ########### ###########

def joinFactors(factors: List[Factor]):
    """
    Input factors is a list of factors.  
    
    You should calculate the set of unconditioned variables and conditioned 
    variables for the join of those factors.

    Return a new factor that has those variables and whose probability entries 
    are product of the corresponding rows of the input factors.

    You may assume that the variableDomainsDict for all the input 
    factors are the same, since they come from the same BayesNet.

    joinFactors will only allow unconditionedVariables to appear in 
    one input factor (so their join is well defined).

    Hint: Factor methods that take an assignmentDict as input 
    (such as getProbability and setProbability) can handle 
    assignmentDicts that assign more variables than are in that factor.

    Useful functions:
    Factor.getAllPossibleAssignmentDicts
    Factor.getProbability
    Factor.setProbability
    Factor.unconditionedVariables
    Factor.conditionedVariables
    Factor.variableDomainsDict
    """

    factors = list(factors)

    # typecheck portion
    setsOfUnconditioned = [set(factor.unconditionedVariables()) for factor in factors]
    if len(factors) > 1:
        intersect = functools.reduce(lambda x, y: x & y, setsOfUnconditioned)
        if len(intersect) > 0:
            print("Factors failed joinFactors typecheck: ", factors)
            raise ValueError("unconditionedVariables can only appear in one factor. \n"
                    + "unconditionedVariables: " + str(intersect) + 
                    "\nappear in more than one input factor.\n" + 
                    "Input factors: \n" +
                    "\n".join(map(str, factors)))

    if len(factors) == 0: # checking if the list of factors is empty
        raise ValueError("cannot join an empty list of factors")

    unconditioned_set = set() # creating a set for all the unconditioned variables 
    conditioned_set = set() # creating a set for all the conditioned variables

    # going through each factor in the list 
    for factor in factors: 
        unconditioned_variables = factor.unconditionedVariables() # getting the unconditioned variables of the current factor
        conditioned_variables = factor.conditionedVariables() # getting the conditioned variables of the current factor
        unconditioned_set.update(unconditioned_variables) # adding the unconditioned variables to the set
        conditioned_set.update(conditioned_variables) # adding the conditioned variables to the set

    conditioned_set = conditioned_set - unconditioned_set # making sure to remove any variables that are also unconditioned

    variable_domains_dict = factors[0].variableDomainsDict() # getting the variable domains from the first factor

    new_factor = Factor(unconditioned_set, conditioned_set, variable_domains_dict) # creating a new factor with the combined variables
    all_assignments = new_factor.getAllPossibleAssignmentDicts() # getting all possible assignments for the new factor
    
    # going through each possible assignment
    for assignment in all_assignments: 
        probability = 1 # starting the probability at 1 to begin multiplying

        for factor in factors: # going through each factor
            probability *= factor.getProbability(assignment) # multiplying the probability for the current assignment

        new_factor.setProbability(assignment, probability) # setting the calculated probability in the new factor
    
    return new_factor # returning the new factor

########### ########### ###########
########### QUESTION 3  ###########
########### ########### ###########

def eliminateWithCallTracking(callTrackingList=None):

    def eliminate(factor: Factor, eliminationVariable: str):
        """
        Input factor is a single factor.
        Input eliminationVariable is the variable to eliminate from factor.
        eliminationVariable must be an unconditioned variable in factor.
        
        You should calculate the set of unconditioned variables and conditioned 
        variables for the factor obtained by eliminating the variable
        eliminationVariable.

        Return a new factor where all of the rows mentioning
        eliminationVariable are summed with rows that match
        assignments on the other variables.

        Useful functions:
        Factor.getAllPossibleAssignmentDicts
        Factor.getProbability
        Factor.setProbability
        Factor.unconditionedVariables
        Factor.conditionedVariables
        Factor.variableDomainsDict
        """
        # supplied operation tracking -- do not remove
        if not (callTrackingList is None):
            callTrackingList.append(('eliminate', eliminationVariable))

        # typecheck portion
        if eliminationVariable not in factor.unconditionedVariables():
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Elimination variable is not an unconditioned variable " \
                            + "in this factor\n" + 
                            "eliminationVariable: " + str(eliminationVariable) + \
                            "\nunconditionedVariables:" + str(factor.unconditionedVariables()))
        
        if len(factor.unconditionedVariables()) == 1:
            print("Factor failed eliminate typecheck: ", factor)
            raise ValueError("Factor has only one unconditioned variable, so you " \
                    + "can't eliminate \nthat variable.\n" + \
                    "eliminationVariable:" + str(eliminationVariable) + "\n" +\
                    "unconditionedVariables: " + str(factor.unconditionedVariables()))

        "*** YOUR CODE HERE ***"
        #raiseNotDefined()
        factors = [factor]
        unconditioned_set = set()
        conditioned_set = set()
        # going through each factor in the list 
        for factor in factors: 
            unconditioned_variables = factor.unconditionedVariables() # getting the unconditioned variables of the current factor
            conditioned_variables = factor.conditionedVariables() # getting the conditioned variables of the current factor
            unconditioned_set.update(unconditioned_variables) # adding the unconditioned variables to the set
            conditioned_set.update(conditioned_variables) # adding the conditioned variables to the set
        unconditioned_set = unconditioned_set - {eliminationVariable} # making sure to remove the variable that is being eliminated
        variable_domains_dict = factors[0].variableDomainsDict() # getting the variable domains from the first factor
        elimination_values = variable_domains_dict[eliminationVariable] # getting all the possible values of the elimination variable
        new_factor = Factor(unconditioned_set, conditioned_set, variable_domains_dict) # creating a new factor with the combined variables
        all_assignments = new_factor.getAllPossibleAssignmentDicts() # getting all possible assignments for the new factor
           
        # going through each possible assignment
        for assignment in all_assignments: 
            probability = 0 # starting the probability at 0 to begin adding
        
            for value in elimination_values: # going through each factor
                full_assignment = dict(assignment)  # copying the assignment so the original one is not changed
                full_assignment[eliminationVariable] = value # adding the elimination variable's value to the assignment
                probability += factor.getProbability(full_assignment) # adding the probability for the current full assignment
            new_factor.setProbability(assignment, probability) # setting the calculated probability in the new factor
        return new_factor
        "*** END YOUR CODE HERE ***"

    return eliminate

eliminate = eliminateWithCallTracking()

