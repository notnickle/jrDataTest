from scipy.optimize import minimize
import numpy as np


#Declaring some global variables for future use
ogSales = ogPrice = changeInSales = None

def func(changeInPrice):
    #Rounding guess to 2 decimal places
    roundedChange = np.round(changeInPrice[0],2)

    #Calculate New Price (Rounded to 2 decimal places)
    newPrice = np.round((ogPrice + roundedChange),2)

    #Calculate New Sales (Rounded down)
    newSales = np.floor(ogSales * (1 - changeInSales * roundedChange))
    return newPrice * newSales

# Define the negative revenue function for minimization
def neg_func(changeInPrice):
    return -func(changeInPrice)


#Define the gradient of the negative revenue function
def neg_revenue_grad(changeInPrice):
    #Rounding guess to 2 decimal places
    roundedChange = np.round(changeInPrice[0],2)

    # Calculate the new price
    newPrice = np.round((ogPrice + roundedChange),2)

    #Calculate the gradient of the negative revenue function
    grad = ogSales * (1 - changeInSales * roundedChange) + newPrice * ogSales * changeInSales
    return np.array([-grad])

#/////////////////////////////////
#Main Driver
#/////////////////////////////////
def main():
    global ogSales
    global ogPrice
    global changeInSales

    #Prompts for original parameters
    ogPrice = float(input('Enter Original Plan Price: \n'))
    ogSales = float(input('Enter Original Sales Count: \n'))
    changeInSales = float(input('Enter %% change in sales per $ change from original price: \n'))
    
    #Initial guesses for the price, ranging from -10 to 10, rounded to 2 decimal places
    initial_guesses = np.round(np.linspace(-10, 10, num=10), 2)

    #Initializing best outcome variables
    bestMaxRevenue = -np.inf
    bestChangeInPrice = None


    for guess in initial_guesses:
        changesInPrice = np.array([guess])

        #Use the minimize function from SciPy with the 'L-BFGS-B' method
        result = minimize(neg_func, changesInPrice, method='L-BFGS-B',jac=neg_revenue_grad, bounds=[(-ogPrice, ogPrice)])

        #Maximum revenue is the negative of the minimized value
        newMaxRevenue = -result.fun
        maxChangeInPrice = result.x[0]
        
        #Keep the best 'MaxRevenue'
        if newMaxRevenue > bestMaxRevenue:
            bestMaxRevenue = newMaxRevenue
            bestChangeInPrice = maxChangeInPrice

    print(f'\nMaximum Possible Revenue: {bestMaxRevenue:.2f}')
    print(f'Price change needed for said Revenue: {bestChangeInPrice:.2f}\n')


#Main method call
if __name__ == "__main__":
    main()