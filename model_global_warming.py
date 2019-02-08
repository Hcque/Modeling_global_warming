# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: 
# Collaborators (discussion):
# Time:

import pylab, numpy
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]

def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]

"""
End helper code
"""

def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    ans = []
    for d in degs:
        model = pylab.polyfit(x, y, d)
        ans.append(pylab.array(model))
    return ans
def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    fenzi = ((y - estimated)**2).sum()
    fenmu = ((y - numpy.mean(y))**2).sum()
    return 1 - fenzi/fenmu
def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for m in models:
        degree = len(m) - 1
        estY = pylab.polyval(m, x)
        r2 = r_squared(y, estY)
        pylab.figure()
        pylab.plot(x, y, 'bo')
        pylab.plot(x, estY, 'r')
        pylab.xlabel('years')
        pylab.ylabel('degree C')
        if degree == 1:
            se = se_over_slope(x, y, estY, m)
            pylab.title('For model of dreg: '+str(degree)+'\n'
                        +'R2 :'+str(r2)+'\n'+'Se :'+str(se))
        else:
            pylab.title('for model of dreg: '+str(degree)+'\n'
                        +'R2 :'+str(r2))
        
    
def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    ans = []
    for y in years:
        # calculate year y average
        temp = []
        for city in multi_cities:
            temp.append(numpy.mean(climate.get_yearly_temp(city, y)))
        ans.append(numpy.mean(temp))
    return pylab.array(ans)
    
def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    ans = []
    for i in range(window_length):
        the_sum = 0
        for j in range(i+1):
            the_sum += y[j]
        ans.append(the_sum/(i+1))
    for i in range(1, len(y)-(window_length-1)):
        the_sum = 0
        for j in range(i, i+window_length):
            the_sum += y[j]
        ans.append(the_sum/window_length)
    return pylab.array(ans)

def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    the_sum = numpy.sum((y-estimated)**2)
    return numpy.sqrt(the_sum/len(y))
    
def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    ans = []
    for y in years:
        # calculate year y average
        temp = []
        for city in multi_cities:
            every_temp = climate.get_yearly_temp(city, y)
            temp.append(every_temp)
        for_std = []
        for i in range(len(temp[0])):
            col = []
            for j in range(len(temp)):
                col.append(temp[j][i])
            for_std.append(numpy.mean(col))
        ans.append(numpy.std(for_std))
    return pylab.array(ans)
    
def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for m in models:
        degree = len(m) - 1
        estY = pylab.polyval(m, x)
        Rmse = rmse(y, estY)
        pylab.figure()
        pylab.plot(x, y, 'bo')
        pylab.plot(x, estY, 'r')
        pylab.xlabel('years')
        pylab.ylabel('degree C')
        pylab.title('for model of dreg: '+str(degree)+'\n'
                    +'rmse :'+str(Rmse))
            
if __name__ == '__main__':

    # Part A.4
    cli_data = Climate('data.csv')
    ansY = []
    for i in TRAINING_INTERVAL:
        ansY.append(cli_data.get_daily_temp('NEW YORK', 1, 10, i))
    x = pylab.array(TRAINING_INTERVAL)
    y = pylab.array(ansY)
    degs = [1]
    models = generate_models(x, y, degs)
    evaluate_models_on_training(x, y, models)
    
    ansY = []
    for j in TRAINING_INTERVAL:
        avg_year = numpy.mean(cli_data.get_yearly_temp('NEW YORK', j))
        ansY.append(avg_year)
    x = pylab.array(TRAINING_INTERVAL)
    y = pylab.array(ansY)
    degs = [1]
    models = generate_models(x, y, degs)
    evaluate_models_on_training(x, y, models)
    
    # Part B
    years = list(TRAINING_INTERVAL)
    multi_cities = CITIES
    y_val = gen_cities_avg(cli_data, multi_cities, years)
    x = pylab.array(TRAINING_INTERVAL)
    y = pylab.array(y_val)
    degs = [1]
    models = generate_models(x, y, degs)
    evaluate_models_on_training(x, y, models)
    
    # Part C
    years = list(TRAINING_INTERVAL)
    multi_cities = CITIES
    national_avg = gen_cities_avg(cli_data, multi_cities, years)
    x = pylab.array(TRAINING_INTERVAL)
    y_val = moving_average(national_avg, 5)
    y = pylab.array(y_val)
    degs = [1]
    models = generate_models(x, y, degs)
    evaluate_models_on_training(x, y, models)
    
    # Part D.2
    # 1. training data
    years = list(TRAINING_INTERVAL)
    multi_cities = CITIES
    national_avg = gen_cities_avg(cli_data, multi_cities, years)
    x = pylab.array(TRAINING_INTERVAL)
    y_val = moving_average(national_avg, 5)
    y = pylab.array(y_val)
    degs = [1,2,20]
    models_test = generate_models(x, y, degs)
    evaluate_models_on_training(x, y, models_test)
    # 2. test data
    xTest = pylab.array(TESTING_INTERVAL)
    national_avg = gen_cities_avg(cli_data, CITIES, list(TESTING_INTERVAL))
    y_val = moving_average(national_avg, 5)
    yTest = pylab.array(y_val)
    evaluate_models_on_testing(xTest, yTest, models_test)

    # Part E
    x = pylab.array(TRAINING_INTERVAL)
    y_val = gen_std_devs(cli_data, CITIES, list(TRAINING_INTERVAL))
    y_val = moving_average(y_val, 5)
    y_val = pylab.array(y_val)
    degs = [1]
    model_extreme = generate_models(x, y_val, degs)
    evaluate_models_on_training(x, y_val, model_extreme)