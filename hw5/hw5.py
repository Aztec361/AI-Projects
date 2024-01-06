import numpy as np
import matplotlib.pyplot as plt
import csv
import sys

if __name__=="__main__":
    filename = sys.argv[1]


    features = []
    targets = []


    with open(filename, 'r') as file:
        csv_reader = csv.reader(file, delimiter=',')


        next(csv_reader)
        for row in csv_reader:

            year = int(row[0])
            days = int(row[1])


            feature_vector = np.array([1, year])


            features.append(feature_vector)


            target = float(days)


            targets.append(target)


    X = np.array(features)
    Y = np.array(targets)

    Z = np.dot(X.T,X)
    I = np.linalg.inv(Z)
    PI = np.dot(I,X.T)
    hat_beta = np.dot(PI,Y)


    x_test = np.array([[1, 2022]])
    y_test = x_test @ hat_beta
    sign = '>' if hat_beta[1] > 0 else '<' if hat_beta[1] < 0 else '='

    x_star = - hat_beta[0]/hat_beta[1]
    answer = "The answer for x_star is not compelling because there are fluctuations in the dataset and the data is not showing any linear trends that are consistent. In order to truly determine the correctness of x_star, we need a more comprehensive testing strategy and dataset."



    year = X[:, 1]
    ice_days = Y


    plt.plot(year, ice_days)

    plt.xlabel('Year')
    plt.ylabel('Number of frozen days')



    plt.savefig('plot.jpg')
    print("Q3a:")
    print(X)
    print("Q3b:")
    print(Y.astype(np.int64))
    print("Q3c:")
    print(Z)
    print("Q3d:")
    print(I)
    print("Q3e:")
    print(PI)
    print("Q3f:")
    print(hat_beta)
    print("Q4: " + str(y_test[0]))
    print("Q5a: " + sign)
    print("Q5b: The sign of ^β1 tells us whether there is a positive or negative linear relationship between the year and the number of ice days on Lake Mendota. A positive sign indicates that the relationship is positive, meaning that as the year increases, the number of ice days also increases. A negative sign indicates the opposite relationship, while a sign of '=' indicates that there is no linear relationship between the two variables.")
    print(x_star)
    print(answer)

