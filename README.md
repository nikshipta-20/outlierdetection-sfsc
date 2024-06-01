# outlierdetection-sfsc
The repository contains the source code for various algorithms, including Spectral Clustering, Spectral Clustering with Group Fairness Constraints and the scalable version of it. The repository also consists the implementation of the outlier detection using spectral clustering and scalable FairSC. This project also compares the results. 


## The following plots represent the results of the 3 algorithms.
Note: The k value is taken as 3 for simplification

# 1. Spectral Clustering (SC)
![sc k = 3](https://github.com/nikshipta-20/outlierdetection-sfsc/assets/122418060/a555dec5-ee60-4515-8a2b-2fe1a02bf738)

# 2. Fair Spectral Clustering (FairSC)
![fairsc k = 3](https://github.com/nikshipta-20/outlierdetection-sfsc/assets/122418060/e42244b5-2c4a-4857-9ae4-94b4738b94c3)

# 3. Scalable Fair Spectral Clustering (s-FairSC)
![sfsc k = 3](https://github.com/nikshipta-20/outlierdetection-sfsc/assets/122418060/0bd9d1fd-11c6-47dc-bbc2-2ef36b883d8e)


## The balance factor is considered to determine the amount of fairness present in the clustering. If we observe, FairSC and s-FairSC had higher balance when compared to SC

# 1. Balance vs SC
![bal vs k sc](https://github.com/nikshipta-20/outlierdetection-sfsc/assets/122418060/1c5a120d-4045-4a04-b258-d3fd49d3e855)

# 2. Balance vs FairSC
![bal vs k fairsc](https://github.com/nikshipta-20/outlierdetection-sfsc/assets/122418060/0402db86-c6c1-4c3e-b1ab-363519a089ba)


The following is the algorithm for spectral basedoutlier detection.


## The following plots are the results of the outlier detection using SC algorithm

# 1. Plot of the second smallest eigenvalues
![sc 2nd smallest](https://github.com/nikshipta-20/outlierdetection-sfsc/assets/122418060/361a6e5a-2e37-41f4-ae86-c7770e663e2d)

# 2. Plot of detected outliers
![sc outliers](https://github.com/nikshipta-20/outlierdetection-sfsc/assets/122418060/19a68ce3-65d9-436b-9dfa-562ac9e89e54)


## The following plots are the results of the outlier detection using s-FairSC algorithm

# 1. Plot of the second smallest eigenvalues
![sfsc 2nd smallest](https://github.com/nikshipta-20/outlierdetection-sfsc/assets/122418060/e88675dc-505b-47b5-a172-99da9be73972)

# 2. Plot of detected outliers
![sfsc outliers](https://github.com/nikshipta-20/outlierdetection-sfsc/assets/122418060/4887baab-215b-47a0-9564-d380054d6216)
