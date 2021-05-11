# Statistical-and-Clustering-Analysis-of-Galaxies
A deep dive into the Galaxy's Science using Statistic and Machine Learning.  

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

1-IDEA OF THE PROJECT

The idea of this project was born from my deep love about Astronomy and all the scientific and mathetical aspects around it.
Using those universal tools i explored, unsterstood and applied all of my statistical, astronomic and ML's knowledge to solve 
some interesting questions: 

    A-How does the main features of these galaxies are related to each other? 

    B-Can we group those galaxies acording to their characteristics? 

    C-How do we scientically analyse those groups? 

Following these three main questions i used in order:

    A-Data Visualization, pearson correlation and R square

    B-Cluster Analysis

    C-Hypothesis Testing (Two sample T-test, Mann-Whitney, Shapiro-Wilk) 

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

2-ABOUT THE DATASET

This Dataset is part of a incredible reaserch made in the 90's years, with a sky survey, generating this tabular dataset named: COMBO-17, avainable on Kaggle, here:

https://www.kaggle.com/mrisdal/combo17-galaxy-dataset

As a lot of Kaggle Datasets, this data is pretty clear, so there isn't a lot of data cleaning in this notebook(
if you are searching for data cleaning take a look on my AirBnbLowCost repository). 

Here you are going to found a lot of astronomical features by galaxy, like Magnitude in Red, RedShift of the galaxy and
photon flux density by wave length. Here i pasted the guide provided by the kaggle's page for the features:

Col 1: Nr, object number

Col 2-3: Total R (red band) magnitude and its error. This was the band at which the basic catalog was constructed. Magnitudes are inverted logarithmic measures of brightness. A galaxy with R=21 is 100-times brighter than one with R=26. The error is the standard deviation derived from detailed knowledge of the measurement process. This dataset is an excellent example of astronomical datasets where each variable is accompanied by heteroscedastic measurement errors of known variances.

Col 4-5: ApDRmag is the difference between the total and aperture magnitude in the R band. This is a rough measure of the size of the galaxy in the image where ApDRmag=0 corresponds to a point source. Negative values are not physically meaningful. mumax is the central surface brightness of the object in the R band. The difference between Rmag and mumax should also be an indicator of galaxy size.

Col 6-9: Mcz and MCzml are two redshift estimates. Mcz is the preferred value. e.Mcz is its estimated error, and chi2red is the reduced chi-squared value of the least-squares fit of the 17-band magnitudes to the best-fit template galaxy spectrum. Galaxies with large e.Mcz or chi2red might be omitted as unreliable.

Col 10-29: These give the absolute magnitudes (i.e. intrinsic luminosities) of the galaxy in 10 bands, with their measurement errors. They are based on the measured magnitudes and the redshifts, and represent the intrinsic luminosities of the galaxies; a galaxy with M=-15 is 100-times less luminous than one with M=-20. These magnitudes are not all independent of each others, but the are important for representing intrinsic properties of the galaxies. Below is one of several redshift-stratified plots of the B-band absolute magnitude (abscissa) against the difference of magnitude (i.e. ratio of luminosities) between the 2800A ultraviolet and blue band, which is a sensitive indicator of star formation. A redshift-dependent bimodal distribution is seen.

Col 30-55: Observed brightnesses in 13 bands in sequence from 420 nm in the ultraviolet to 915 nm in the far red. These are given in linear variables with units of photon flux densities, photons/m2/s/nm. Again, each measurement is accompanied by a measurement error which can be used to distinguish measurement from intrinsic dispersions in the distributions.

Col 56-65: Observed brightnesses in 5 traditional broad spectral bands, UBVRI. These are largely redundant with the 13 bands in the previous columns.

There is a copy of this dataset here in this repository, you can download it.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

3-ABOUT THE PROJECT

Acording to the previous questions i divided this project in three main blocks:

A-THE DATASET:

Here i focused in explore and understand the main features and have some insights
about the relationship and patterns in the data. And thinking about it i created subtopics:

a-Exploratory Data Analysis:

Here i manipulated and chose the features to work with.

b-Understanding the three main variables

In that point i selected the main features to deal with: Rmag, ApDRmag and Mcz.
This three features mean Magnitude in Red.

The difference between the total and the
aperture magnitude in Red band, and it is a good feature to understand the size of the galaxy.

And finally the Redshift of the galaxy (a Doppler effect that is a good feature to understand the 
distance of the galaxy from us), respectively.

In that point we gain some insights like the following thoughts:

There is a clear positive relationship between the Size and the Magnitude in red in the galaxies. (That happens because the more the galaxies merge with each other, and by consequence it grows in size, more gas and dust are lost, wich are the main compounds to form new, bright and blue stars. When the galaxy loses the gas and dust it stops to generate new stars, and the galaxy turns old and red, or yellow, and due to this, less bright)

![image](https://user-images.githubusercontent.com/80376071/117868625-33331880-b270-11eb-930d-bf95daa00e76.png)

The smaller the magnitude of the galaxy more photon density flux it has:

![image](https://user-images.githubusercontent.com/80376071/117869592-5f9b6480-b271-11eb-8148-52f58448b9c5.png)


And finally, my favorite part: The pattern of the Mvz (distance) founded in the Data. Here we can clearly see the vertical bars of points along the Mcz axis (x-axis) in relation to the UB-color_index. Those vertical structure are the clusters of galaxies founded in the big structure of the Universe. For more information you can read the notebook, there is more astronomical background there.

![image](https://user-images.githubusercontent.com/80376071/117870151-1dbeee00-b272-11eb-81f6-06bca2dff133.png)


B-CLUSTER ANALYSIS:

Our main objective here was discover if there was any natural cluster of galaxyes that could be founded on this Dataset. Bases on this i did some steps to reach this objective:

a-Dealing with outliers: here i made some considerations about the nature of this dataset and its relation with the art of cut outliers, who is essential for Cluster Algorithms, and why a chose to made small cuts. To be able to do that i used blockspots and histograms

Blockspot of ApDRmag before the cut:

![image](https://user-images.githubusercontent.com/80376071/117870945-2ebc2f00-b273-11eb-8327-ce28b223a32f.png)

Histogram of Rmag:

![image](https://user-images.githubusercontent.com/80376071/117871042-527f7500-b273-11eb-844f-351ef1978f8a.png)


b-The Algorithms: at this point i chose the three main cluster algorithms to work with. That was one of the most challenging steps of this project, where there is a lot of hyperparameter tuning, choses about scaling or not the dataset, and the types of scaling to be made, metrics to be used to evaluate the clusters qualities (I mainly used the Silhouette Score, that you can see one plot above). Inside the notebook there is lot of technical discutions and understanding about the data. The three main algorithms used were:

- Agglomerative Hierarchical Clustering:

Dendogram produced by an agglomerative cluster with silhouette score of 0.61. Inside of the notebook there is a explanation of why this algorithm and dendogram are not good results.

![image](https://user-images.githubusercontent.com/80376071/117872088-a2ab0700-b274-11eb-962d-276792af2b3c.png)

Plot of the Silhouette Score:

![image](https://user-images.githubusercontent.com/80376071/117872639-4c8a9380-b275-11eb-8426-861d37ba9dbd.png)


- K-MEANS cluster:

Elbow method used to avaluate k-means cluster with silhouette score of 0.49 :

![image](https://user-images.githubusercontent.com/80376071/117872425-06cdcb00-b275-11eb-8bdf-927442f1c175.png)

Silhouette Score used to evaluate k-means cluster:

![image](https://user-images.githubusercontent.com/80376071/117872675-590eec00-b275-11eb-84cb-2548eacaa959.png)


- DBSCAN Cluster:

Plot avaluating the DBSCAN algorithm with our best silhouette score, around 0.66:

![image](https://user-images.githubusercontent.com/80376071/117872971-a9864980-b275-11eb-92d7-3b4edff9a312.png)

And then, based on two main parameters raised by some limitations of the dataset i selected the DBSCAN algorithm to create our final cluster, that resulted in some tables and 3D plots like these:

Table with mean of the features for DBSCAN cluster

![image](https://user-images.githubusercontent.com/80376071/117873359-113c9480-b276-11eb-9ca3-73c9455cc56a.png)

Distribution of the cluster in a 3D plot:

![image](https://user-images.githubusercontent.com/80376071/117873582-552f9980-b276-11eb-9edf-591349079f1e.png)


3-STATISTICAL ANALYSIS

In this section of the project i proposed an Statistical Analysis of the clusters made before. And then two main points come to my mind:

Are those differences founded in the cluster algorithm real ?

And with this i realized that question could be solved with some simple statistical tools, like hypothesis test. To be able to do that was necessary to understand the distribution of the data, with the quastion :

Does this data follow the Normal Distribution? To answer that question i used three techniques: the histogram, the qqplot and the Shapiro-Wilk test (discutions abou wich one of those is better on the notebook)

- Histogram:

![image](https://user-images.githubusercontent.com/80376071/117874968-04b93b80-b278-11eb-8647-251dd9b0ce05.png)


-QQplot:

![image](https://user-images.githubusercontent.com/80376071/117875050-1b5f9280-b278-11eb-8182-cd3633c970e0.png)

-Shapiro-Wilk test:

![image](https://user-images.githubusercontent.com/80376071/117875206-4518b980-b278-11eb-8ad6-d7b088c1c1d4.png)

Two more questions were made after that, and i needed to use NON-PARAMETRIC tests, like Mann-Whitney, and a simple Z-Score analysis as a interesting Bonus in the end, where there was some cool techniques to know where our galaxy falls in the distribution of galaxy Brightness.

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

4- FINAL THOUGHTS:

This project is a good estrategy to use some machine learning tools and statistical analysis to understand our universe and find hidden patterns in the data. That was a fun and challenging experience where the mathematical, statistical and the astronomical knowledge made me push my self harder in a lot of situations, and i think thats great path if you want to do the same. If you want to share any thoughts with me you can send my an email: breno_valle@outlook.com. Thank you for you attention and good luck discoring the universe with Data!!!

