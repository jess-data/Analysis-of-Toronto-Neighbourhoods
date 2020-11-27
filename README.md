# Analysis of Toronto Neighbourhoods using Machine Learning
This is a repository for IBM Data Science Capstone Project. All analysis was done by Jessica Uwoghiren.  <br />
The datasets and Jupyter notebook used in the project is also included in this repository.  <br />
The full report is also included in this repository.
See link to **blog post** for a more concise report - 

## Introduction
In 2019, 35% of new Canadian immigrants chose to settle in the City of Toronto. The City has 140 neighbourhoods, so, as a new immigrant, a vital question to answer is “What neighbourhood do I settle in?”. The aim of this project is to group Toronto neighborhoods in order of desirability using Machine Learning and Data Visualization techniques. I performed my analysis using on the following criteria:
• Total number of Essential Venues in each neighbourhood
• Primary and Secondary Benchmarks: Primary benchmarks considered were Unemployment rate, Crime rate and COVID-19 rates while the Secondary benchmark was housing price for a one-bedroom apartment in each neighbourhood.

## Methodology
### Python Libraries
The libraries used on this project include:
* Pandas – For storing and manipulating structured data. Pandas functionality is built on NumPy
* Numpy – For multi-dimensional array and matrix data structures. 
* Geopandas – For storing spatial data coordinates and shape files
*	Scikit learn – For Machine learning tasks
* Plotly Visualization Package – For all visualizations (including maps and graphs)
*	Requests - to send HTTP requests easily
*	Geopy – To retrieve location coordinates

### Project Flowchart
The main steps for this project can be summarized in the flow chart below:

![alt text](https://github.com/jess-data/Coursera_Capstone/blob/main/Screenshot%202020-11-24%20014704.png)

## Final Results
k-means clustering was used to group the neighbourhoods to obtain a final Neighbourhood Desirability index Choropleth Map
![alt text](https://github.com/jess-data/Coursera_Capstone/blob/main/Toronto%20Neighbourhood%20Desirability%20Map.png)
