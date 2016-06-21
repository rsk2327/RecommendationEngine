# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 09:31:37 2016

@author: rsk
"""
from pyspark import SparkContext
from pyspark import SQLContext
sc = SparkContext("local","recommendationEngineApp")
sqlContext = SQLContext(sc)

from pyspark.sql import SQLContext,Row
#from pyspark.sql import Functions as F

dataDir = "/home/rsk/Documents/Spark"

userData = sc.textFile(dataDir+"/ml-100k/u.user").map(lambda x : x.split("|"))
movieData = sc.textFile(dataDir+"/ml-100k/u.item").map(lambda x : x.split("|"))
ratingData = sc.textFile(dataDir+"/ml-100k/u.data").map(lambda x : x.split("\t"))

#%%

ratingDataDF = ratingData.map(lambda x : Row(userID = int(x[0]),
                        movieID = int(x[1]),
                        rating=float(x[2]),
                        timestamp = int(x[3])))
ratingDataDF = sqlContext.createDataFrame(ratingDataDF)

userDataDF = userData.map(lambda x : Row(userID=int(x[0]),
                                        age = int(x[1]),
                                        gender = x[2],
                                        occupation = x[3],
                                        zipcode = x[4]))
userDataDF = sqlContext.createDataFrame(userDataDF)

movieDataDF = movieData.map(lambda x : Row(movieID = int(x[0]),
                                            movieTitle = x[1],
                                            releaseDate = x[2],
                                            videoReleaseDate = x[3],
                                            IMDBurl = x[4],
                                            unknown= int(x[5]),
                                            action = int(x[6]),
                                            adventure = int(x[7]),
                                            animation = int(x[8]),
                                            childrens = int(x[9]),
                                            comedy = int(x[10]),
                                             crime = int(x[11]),
                                             documentary = int(x[12]),
                                             drama = int(x[13]),
                                             fantasy = int(x[14]),
                                             filmNoir = int(x[15]),
                                             horror = int(x[16]),
                                             musical = int(x[17]),
                                             mystery = int(x[18]),
                                             romance = int(x[19]),
                                             sciFi = int(x[20]),
                                             thriller = int(x[21]),
                                             war = int(x[22]),
                                             western = int(x[23])))
movieDataDF = sqlContext.createDataFrame(movieDataDF)

#%%


userDataDF.show(5)
ratingDataDF.show(5)                                             