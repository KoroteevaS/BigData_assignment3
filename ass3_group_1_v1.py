#!/usr/bin/env python
# coding: utf-8

# In[41]:


from pyspark.sql.functions import split, when
print("here")
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import split
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
import pandas as pd
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("DecisionTree").getOrCreate()
print("here1")
kdd = spark.read.csv("data/kdd.data")
print(kdd)
print(kdd.show())



# In[85]:


first_value = dataset.select(dataset.columns[0]).first()[0]
print(first_value)



# In[44]:


feature_columns = kdd.columns[:-1]  # Select all columns except the last one
label_column = kdd.columns[-1] 


# Create a StringIndexer to encode the label column
label_indexer = StringIndexer(inputCol=label_column, outputCol="indexedLabel")
data = label_indexer.fit(kdd).transform(kdd)


# # List of columns to convert
columns_to_convert = ["_c0","_c1", "_c2", "_c3", "_c4", "_c5", "_c6", "_c7", "_c8", "_c9", "_c10",
                      "_c11", "_c12", "_c13", "_c14", "_c15", "_c16", "_c17", "_c18", "_c19", "_c20",
                      "_c21", "_c22", "_c23", "_c24", "_c25", "_c26", "_c27", "_c28", "_c29", "_c30",
                      "_c31", "_c32", "_c33", "_c34", "_c35", "_c36", "_c37", "_c38", "_c39", "_c40"]

# Convert columns to numerical types
for column in columns_to_convert:
    data = data.withColumn(column, col(column).cast("double"))
# Create a vector assembler to combine the feature columns into a single vector column
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
kdd_vec = assembler.transform(data)
kdd_vec.select("features").show(truncate=False)
(trainingData, testData) = kdd_vec.randomSplit([0.7, 0.3])
trainingData.show()
testData.show()


# In[46]:


dt = DecisionTreeClassifier(labelCol="indexedLabel", featuresCol="features")
dtModel = dt.fit(trainingData)
# # Train the Decision Tree model
# dt = DecisionTreeClassifier(labelCol="labelIndex", featuresCol="features_vector")
# dtModel = dt.fit(trainingData)

# Make predictions on the test set
predictions = dtModel.transform(testData)



# Evaluate the model
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy"
)
accuracy = evaluator.evaluate(predictions)

# Print the accuracy
print("Decision Tree Accuracy:", accuracy)

# Clean up resources
spark.stop()


# In[ ]:




