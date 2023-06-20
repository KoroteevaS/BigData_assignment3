from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import ChiSqSelector, StringIndexer, VectorAssembler
import numpy as np
import itertools
import pandas as pd
import time



spark = SparkSession.builder.getOrCreate()
all_data=[]
def get_first_n_elements(dictionary, n):
    return dict(itertools.islice(dictionary.items(), n))

def rank_features(df, num_features, method='chi-square'):

    pandas_df= df.toPandas()
    corr_matrix = pd.DataFrame(pandas_df[feature_columns].corr())
    feature_scores = {}

    if method == 'chi-square':
        selector = ChiSqSelector(numTopFeatures=num_features, featuresCol="features",
                                 outputCol="selected_features", labelCol="indexed_lab")
        selector_model = selector.fit(df)
        feature_indices = selector_model.selectedFeatures

        # Get the names of the selected features
        selected_features = [feature_columns[i] for i in feature_indices]
        return selected_features


    elif method == 'pearson':
        for feature in feature_columns:
            score = corr_matrix.loc[feature, 'indexed_lab']
            feature_scores[score]=feature

        sorted_d = dict(sorted(feature_scores.items(),reverse=True))
        feature_dict = get_first_n_elements(sorted_d, num_features)

        selected_feature_names = [feature_dict[el] for el in feature_dict.keys() ]

        return selected_feature_names


list_of_labels = [
    "CONDUCTONLY",
    "ANTISOCDX2",
    "AVOIDPDX2",
    "DEPPDDX2",
    "OBCOMDX2",
    "PARADX2",
    "SCHIZDX2",
    "HISTDX2",
    "DEP",
    "PAN",
    "AGORA",
    "SOCPHOB",
    "SPECPHOB",
    "ANX"
]

for class_name in list_of_labels:
    df = spark.read.csv("nesarc_final.csv", header=True, inferSchema=True)
    label_indexer = StringIndexer(inputCol=class_name, outputCol="indexed_lab", handleInvalid="keep")
    df = label_indexer.fit(df).transform(df)
    feature_columns = [col for col in df.columns if col != class_name]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    df = assembler.transform(df)
    selected_features = rank_features(df, 10, method='pearson')
    my_data = [class_name] + selected_features
    all_data.append(my_data)
    print(my_data)

all_data_df = pd.DataFrame(all_data, columns=["Class"] + selected_features)
print(all_data_df)
spark_df = spark.createDataFrame(all_data_df)
spark_df.coalesce(1).write.mode("overwrite").csv("/user/korotesvet/features_csv", header=True)
    
spark.stop()   
