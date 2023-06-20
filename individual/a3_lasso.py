from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession
import time
import pandas as pd
import matplotlib.pyplot as plt

# Initialize SparkSession
spark = SparkSession.builder.getOrCreate()

def LassoModel(df, seed, class_name):
    (train_data, test_data) = df.randomSplit([0.7, 0.3], seed=seed)
    
    feature_columns = list(df.columns)
    feature_columns.remove(class_name)
    
    vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    train_data = vector_assembler.transform(train_data)
    test_data = vector_assembler.transform(test_data)
    
    lr = LogisticRegression(featuresCol="features", labelCol=class_name)
    
    param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.001, 0.01, 0.1, 1.0]).build()
    
    evaluator = BinaryClassificationEvaluator(labelCol=class_name, metricName="areaUnderROC")
    cross_validator = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)

    start_time = time.time()
    cv_model = cross_validator.fit(train_data)
    end_time = time.time()
    
    lr_model = cv_model.bestModel
    coefficients = lr_model.coefficients
    coefficients_list = coefficients.toArray().tolist()
    print(coefficients_list)
    feature_names = feature_columns
    print(feature_names)
    
    train_predictions = lr_model.transform(train_data)
    train_accuracy = evaluator.evaluate(train_predictions)
    test_predictions = lr_model.transform(test_data)
    test_accuracy = evaluator.evaluate(test_predictions)
    running_time = end_time - start_time
    return train_accuracy, test_accuracy, running_time


data_path = "nesarc_final_lack.csv"
df_original = spark.read.csv(data_path, header=True, inferSchema=True)
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
classifiers = ["LassoModel"]

seeds = [987, 1052, 777]

results = []
train_accuracies = []
test_accuracies = []
running_times = []
for class_name in list_of_labels:
    for classifier in classifiers:
        for seed in seeds:
            df = df_original.alias("df_copy")
            train_accuracy, test_accuracy, running_time = eval(classifier)(df, seed, class_name)
            results.append((classifier, class_name, train_accuracy, test_accuracy, running_time))
            train_accuracies.append((classifier, class_name,  train_accuracy))
            test_accuracies.append((classifier, class_name, test_accuracy))
            running_times.append((classifier, class_name, running_time))

pd_df = pd.DataFrame(results)
print(pd_df)

results_df = spark.createDataFrame(results, ["Classifier", "Class", "Training Accuracy", "Test Accuracy", "Running Time"])
results_df.coalesce(1).write.mode("overwrite").csv("/user/korotesvet/results_lasso", header=True)
spark.stop()


