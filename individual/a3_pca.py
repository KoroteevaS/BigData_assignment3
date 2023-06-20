from pyspark.ml.feature import VectorAssembler, PCA
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import SparkSession
import matplotlib.pyplot as plt
import time
import pandas as pd

spark = SparkSession.builder.getOrCreate()

def PCA_model(data, seed, class_name, num):
    # Split the data into training and test sets (70% for training, 30% for testing)
    (training_data, test_data) = data.randomSplit([0.7, 0.3], seed=seed)
    
    feature_columns = list(data.columns)
    feature_columns.remove(class_name)
    
    vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    train_data = vector_assembler.transform(training_data)
    
    pca = PCA(k=num, inputCol="features", outputCol="pca_features")
    lr = LogisticRegression(labelCol=class_name, featuresCol="pca_features", predictionCol="prediction")
    
    pipeline = Pipeline(stages=[pca, lr])  
    start_time = time.time()
    trained_model = pipeline.fit(train_data)
    end_time = time.time()
    train_predictions = trained_model.transform(train_data)
    train_accuracy = train_predictions.filter(train_predictions[class_name] == train_predictions["prediction"]).count() / float(train_predictions.count())

    test_data = vector_assembler.transform(test_data)
    test_data = trained_model.transform(test_data)

    evaluator = BinaryClassificationEvaluator(labelCol=class_name, metricName="areaUnderROC")
    accuracy = evaluator.evaluate(test_data)
    
    running_time = end_time - start_time
    
    return trained_model, train_accuracy,accuracy, running_time

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
classifiers = [
    "RandomForestClassifier",
]
classifiers = ["PCA_model"]

seeds = [987, 1052, 777]

results = []
train_accuracies = []
test_accuracies = []
running_times = []

for class_name in list_of_labels:
    for classifier in classifiers:
        for seed in seeds:
            for num in [20,30]:
                df = df_original.alias("df_copy")

                trained_model,train_accuracy, accuracy, running_time = PCA_model(df, seed, class_name, num)
                train_accuracies.append((classifier, class_name, seed,num, train_accuracy))
                test_accuracies.append((classifier, class_name,seed, num, accuracy))
                running_times.append((classifier, class_name,seed, num, running_time))
                # Append the results to the list
                results.append((classifier, class_name, seed, num, train_accuracy ,accuracy, running_time))

pandas_df = pd.DataFrame(results, columns= ["Classifier", "Class", "Seed", "Num PCA Comp", "Train Accuracy","Test Accuracy", "RunTime"])
pandas_df.to_csv('data2.csv', index=False)

results_df = spark.createDataFrame(results, ["Classifier", "Class", "Seed", "Num PCA Comp", "Train Accuracy","Test Accuracy", "RunTime"])

for result in train_accuracies:
    classifier, class_name,seed, num, accuracy = result
    plt.plot(num, accuracy, marker='o', label=f"{class_name}-{classifier}-{num}")
plt.xlabel("Number of PCA Components")
plt.ylabel("Training Accuracy")
plt.title("Training Accuracy vs Number of PCA Components")
plt.legend()
plt.show()

for result in test_accuracies:
    classifier, class_name,seed, num, accuracy = result
    plt.plot(num, accuracy, marker='o', label=f"{class_name}-{classifier}-{num}")
plt.xlabel("Number of PCA Components")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs Number of PCA Components")
plt.legend()
plt.show()

for result in running_times:
    classifier, class_name,seed, num, running_time = result
    plt.plot(num, running_time, marker='o', label=f"{class_name}-{classifier}-{num}")
plt.xlabel("Number of PCA Components")
plt.ylabel("Running Time (seconds)")
plt.title("Running Time vs Number of PCA Components")
plt.legend()
plt.show()


results_df.show()

# Save the DataFrame as a CSV file
results_df.coalesce(1).write.mode("overwrite").csv("/user/korotesvet/results_pca", header=True)
spark.stop() 
