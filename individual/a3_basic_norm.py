from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, NaiveBayes
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import time


spark = SparkSession.builder.getOrCreate()

data_path = "nesarc_final_lack.csv"
df_original = spark.read.csv(data_path, header=True, inferSchema=True)

def avg(lst):
    return sum(lst) / len(lst)

def train_and_evaluate_model(classifier, df, seed, class_name):
    
    (train_data, test_data) = df.randomSplit([0.7, 0.3], seed=seed)
    my_string = classifier + "(labelCol='" + class_name + "')"
    model = eval(my_string)
    pipeline = Pipeline(stages=[model])
    start_time = time.time()
    model = pipeline.fit(train_data)
    end_time = time.time()
    train_predictions = model.transform(train_data)
    train_accuracy = train_predictions.filter(train_predictions[class_name] == train_predictions["prediction"]).count() / float(train_predictions.count())
    test_predictions = model.transform(test_data)
    test_accuracy = test_predictions.filter(test_predictions[class_name] == test_predictions["prediction"]).count() / float(test_predictions.count())
    running_time = end_time - start_time
    return train_accuracy, test_accuracy, running_time


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
    "LogisticRegression",
    "RandomForestClassifier",
    "NaiveBayes"
]

seeds = [987, 1052, 777]

results = []

for class_ind, class_name in enumerate(list_of_labels):
    print(class_ind + 1, class_name)

    for classifier in classifiers:
        print(class_ind + 1, classifier)

        seeds_results_training = []
        seeds_results_test = []
        running_times = []

        for seed in seeds:
            print("Seed:", seed)
            df = df_original.alias("df_copy")
            feature_columns = list(df.columns)
            feature_columns.remove(class_name)
            assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
            df = assembler.transform(df)
            scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
            scaler_model = scaler.fit(df)
            df = scaler_model.transform(df)
            train_accuracy, test_accuracy, running_time = train_and_evaluate_model(classifier, df, seed, class_name)

            seeds_results_training.append(train_accuracy)
            seeds_results_test.append(test_accuracy)
            running_times.append(running_time)
        avg_train_accuracy = avg(seeds_results_training)
        avg_test_accuracy = avg(seeds_results_test)
        avg_running_time = avg(running_times)
        results.append((classifier, class_name, avg_train_accuracy, avg_test_accuracy, avg_running_time))

results_df = spark.createDataFrame(results, ["Classifier", "Class", "Training Accuracy", "Test Accuracy", "Running Time"])
results_df.coalesce(1).write.mode("overwrite").csv("/user/korotesvet/results_norm_csv", header=True)
spark.stop()
