Files:

Scripts:

a3_correlation.py
a3_basic.py
a3_basic_norm.py
a3_pca.py
a3_lasso.py

Data:

nesarc_final.csv - full set
nesarc_final_lack.csv - set with covariant feature removed

Output folders:

features_csv
results_csv
results_norm_csv
results_pca_csv

Installation for cluster:

1. Prepare setup file (setup.csh) with your environmental variables:
	export HADOOP_VERSION=2.8.0
	export HADOOP_PREFIX=/local/Hadoop/hadoop-$HADOOP_VERSION
	PATH=${PATH}:$HADOOP_PREFIX/bin
	export SPARK_HOME=/home/username/spark-3.2.3-bin-hadoop2.7
	PATH=$SPARK_HOME/bin:$PATH
	export HADOOP_CONF_DIR=$HADOOP_PREFIX/etc/hadoop
	export YARN_CONF_DIR=$HADOOP_PREFIX/etc/hadoop
	need java8

2. Install pyspark: python -m pip install pyspark==3.2.3 (your package version)
3. Run command: source setup.csh

Locally:

Similarly, but spark package will be enough for basic troubleshooting.
HADOOP_HOME variables set the same as SPARK_HOME.
Run script with Anaconda,VS or other IDE.

Preparation:

1. Put datasets to hadoop cluster space: 
 hdfs dfs -put nesarc_final.csv /user/username/
 hdfs dfs -put nesarc_final_lack.csv /user/username/
2. For logging update path to csv file with your username.

Run and check:

1. Run python file: "spark-submit –master yarn -–deploy-mode cluster script.py" (can run first spark-submit to check)
2. Access log folders 
 hdfs dfs -ls /user/username/results_csv 
 and copy file started with part to you space  via :
 hdfs dfs -get /user/username/results_csv/part......scv /home/username

------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
Script description:

a3_correlation.py - feature analysis tool performing ranking and selection.

extracts the features higher correlated with labels.
It loops 14 label columns one by one and write down results to spark dataframe in features_csv folder on cluster.
By default it set as 10 features and "pearson" correlation.
But could be reset to a custom number of features.

a3_basic.py - run model based on different classifiers for different seeds and labels. Saves results in spark dataframe in results_csv folder on cluster.
a3_basic_norm.py -scaling added, saved results in spark dataframe in results_norm_csv folder.
a3_pca.py - PCA is used with LogisticRegression. Results are saved into results_pca on cluster
a3_lasso - Lasso implemented. Results are saved into results_lasso folder on cluster.
