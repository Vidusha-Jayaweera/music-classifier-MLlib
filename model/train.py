# import os
# import sys

# if 'SPARK_HOME' in os.environ:
#     del os.environ['SPARK_HOME']

# os.environ['PYSPARK_PYTHON'] = sys.executable
# os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

print("Booting local Spark Session...")
spark = SparkSession.builder \
    .appName("MusicClassifierTraining") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# =====================================================================
# PHASE 1: Train on Original 7-Class Mendeley Dataset
# =====================================================================
print("\n--- PHASE 1: Training on Original 7-Class Mendeley Dataset ---")
mendeley_data = spark.read.option("header", "true").option("quote", "\"").option("escape", "\"").csv("../Mendeley_dataset.csv")
mendeley_data = mendeley_data.dropna(subset=["lyrics", "genre"])

labelIndexer = StringIndexer(inputCol="genre", outputCol="label")
tokenizer = RegexTokenizer(inputCol="lyrics", outputCol="words", pattern="\\W+")
remover = StopWordsRemover(inputCol="words", outputCol="filteredWords")
hashingTF = HashingTF(inputCol="filteredWords", outputCol="rawFeatures", numFeatures=10000)
idf = IDF(inputCol="rawFeatures", outputCol="features")
nb = NaiveBayes(labelCol="label", featuresCol="features")

pipeline = Pipeline(stages=[labelIndexer, tokenizer, remover, hashingTF, idf, nb])

train_7, test_7 = mendeley_data.randomSplit([0.8, 0.2], seed=42)
model_7_classes = pipeline.fit(train_7)

predictions_7 = model_7_classes.transform(test_7)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
acc_7 = evaluator.evaluate(predictions_7)
print(f"Phase 1 (7 Classes) Test Accuracy: {acc_7:.4f}")


# =====================================================================
# PHASE 2: Re-Train on 8-Class Merged Dataset
# =====================================================================
print("\n--- PHASE 2: Re-training on 8-Class Merged Dataset ---")
merged_data = spark.read.option("header", "true").option("quote", "\"").option("escape", "\"").csv("../Merged_dataset.csv")
merged_data = merged_data.dropna(subset=["lyrics", "genre"])

train_8, test_8 = merged_data.randomSplit([0.8, 0.2], seed=42)
final_model = pipeline.fit(train_8)

predictions_8 = final_model.transform(test_8)
acc_8 = evaluator.evaluate(predictions_8)
print(f"Phase 2 (8 Classes) Test Accuracy: {acc_8:.4f}")

model_path = "trained_lyrics_model"
final_model.write().overwrite().save(model_path)
print(f"\nSUCCESS: Final 8-class model saved to '{model_path}' folder!")

spark.stop()