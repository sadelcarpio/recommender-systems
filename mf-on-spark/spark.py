from pyspark.mllib.recommendation import ALS, Rating
from pyspark import SparkContext

SparkContext.setSystemProperty("spark.driver.memory", "16g")
SparkContext.setSystemProperty("spark.executor.memory", "16g")

sc = SparkContext("local", "MF On Spark")

data = sc.textFile("movielens-20m-dataset/rating.csv")
header = data.first()
data = data.filter(lambda line: line != header)

ratings = data.map(
    lambda l: l.split(",")
).map(
    lambda l: Rating(int(l[0]), int(l[1]), float(l[2]))
)

train, test = ratings.randomSplit([0.8, 0.2])

K = 10
epochs = 10
model = ALS.train(train, K, epochs)

x = train.map(lambda p: (p[0], p[1]))
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = train.map(lambda r: ((r[0], r[1]), r[2])).join(p)  # join predicted and actual ratings
rates_and_preds.take(5)

mse = rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
print("train mse: ", mse)

x = test.map(lambda p: (p[0], p[1]))
p = model.predictAll(x).map(lambda r: ((r[0], r[1]), r[2]))
rates_and_preds = test.map(lambda r: ((r[0], r[1]), r[2])).join(p)
mse = rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
print("test mse: ", mse)
