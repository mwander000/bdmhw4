import pyspark
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
import numpy as np
import csv
import datetime
import json
import sys
import os

sc = pyspark.SparkContext()
spark = SparkSession(sc)

# user-added argument output folder path
OUTPUT_PREFIX = sys.argv[0]

rdd = sc.textFile('hdfs:///data/share/bdm/core-places-nyc.csv')

place_naics = {'452210': 'big_box_grocers', '452311': 'big_box_grocers', '445120': 'convenience_stores',
               '722410': 'drinking_places', '722511': 'full_service_restaurants',
               '722513': 'limited_service_restaurants', '446110': 'pharmacies_and_drug_stores',
               '446191': 'pharmacies_and_drug_stores', '311811': 'snack_and_bakeries',
               '722515': 'snack_and_bakeries', '445210': 'specialty_food_stores', '445220': 'specialty_food_stores',
               '445230': 'specialty_food_stores', '445291': 'specialty_food_stores', '445292': 'specialty_food_stores',
               '445299': 'supermarkets_except_convenience_stores'}

# creating a dictionary of safegraph place ids for filtering on weekly patterns
place_ids = dict(
    rdd
    .map(lambda e: next(csv.reader([e])))
    .filter(lambda e: e[9] in place_naics.keys())
    .map(lambda e: (e[1], e[9]))
    .collect()
)

rdd = sc.textFile('hdfs:///data/share/bdm/weekly-patterns-nyc-2019-2020/*')

rdd \
  .map(lambda e: next(csv.reader([e]))) \
  .filter(lambda e: e[1] in place_ids.keys()) \
  .map(lambda e: (place_naics.get(place_ids.get(e[1])), e[12], e[13], e[16])) \
  .filter(lambda e: e[1] >= '2018-12-31T00:00:00-05:00' and e[2] <= '2021-01-04T00:00:00-05:00' ) \
  .map(parseVisits) \
  .flatMap(lambda e: e) \
  .map(lambda e: ((e[0], e[1][0]), e[1][1])) \
  .groupByKey() \
  .map(prepare_rows) \
  .toDF() \
  .write \
  .partitionBy('_1') \
  .option('header', 'true') \
  .csv(OUTPUT_PREFIX)

for dirname in os.listdir(OUTPUT_PREFIX):
    os.rename(OUTPUT_PREFIX + '/' + dirname, OUTPUT_PREFIX + '/' + dirname.replace('_1=', ''))

def parseVisits(e):
    # e[1] == start_date
    # e[2] == end_date
    start_date = datetime.datetime.strptime(e[1][:10], "%Y-%m-%d")
    dates = [start_date + datetime.timedelta(days=n) for n in range(8)]
    visits_by_day = json.loads(e[3])

    if e[1] == '2018-12-31T00:00:00-05:00':
        start_date = datetime.datetime(2019, 1, 1)
        dates = [start_date + datetime.timedelta(days=n) for n in range(7)]
        visits_by_day = visits_by_day[1:]
    elif e[1] == '2020-12-28T00:00:00-05:00':
        end_date = datetime.datetime(2020, 12, 31)
        dates = [datetime.datetime(2020, 12, 28) + datetime.timedelta(days=n) for n in range(4)]
        visits_by_day = visits_by_day[0:4]

    return tuple(zip([e[0]] * len(dates), zip(dates, visits_by_day)))


def prepare_rows(e):
    # e[1] == list of visits by day
    truncate_neg = lambda x: x if x >= 0 else 0

    # median visits by day value for date
    median = int(np.median(tuple(e[1])))
    std = int(np.std(tuple(e[1])))
    low = int(truncate_neg(median - std))
    high = int(median + std)

    # date and year
    year = int(e[0][1].year)

    # for projections onto 2020
    date = str(e[0][1].replace(2020).date())

    return e[0][0], year, date, median, low, high
