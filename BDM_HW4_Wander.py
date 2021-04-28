from pyspark import SparkContext
from pyspark.sql import SparkSession
import numpy as np
import csv
import datetime
import json
import sys
import os

def extract_places(partId, records):
    if partId==0:
        next(records)
    reader = csv.reader(records)
    for row in reader:
      if row[1] in place_ids.keys():
        yield (place_naics.get(place_ids.get(row[1])), row[12], row[13], row[16])

def fix_rows(partId, rows):
  for e in rows:
    yield ((e[0], e[1][0]), e[1][1])

def parse_visits(e):
        # e[1] == start_date
        # e[2] == end_date
        start_date = datetime.datetime.strptime(e[1][:10], "%Y-%m-%d")
        dates = [start_date + datetime.timedelta(days=n) for n in range(8)]
        visits_by_day = json.loads(e[3])
        return tuple(zip([e[0]] * len(dates), zip(dates, visits_by_day)))

def parse_visits(partId, rows):
  # e[1] == start_date
  # e[2] == end_date
  for e in rows:
    start_date = datetime.datetime.strptime(e[1][:10], "%Y-%m-%d")
    dates = [start_date + datetime.timedelta(days=n) for n in range(8)]
    visits_by_day = json.loads(e[3])
    visits_by_day = visits_by_day[0:4]
    yield tuple(zip([e[0]]*len(dates), zip(dates, visits_by_day)))

def prepare_rows(partId, rows):
  # e[1] == list of visits by day 
  for e in rows:
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
  
    yield (e[0][0], year, date, median, low, high)

if __name__ == '__main__':
    
    # Initialize spark contexts
    sc = SparkContext()
    spark = SparkSession(sc)

    # user-added argument output folder path
    OUTPUT_PREFIX = sys.argv[1]

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
        .map(lambda e: next(csv.reader([e.encode('utf-8')])))
        .filter(lambda e: e[9] in place_naics.keys())
        .map(lambda e: (e[1], e[9]))
        .collect())

    rdd = sc.textFile('hdfs:///data/share/bdm/weekly-patterns-nyc-2019-2020/part-00000')

    rdd \
      .mapPartitionsWithIndex(extract_places) \
      .mapPartitionsWithIndex(parse_visits) \
      .flatMap(lambda e: e) \
      .mapPartitionsWithIndex(fix_rows) \
      .groupByKey() \
      .mapPartitionsWithIndex(prepare_rows) \
      .toDF(['category', 'year', 'date', 'median', 'low', 'high']) \
      .write \
      .partitionBy('category') \
      .option('header', 'true') \
      .csv(OUTPUT_PREFIX)
