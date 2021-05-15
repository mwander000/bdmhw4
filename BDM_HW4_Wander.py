from pyspark import SparkContext
import datetime
import csv
import functools
import json
import numpy as np
import sys
 
def main(sc):
    '''
    Transfer our code from the notebook here, however, remember to replace
    the file paths with the ones provided in the problem description.
    '''
    rddPlaces = sc.textFile('/data/share/bdm/core-places-nyc.csv')
    rddPattern = sc.textFile('/data/share/bdm/weekly-patterns-nyc-2019-2020/*')
    OUTPUT_PREFIX = sys.argv[1]

    # Constant dicts
    CAT_CODES = {'452210', '452311', '445120', '722410', '722511', '722513', '446110', '446191', '311181', '722515',
                 '445120', '445220', '445230', '445291', '445292', '445299', '445110'}
    CAT_GROUP = {'452210': 0, '452311': 0, '445120': 1, '722410':2, '722511': 3, '722513': 4, '446110': 5, '446191': 5,
                 '722515': 6, '311811': 6, '445210': 7, '445299': 7, '445230': 7, '445291': 7, '445220': 7, '445292': 7,
                 '445110': 8}
    FILENAMES = {0: 'big_box_grocers', 1: 'convenience_stores', 2: 'drinking_places', 3: 'full_service_restaurants',
                 4: 'limited_service_restaurants', 5: 'pharmacies_and_drug_stores', 6: 'snack_and_bakeries',
                 7: 'specialty_food_stores', 8: 'supermarkets_except_convenience_stores'}

    def filterPOIs(_, lines):
        reader = csv.reader(lines)
        next(reader)
        for line in reader:
          if line[9] in CAT_CODES:
            yield (line[0], CAT_GROUP.get(line[9]))

    def extractVisits(storeGroup, _, lines):
        reader = csv.reader(lines)
        next(reader)
        for line in reader:

            group_number = storeGroup.get(line[0])
            start_date = datetime.datetime.strptime(str(line[12][:10]), "%Y-%m-%d")

            if group_number is not None:

                dates = [(start_date + datetime.timedelta(days=n)).date() for n in range(7)]

                correct_date_diffs = [(date - datetime.date(2019, 1, 1)).days for date in dates if
                                      date.year not in {2018, 2021}]
                correct_indices = [date[0] for date in enumerate(dates) if date[1].year not in {2018, 2021}]

                visits_by_day = [json.loads(line[16])[i] for i in correct_indices]
                row = zip(tuple(zip([group_number] * len(correct_date_diffs), correct_date_diffs)), visits_by_day)
                for elem in row:
                    yield elem

    def computeStats(groupCount, _, records):
        for row in records:
            visits_all_stores = np.concatenate((np.zeros(groupCount[row[0][0]] - len(row[1])), np.array(list(row[1]))))

            # Calculate statistics
            stdev = np.std(visits_all_stores)
            med = int(np.median(visits_all_stores))
            low = int(np.heaviside(med - stdev, 0))
            high = int(med + stdev)

            # Restore dates and projected years
            projected_year = (datetime.datetime(2019, 1, 1) + datetime.timedelta(days=row[0][1])).year
            date = (datetime.datetime(2019, 1, 1) + datetime.timedelta(days=row[0][1])).replace(2020)

            yield row[0][0], ','.join((str(projected_year), str(date.date()), str(med), str(low), str(high)))

    # Filtering groups by key
    rddD = rddPlaces.mapPartitionsWithIndex(filterPOIs) \
        .cache()
 
    storeGroup = dict(rddD.collect())
    groupCount = rddD \
        .map(lambda x: (x[1], 1)) \
        .reduceByKey(lambda x, y: x + y) \
        .sortByKey() \
        .map(lambda x: x[1]) \
        .collect()

    # Exploding visits by day
    rddG = rddPattern \
        .mapPartitionsWithIndex(functools.partial(extractVisits, storeGroup))

    # Computing stats
    rddH = rddG.groupByKey() \
        .mapPartitionsWithIndex(functools.partial(computeStats, groupCount))
        
    # Coalesce for writing
    rddJ = rddH.sortBy(lambda x: x[1][:15])
    header = sc.parallelize([(-1, 'year,date,median,low,high')]).coalesce(1)
    rddJ = (header + rddJ).coalesce(10).cache()

    # Writing
    for group_num in range(9):
      filename = FILENAMES.get(group_num)
      rddJ.filter(lambda x: x[0]==group_num or x[0]==-1).values() \
        .saveAsTextFile(f'{OUTPUT_PREFIX}/{filename}')

if __name__=='__main__':
    sc = SparkContext()
    main(sc)