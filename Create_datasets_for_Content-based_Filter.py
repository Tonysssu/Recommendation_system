# collect the data via a collection of SQL queries from the publicly avialable Kurier.at dataset in BigQuery

import os
import tensorflow as tf
import numpy as np
import google.datalab.bigquery as bq

# PROJECT = 'cloud-training'
# BUCKET = 'cloud-training-ml'
# REGION = 'us-central1'
#
# os.environ['PROJECT'] = PROJECT
# os.environ['BUCKET'] = BUCKET
# os.environ['REGION'] = REGION
# os.environ['TFVERSION'] = '1.8'

# Helper functio to write list of info to local files
def write_list_to_disk(my_list, filename):
    with open(filename, "w") as f:
        for item in my_list:
            line = "%s\n" % item
            f.write(line.encode("utf8"))


# Pull data from BigQuery
# Content_id
import google.datalab.bigquery as bq

sql = """
SELECT
  (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) AS content_id
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,
  UNNEST(hits) AS hits
WHERE
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
GROUP BY
  content_id
"""
content_ids_list = (
    bq.Query(sql).execute().result().to_dataframe()["content_id"].tolist()
)
write_list_to_disk(content_ids_list, "content_ids.txt")
# print("Some sample content IDs {}".format(content_ids_list[:3]))
# print("The total number of articles is {}".format(len(content_ids_list)))
# Some sample content IDs ['299965853', '299972248', '299410466']
# The total number of articles is 15634

# Category
sql = """
SELECT
  (SELECT MAX(IF(index=7, value, NULL)) FROM UNNEST(hits.customDimensions)) AS category
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,
  UNNEST(hits) AS hits
WHERE
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index=7, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
GROUP BY
  category
"""
categories_list = bq.Query(sql).execute().result().to_dataframe()["category"].tolist()
write_list_to_disk(categories_list, "categories.txt")
# print(categories_list)
# Only three different categories

# Author
sql = """
SELECT
  REGEXP_EXTRACT((SELECT MAX(IF(index=2, value, NULL)) FROM UNNEST(hits.customDimensions)), r"^[^,]+")  AS first_author
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,
  UNNEST(hits) AS hits
WHERE
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index=2, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
GROUP BY
  first_author
"""
authors_list = bq.Query(sql).execute().result().to_dataframe()["first_author"].tolist()
write_list_to_disk(authors_list, "authors.txt")
# print("Some sample authors {}".format(authors_list[:10]))
# print("The total number of authors is {}".format(len(authors_list)))

# Create train and test set
## Use the concatenated values for visitor id and content id to create a farm fingerprint,
## taking approximately 90% of the data for the training set and 10% for the test set

sql = """
WITH site_history as (
  SELECT
      fullVisitorId as visitor_id,
      (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) AS content_id,
      (SELECT MAX(IF(index=7, value, NULL)) FROM UNNEST(hits.customDimensions)) AS category,
      (SELECT MAX(IF(index=6, value, NULL)) FROM UNNEST(hits.customDimensions)) AS title,
      (SELECT MAX(IF(index=2, value, NULL)) FROM UNNEST(hits.customDimensions)) AS author_list,
      SPLIT(RPAD((SELECT MAX(IF(index=4, value, NULL)) FROM UNNEST(hits.customDimensions)), 7), '.') as year_month_array,
      LEAD(hits.customDimensions, 1) OVER (PARTITION BY fullVisitorId ORDER BY hits.time ASC) as nextCustomDimensions
  FROM
    `cloud-training-demos.GA360_test.ga_sessions_sample`,
     UNNEST(hits) AS hits
   WHERE
     # only include hits on pages
      hits.type = "PAGE"
      AND
      fullVisitorId IS NOT NULL
      AND
      hits.time != 0
      AND
      hits.time IS NOT NULL
      AND
      (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
)
SELECT
  visitor_id,
  content_id,
  category,
  REGEXP_REPLACE(title, r",", "") as title,
  REGEXP_EXTRACT(author_list, r"^[^,]+") as author,
  DATE_DIFF(DATE(CAST(year_month_array[OFFSET(0)] AS INT64), CAST(year_month_array[OFFSET(1)] AS INT64), 1), DATE(1970,1,1), MONTH) as months_since_epoch,
  (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) as next_content_id
FROM
  site_history
WHERE (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) IS NOT NULL
      AND MOD(ABS(FARM_FINGERPRINT(CONCAT(visitor_id, content_id))), 10) < 9
"""
training_set_df = bq.Query(sql).execute().result().to_dataframe()
training_set_df.to_csv("training_set.csv", header=False, index=False, encoding="utf-8")
# training_set_df.head()


sql = """
WITH site_history as (
  SELECT
      fullVisitorId as visitor_id,
      (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) AS content_id,
      (SELECT MAX(IF(index=7, value, NULL)) FROM UNNEST(hits.customDimensions)) AS category,
      (SELECT MAX(IF(index=6, value, NULL)) FROM UNNEST(hits.customDimensions)) AS title,
      (SELECT MAX(IF(index=2, value, NULL)) FROM UNNEST(hits.customDimensions)) AS author_list,
      SPLIT(RPAD((SELECT MAX(IF(index=4, value, NULL)) FROM UNNEST(hits.customDimensions)), 7), '.') as year_month_array,
      LEAD(hits.customDimensions, 1) OVER (PARTITION BY fullVisitorId ORDER BY hits.time ASC) as nextCustomDimensions
  FROM
    `cloud-training-demos.GA360_test.ga_sessions_sample`,
     UNNEST(hits) AS hits
   WHERE
     # only include hits on pages
      hits.type = "PAGE"
      AND
      fullVisitorId IS NOT NULL
      AND
      hits.time != 0
      AND
      hits.time IS NOT NULL
      AND
      (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
)
SELECT
  visitor_id,
  content_id,
  category,
  REGEXP_REPLACE(title, r",", "") as title,
  REGEXP_EXTRACT(author_list, r"^[^,]+") as author,
  DATE_DIFF(DATE(CAST(year_month_array[OFFSET(0)] AS INT64), CAST(year_month_array[OFFSET(1)] AS INT64), 1), DATE(1970,1,1), MONTH) as months_since_epoch,
  (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) as next_content_id
FROM
  site_history
WHERE (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) IS NOT NULL
      AND MOD(ABS(FARM_FINGERPRINT(CONCAT(visitor_id, content_id))), 10) >= 9
"""
test_set_df = bq.Query(sql).execute().result().to_dataframe()
test_set_df.to_csv("test_set.csv", header=False, index=False, encoding="utf-8")
# test_set_df.head()
