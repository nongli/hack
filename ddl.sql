CREATE DATABASE IF NOT EXISTS demo_test;

DROP TABLE IF EXISTS demo_test.titanic_raw;
CREATE EXTERNAL TABLE demo_test.titanic_raw(
 survived INT,
 pclass INT,
 name STRING,
 gender STRING,
 age float,
 sibsp INT,
 parch INT,
 ticket INT,
 fare FLOAT)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES('separatorChar'=',', 'skip.header.lin.count'='1')
LOCATION 's3://okera-datalake/tableau/titanic_passengers';

DROP VIEW IF EXISTS demo_test.titanic;
CREATE VIEW demo_test.titanic
AS SELECT * from demo_test.titanic_raw;
