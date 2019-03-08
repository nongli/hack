CREATE DATABASE IF NOT EXISTS demo_test;

DROP TABLE IF EXISTS demo_test.titanic;
CREATE EXTERNAL TABLE demo_test.titanic(
 pclass INT,
 survived INT,
 name STRING,
 sex STRING,
 age INT,
 sibsp INT,
 parch INT,
 ticket INT,
 fare FLOAT)
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
WITH SERDEPROPERTIES('separatorChar'=',', 'skip.header.lin.count'='1')
LOCATION 's3://okera-datalake/tableau/titanic_passengers';
