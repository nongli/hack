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

DROP VIEW IF EXISTS demo_test.titanic_safe1;
CREATE VIEW demo_test.titanic_safe1
AS SELECT survived, pclass, name, "" as gender, age, sibsp, parch, ticket, fare 
FROM demo_test.titanic_raw;

DROP VIEW IF EXISTS demo_test.titanic_safe2;
CREATE VIEW demo_test.titanic_safe2
AS SELECT survived, pclass, name, "" as gender, 0 as age, sibsp, parch, ticket, fare 
FROM demo_test.titanic_raw;

DROP TABLE IF EXISTS demo_test.cifar_train;
CREATE EXTERNAL TABLE demo_test.cifar_train(label_idx INT, img STRING)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
LOCATION 's3://cerebrodata-test/cifar/train';

DROP TABLE IF EXISTS demo_test.cifar_train_single;
CREATE EXTERNAL TABLE demo_test.cifar_train_single(label_idx INT, img STRING)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
LOCATION 's3://cerebrodata-test/cifar/train/0.ppm';

DROP TABLE IF EXISTS demo_test.cifar_test;
CREATE EXTERNAL TABLE demo_test.cifar_test(label_idx INT, img STRING)
ROW FORMAT DELIMITED FIELDS TERMINATED BY '|'
STORED AS TEXTFILE
LOCATION 's3://cerebrodata-test/cifar/test';
