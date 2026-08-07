
SET hive.exec.parallel=false;
SET hive.auto.convert.join=true;
SET hive.mapjoin.smalltable.filesize=25000000;
SET hive.groupby.skewindata=true;
SET hive.exec.reducers.bytes.per.reducer=256000000;
SET hive.exec.reducers.max=4;
SET hive.exec.max.dynamic.partitions=1000;
SET hive.exec.max.dynamic.partitions.pernode=200;
SET hive.optimize.sort.dynamic.partition=true;
SET hive.vectorized.execution.enabled=false;
SET hive.vectorized.execution.reduce.enabled=false;
SET hive.cbo.enable=false;

-- 资源配置
SET mapreduce.map.memory.mb=768;
SET mapreduce.reduce.memory.mb=1536;
SET mapreduce.map.java.opts=-Xmx576m;
SET mapreduce.reduce.java.opts=-Xmx1152m;
SET mapreduce.task.io.sort.mb=64;
SET mapreduce.task.io.sort.factor=64;
SET mapreduce.map.sort.spill.percent=0.80;
SET mapreduce.reduce.shuffle.parallelcopies=20;
SET mapreduce.reduce.shuffle.input.buffer.percent=0.60;
SET mapreduce.reduce.input.buffer.percent=0.50;

-- ==================== 创建数据库 ====================

-- 创建ODS层数据库
CREATE DATABASE IF NOT EXISTS ods COMMENT '原始数据层' LOCATION '/user/loudi/ods';

-- ==================== ODS层（原始数据层） ====================

-- 使用ODS数据库
USE ods;

-- 1. 商户原始数据表
CREATE EXTERNAL TABLE IF NOT EXISTS ods.merchant (
    merchant_id STRING COMMENT '商户ID',
    merchant_name STRING COMMENT '商户名称',
    area STRING COMMENT '区域',
    avg_price DOUBLE COMMENT '平均价格',
    score DOUBLE COMMENT '评分',
    main_scene STRING COMMENT '主要场景',
    business_hours STRING COMMENT '营业时间',
    dish_feature STRING COMMENT '菜品特色'
) COMMENT '商户原始数据表'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/loudi/ods/merchant';

-- 2. 用户原始数据表
CREATE EXTERNAL TABLE IF NOT EXISTS ods.users (
    user_id STRING COMMENT '用户ID',
    age INT COMMENT '年龄',
    gender STRING COMMENT '性别',
    location STRING COMMENT '位置',
    occupation STRING COMMENT '职业',
    preference_cuisine STRING COMMENT '偏好菜系',
    preference_price_range STRING COMMENT '偏好价格区间',
    preference_scene STRING COMMENT '偏好场景',
    registration_date STRING COMMENT '注册日期'
) COMMENT '用户原始数据表'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/loudi/ods/user';

-- 3. 行为原始数据表
CREATE EXTERNAL TABLE IF NOT EXISTS ods.behavior (
    behavior_id STRING COMMENT '行为ID',
    user_id STRING COMMENT '用户ID',
    merchant_id STRING COMMENT '商户ID',
    merchant_area STRING COMMENT '商户区域',
    merchant_avg_price DOUBLE COMMENT '商户平均价格',
    merchant_score DOUBLE COMMENT '商户评分',
    merchant_scene STRING COMMENT '商户场景',
    behavior_type STRING COMMENT '行为类型',
    behavior_time STRING COMMENT '行为时间',
    amount DOUBLE COMMENT '金额',
    rating DOUBLE COMMENT '评分',
    review_content STRING COMMENT '评价内容'
) COMMENT '用户行为原始数据表'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY ','
STORED AS TEXTFILE
LOCATION '/user/loudi/ods/behavior';

