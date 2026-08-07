-- 娄底餐饮推荐系统数据仓库 - DWD层（明细数据层）
-- 基于三层架构：ODS（原始数据层）、DWD（明细数据层）、DWS（汇总数据层）
-- 集群配置：主节点4GB内存，从节点2GB×2

-- ==================== 环境配置 ====================

-- Hive执行优化（适合当前硬件配置）
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

-- 创建DWD层数据库
CREATE DATABASE IF NOT EXISTS dwd COMMENT '明细数据层' LOCATION '/user/loudi/dwd';

-- ==================== DWD层（明细数据层） ====================

-- 使用DWD数据库
USE dwd;

-- 1. 商户维度表
CREATE TABLE IF NOT EXISTS dwd.dim_merchant (
    merchant_id STRING COMMENT '商户ID',
    merchant_name STRING COMMENT '商户名称',
    area STRING COMMENT '区域',
    avg_price DOUBLE COMMENT '平均价格',
    score DOUBLE COMMENT '评分',
    main_scene STRING COMMENT '主要场景',
    business_hours STRING COMMENT '营业时间',
    dish_feature STRING COMMENT '菜品特色',
    price_level STRING COMMENT '价格等级',
    score_level STRING COMMENT '评分等级'
) COMMENT '商户维度表'
STORED AS PARQUET;

-- 2. 用户维度表
CREATE TABLE IF NOT EXISTS dwd.dim_user (
    user_id STRING COMMENT '用户ID',
    age INT COMMENT '年龄',
    gender STRING COMMENT '性别',
    location STRING COMMENT '位置',
    occupation STRING COMMENT '职业',
    preference_cuisine STRING COMMENT '偏好菜系',
    preference_price_range STRING COMMENT '偏好价格区间',
    preference_scene STRING COMMENT '偏好场景',
    registration_date STRING COMMENT '注册日期',
    age_group STRING COMMENT '年龄分组',
    user_level STRING COMMENT '用户等级'
) COMMENT '用户维度表'
STORED AS PARQUET;

-- 3. 行为事实表
CREATE TABLE IF NOT EXISTS dwd.fact_behavior (
    behavior_id STRING COMMENT '行为ID',
    user_id STRING COMMENT '用户ID',
    merchant_id STRING COMMENT '商户ID',
    behavior_type STRING COMMENT '行为类型',
    behavior_time STRING COMMENT '行为时间',
    behavior_date STRING COMMENT '行为日期',
    behavior_hour INT COMMENT '行为小时',
    amount DOUBLE COMMENT '金额',
    rating DOUBLE COMMENT '评分',
    review_content STRING COMMENT '评价内容',
    is_paid INT COMMENT '是否付费',
    is_rated INT COMMENT '是否评价'
) COMMENT '用户行为事实表'
PARTITIONED BY (dt STRING COMMENT '分区日期')
STORED AS PARQUET;

-- 4. 用户-商户-行为明细宽表
CREATE TABLE IF NOT EXISTS dwd.dwd_user_merchant_behavior_wide (
    behavior_id STRING COMMENT '行为ID',
    user_id STRING COMMENT '用户ID',
    user_age INT COMMENT '用户年龄',
    user_gender STRING COMMENT '用户性别',
    user_location STRING COMMENT '用户位置',
    user_occupation STRING COMMENT '用户职业',
    user_preference_cuisine STRING COMMENT '用户偏好菜系',
    user_preference_price_range STRING COMMENT '用户偏好价格区间',
    user_preference_scene STRING COMMENT '用户偏好场景',
    user_age_group STRING COMMENT '用户年龄分组',
    user_level STRING COMMENT '用户等级',
    merchant_id STRING COMMENT '商户ID',
    merchant_name STRING COMMENT '商户名称',
    merchant_area STRING COMMENT '商户区域',
    merchant_avg_price DOUBLE COMMENT '商户平均价格',
    merchant_score DOUBLE COMMENT '商户评分',
    merchant_main_scene STRING COMMENT '商户主要场景',
    merchant_price_level STRING COMMENT '商户价格等级',
    merchant_score_level STRING COMMENT '商户评分等级',
    behavior_type STRING COMMENT '行为类型',
    behavior_time STRING COMMENT '行为时间',
    behavior_date STRING COMMENT '行为日期',
    behavior_hour INT COMMENT '行为小时',
    amount DOUBLE COMMENT '金额',
    rating DOUBLE COMMENT '评分',
    review_content STRING COMMENT '评价内容',
    is_paid INT COMMENT '是否付费',
    is_rated INT COMMENT '是否评价',
    price_match INT COMMENT '价格匹配度',
    scene_match INT COMMENT '场景匹配度'
) COMMENT '用户-商户-行为明细宽表'
PARTITIONED BY (dt STRING COMMENT '分区日期')
STORED AS PARQUET;

-- ==================== ETL流程：ODS -> DWD ====================

-- 启用动态分区
SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;

-- 使用DWD数据库
USE dwd;

-- 1. 加载商户维度数据
INSERT OVERWRITE TABLE dwd.dim_merchant
SELECT 
    merchant_id,
    merchant_name,
    area,
    avg_price,
    score,
    main_scene,
    business_hours,
    dish_feature,
    CASE 
        WHEN avg_price < 20 THEN '低价' 
        WHEN avg_price < 50 THEN '中价' 
        ELSE '高价' 
    END AS price_level,
    CASE 
        WHEN score < 4.0 THEN '低分' 
        WHEN score < 4.5 THEN '中分' 
        ELSE '高分' 
    END AS score_level
FROM ods.merchant
WHERE merchant_id IS NOT NULL
  AND merchant_name IS NOT NULL
  AND avg_price > 0
  AND score >= 1.0
  AND score <= 5.0;

-- 2. 加载用户维度数据
INSERT OVERWRITE TABLE dwd.dim_user
SELECT 
    user_id,
    age,
    gender,
    location,
    occupation,
    preference_cuisine,
    preference_price_range,
    preference_scene,
    registration_date,
    CASE 
        WHEN age < 25 THEN '青年' 
        WHEN age < 40 THEN '中年' 
        ELSE '老年' 
    END AS age_group,
    CASE 
        WHEN occupation = '学生' THEN '学生用户' 
        WHEN occupation IN ('上班族', '工程师') THEN '职场用户' 
        WHEN occupation IN ('教师', '医生', '公务员') THEN '专业用户' 
        ELSE '其他用户' 
    END AS user_level
FROM ods.users
WHERE user_id IS NOT NULL
  AND age >= 18
  AND age <= 80
  AND gender IS NOT NULL;

-- 3. 加载行为事实数据（按日期分区）
INSERT OVERWRITE TABLE dwd.fact_behavior PARTITION(dt)
SELECT 
    behavior_id,
    user_id,
    merchant_id,
    behavior_type,
    behavior_time,
    SUBSTR(behavior_time, 1, 10) AS behavior_date,
    CAST(SUBSTR(behavior_time, 12, 2) AS INT) AS behavior_hour,
    amount,
    rating,
    review_content,
    CASE WHEN behavior_type = '下单' THEN 1 ELSE 0 END AS is_paid,
    CASE WHEN behavior_type = '评价' THEN 1 ELSE 0 END AS is_rated,
    SUBSTR(behavior_time, 1, 10) AS dt
FROM ods.behavior
WHERE behavior_id IS NOT NULL
  AND user_id IS NOT NULL
  AND merchant_id IS NOT NULL
  AND behavior_type IS NOT NULL
  AND behavior_time IS NOT NULL
  AND (amount IS NULL OR (amount > 0 AND amount < 10000))
  AND (rating IS NULL OR (rating >= 1.0 AND rating <= 5.0));

-- 4. 加载用户-商户-行为明细宽表
INSERT OVERWRITE TABLE dwd.dwd_user_merchant_behavior_wide PARTITION(dt)
SELECT 
    f.behavior_id,
    f.user_id,
    u.age AS user_age,
    u.gender AS user_gender,
    u.location AS user_location,
    u.occupation AS user_occupation,
    u.preference_cuisine AS user_preference_cuisine,
    u.preference_price_range AS user_preference_price_range,
    u.preference_scene AS user_preference_scene,
    u.age_group AS user_age_group,
    u.user_level AS user_level,
    f.merchant_id,
    m.merchant_name,
    m.area AS merchant_area,
    m.avg_price AS merchant_avg_price,
    m.score AS merchant_score,
    m.main_scene AS merchant_main_scene,
    m.price_level AS merchant_price_level,
    m.score_level AS merchant_score_level,
    f.behavior_type,
    f.behavior_time,
    f.behavior_date,
    f.behavior_hour,
    f.amount,
    f.rating,
    f.review_content,
    f.is_paid,
    f.is_rated,
    CASE 
        WHEN (u.preference_price_range = '经济型' AND m.price_level = '低价') OR
             (u.preference_price_range = '中档' AND m.price_level = '中价') OR
             (u.preference_price_range = '高档' AND m.price_level = '高价')
        THEN 1 ELSE 0 
    END AS price_match,
    CASE 
        WHEN u.preference_scene = m.main_scene THEN 1 ELSE 0 
    END AS scene_match,
    f.dt
FROM dwd.fact_behavior f
INNER JOIN dwd.dim_user u ON f.user_id = u.user_id
INNER JOIN dwd.dim_merchant m ON f.merchant_id = m.merchant_id;

