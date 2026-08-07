
-- 基本配置（适合20000条数据，主节点4GB+两个从节点2GB）
SET hive.exec.parallel=false;
SET hive.auto.convert.join=true;
SET hive.mapjoin.smalltable.filesize=10000000;
SET hive.exec.reducers.max=2;

-- 动态分区配置
SET hive.exec.dynamic.partition=true;
SET hive.exec.dynamic.partition.mode=nonstrict;

-- 资源配置（适合2GB从节点）
SET mapreduce.map.memory.mb=384;
SET mapreduce.reduce.memory.mb=768;
SET mapreduce.map.java.opts=-Xmx288m;
SET mapreduce.reduce.java.opts=-Xmx576m;

-- 内存优化
SET hive.map.aggr=true;
SET hive.groupby.mapaggr.checkinterval=100000;
SET hive.map.aggr.hash.percentmemory=0.4;
SET hive.groupby.skewindata=false;
SET hive.optimize.skewjoin=false;



-- 其他优化
SET hive.merge.mapfiles=true;
SET hive.merge.mapredfiles=true;
SET hive.merge.size.per.task=128000000;
SET hive.merge.smallfiles.avgsize=64000000;
SET hive.optimize.sort.dynamic.partition=false;

-- ==================== 创建数据库 ====================

-- 创建DWS层数据库
CREATE DATABASE IF NOT EXISTS dws COMMENT '汇总数据层' LOCATION '/user/loudi/dws';

-- ==================== DWS层（汇总数据层） ====================

-- 使用DWS数据库
USE dws;

-- 1. 商户维度汇总表
CREATE TABLE IF NOT EXISTS dws.dws_merchant_stats (
    merchant_id STRING COMMENT '商户ID',
    merchant_name STRING COMMENT '商户名称',
    area STRING COMMENT '区域',
    avg_price DOUBLE COMMENT '平均价格',
    score DOUBLE COMMENT '评分',
    main_scene STRING COMMENT '主要场景',
    price_level STRING COMMENT '价格等级',
    score_level STRING COMMENT '评分等级',
    total_orders DOUBLE COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总金额',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    total_views DOUBLE COMMENT '总浏览量',
    total_collections DOUBLE COMMENT '总收藏量',
    total_reviews DOUBLE COMMENT '总评价数',
    avg_rating DOUBLE COMMENT '平均评价分数'
) COMMENT '商户维度汇总表'
PARTITIONED BY (dt STRING COMMENT '分区日期')
STORED AS PARQUET;

-- 2. 用户维度汇总表
CREATE TABLE IF NOT EXISTS dws.dws_user_stats (
    user_id STRING COMMENT '用户ID',
    age INT COMMENT '年龄',
    gender STRING COMMENT '性别',
    location STRING COMMENT '位置',
    occupation STRING COMMENT '职业',
    age_group STRING COMMENT '年龄分组',
    user_level STRING COMMENT '用户等级',
    total_orders DOUBLE COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总消费金额',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    total_views DOUBLE COMMENT '总浏览量',
    total_collections DOUBLE COMMENT '总收藏量',
    total_reviews DOUBLE COMMENT '总评价数',
    preferred_cuisine STRING COMMENT '偏好菜系',
    preferred_price_range STRING COMMENT '偏好价格区间',
    preferred_scene STRING COMMENT '偏好场景'
) COMMENT '用户维度汇总表'
PARTITIONED BY (dt STRING COMMENT '分区日期')
STORED AS PARQUET;

-- 3. 区域维度汇总表
CREATE TABLE IF NOT EXISTS dws.dws_area_stats (
    area STRING COMMENT '区域',
    total_merchants DOUBLE COMMENT '总商户数',
    avg_merchant_score DOUBLE COMMENT '平均商户评分',
    avg_price DOUBLE COMMENT '平均价格',
    total_orders DOUBLE COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总金额',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    popular_cuisine STRING COMMENT '热门菜系',
    popular_scene STRING COMMENT '热门场景'
) COMMENT '区域维度汇总表'
PARTITIONED BY (dt STRING COMMENT '分区日期')
STORED AS PARQUET;

-- 4. 用户商户特质关联表
CREATE TABLE IF NOT EXISTS dws.dws_user_merchant_match (
    user_id STRING COMMENT '用户ID',
    merchant_id STRING COMMENT '商户ID',
    user_age_group STRING COMMENT '用户年龄分组',
    user_level STRING COMMENT '用户等级',
    merchant_price_level STRING COMMENT '商户价格等级',
    merchant_score_level STRING COMMENT '商户评分等级',
    price_match INT COMMENT '价格匹配度',
    scene_match INT COMMENT '场景匹配度',
    total_orders DOUBLE COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总金额',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    interaction_count DOUBLE COMMENT '交互次数'
) COMMENT '用户商户特质关联表'
PARTITIONED BY (dt STRING COMMENT '分区日期')
STORED AS PARQUET;

-- ==================== ETL流程：DWD -> DWS ====================

-- 使用DWS数据库
USE dws;

-- 1. 生成商户维度汇总数据
INSERT OVERWRITE TABLE dws.dws_merchant_stats PARTITION(dt)
SELECT 
    m.merchant_id,
    m.merchant_name,
    m.area,
    ROUND(m.avg_price, 2) AS avg_price,
    ROUND(m.score, 2) AS score,
    m.main_scene,
    m.price_level,
    m.score_level,
    NVL(f.order_count, 0) AS total_orders,
    ROUND(NVL(f.order_amount, 0), 2) AS total_amount,
    ROUND(IF(NVL(f.order_count, 0) > 0, NVL(f.order_amount, 0) / NVL(f.order_count, 0), 0), 2) AS avg_order_amount,
    NVL(f.view_count, 0) AS total_views,
    NVL(f.collect_count, 0) AS total_collections,
    NVL(f.review_count, 0) AS total_reviews,
    ROUND(IF(NVL(f.review_count, 0) > 0, NVL(f.rating_sum, 0) / NVL(f.review_count, 0), 0), 2) AS avg_rating,
    CURRENT_DATE() AS dt
FROM (
    SELECT 
        merchant_id,
        MAX(merchant_name) AS merchant_name,
        MAX(area) AS area,
        MAX(avg_price) AS avg_price,
        MAX(score) AS score,
        MAX(main_scene) AS main_scene,
        MAX(price_level) AS price_level,
        MAX(score_level) AS score_level
    FROM dwd.dim_merchant
    WHERE merchant_id IS NOT NULL
    GROUP BY merchant_id
) m
LEFT JOIN (
    SELECT 
        merchant_id,
        SUM(IF(behavior_type = '下单', 1, 0)) AS order_count,
        SUM(IF(behavior_type = '下单', NVL(amount, 0), 0)) AS order_amount,
        SUM(IF(behavior_type = '浏览', 1, 0)) AS view_count,
        SUM(IF(behavior_type = '收藏', 1, 0)) AS collect_count,
        SUM(IF(behavior_type = '评价', 1, 0)) AS review_count,
        SUM(IF(behavior_type = '评价', NVL(rating, 0), 0)) AS rating_sum
    FROM dwd.fact_behavior
    WHERE merchant_id IS NOT NULL
    GROUP BY merchant_id
) f ON m.merchant_id = f.merchant_id;

-- 2. 生成用户维度汇总数据
INSERT OVERWRITE TABLE dws.dws_user_stats PARTITION(dt)
SELECT 
    u.user_id,
    u.age,
    u.gender,
    u.location,
    u.occupation,
    u.age_group,
    u.user_level,
    NVL(f.order_count, 0) AS total_orders,
    ROUND(NVL(f.order_amount, 0), 2) AS total_amount,
    ROUND(IF(NVL(f.order_count, 0) > 0, NVL(f.order_amount, 0) / NVL(f.order_count, 0), 0), 2) AS avg_order_amount,
    NVL(f.view_count, 0) AS total_views,
    NVL(f.collect_count, 0) AS total_collections,
    NVL(f.review_count, 0) AS total_reviews,
    u.preference_cuisine AS preferred_cuisine,
    u.preference_price_range AS preferred_price_range,
    u.preference_scene AS preferred_scene,
    CURRENT_DATE() AS dt
FROM (
    SELECT 
        user_id,
        MAX(age) AS age,
        MAX(gender) AS gender,
        MAX(location) AS location,
        MAX(occupation) AS occupation,
        MAX(age_group) AS age_group,
        MAX(user_level) AS user_level,
        MAX(preference_cuisine) AS preference_cuisine,
        MAX(preference_price_range) AS preference_price_range,
        MAX(preference_scene) AS preference_scene
    FROM dwd.dim_user
    WHERE user_id IS NOT NULL
    GROUP BY user_id
) u
LEFT JOIN (
    SELECT 
        user_id,
        SUM(IF(behavior_type = '下单', 1, 0)) AS order_count,
        SUM(IF(behavior_type = '下单', NVL(amount, 0), 0)) AS order_amount,
        SUM(IF(behavior_type = '浏览', 1, 0)) AS view_count,
        SUM(IF(behavior_type = '收藏', 1, 0)) AS collect_count,
        SUM(IF(behavior_type = '评价', 1, 0)) AS review_count
    FROM dwd.fact_behavior
    WHERE user_id IS NOT NULL
    GROUP BY user_id
) f ON u.user_id = f.user_id;

-- 3. 生成区域维度汇总数据
INSERT OVERWRITE TABLE dws.dws_area_stats PARTITION(dt)
SELECT 
    t.area,
    t.total_merchants,
    ROUND(t.avg_merchant_score, 2) AS avg_merchant_score,
    ROUND(t.avg_price, 2) AS avg_price,
    NVL(b.order_count, 0) AS total_orders,
    ROUND(NVL(b.order_amount, 0), 2) AS total_amount,
    ROUND(IF(NVL(b.order_count, 0) > 0, NVL(b.order_amount, 0) / NVL(b.order_count, 0), 0), 2) AS avg_order_amount,
    m_top.main_scene AS popular_cuisine,
    m_top.main_scene AS popular_scene,
    CURRENT_DATE() AS dt
FROM (
    SELECT 
        area,
        COUNT(DISTINCT merchant_id) AS total_merchants,
        AVG(score) AS avg_merchant_score,
        AVG(avg_price) AS avg_price
    FROM dwd.dim_merchant
    WHERE area IS NOT NULL
    GROUP BY area
) t
LEFT JOIN (
    SELECT 
        m.area,
        SUM(IF(f.behavior_type = '下单', 1, 0)) AS order_count,
        SUM(IF(f.behavior_type = '下单', NVL(f.amount, 0), 0)) AS order_amount
    FROM dwd.dim_merchant m
    LEFT JOIN dwd.fact_behavior f ON m.merchant_id = f.merchant_id
    WHERE m.area IS NOT NULL
    GROUP BY m.area
) b ON t.area = b.area
LEFT JOIN (
    SELECT 
        area,
        main_scene,
        ROW_NUMBER() OVER (PARTITION BY area ORDER BY score DESC) AS rn
    FROM dwd.dim_merchant
    WHERE area IS NOT NULL
) m_top ON t.area = m_top.area AND m_top.rn = 1;

-- 4. 生成用户商户特质关联数据
INSERT OVERWRITE TABLE dws.dws_user_merchant_match PARTITION(dt)
SELECT 
    f.user_id,
    f.merchant_id,
    u.age_group AS user_age_group,
    u.user_level AS user_level,
    m.price_level AS merchant_price_level,
    m.score_level AS merchant_score_level,
    MAX(w.price_match) AS price_match,
    MAX(w.scene_match) AS scene_match,
    COUNT(CASE WHEN f.behavior_type = '下单' THEN 1 END) AS total_orders,
    ROUND(SUM(CASE WHEN f.behavior_type = '下单' THEN NVL(f.amount, 0) ELSE 0 END), 2) AS total_amount,
    ROUND(CASE WHEN COUNT(CASE WHEN f.behavior_type = '下单' THEN 1 END) > 0 THEN 
        SUM(CASE WHEN f.behavior_type = '下单' THEN NVL(f.amount, 0) ELSE 0 END) / 
        COUNT(CASE WHEN f.behavior_type = '下单' THEN 1 END)
    ELSE 0 END, 2) AS avg_order_amount,
    COUNT(*) AS interaction_count,
    CURRENT_DATE() AS dt
FROM dwd.fact_behavior f
LEFT JOIN dwd.dim_user u ON f.user_id = u.user_id
LEFT JOIN dwd.dim_merchant m ON f.merchant_id = m.merchant_id
LEFT JOIN dwd.dwd_user_merchant_behavior_wide w ON f.behavior_id = w.behavior_id
WHERE f.user_id IS NOT NULL
  AND f.merchant_id IS NOT NULL
GROUP BY 
    f.user_id, f.merchant_id, u.age_group, u.user_level, 
    m.price_level, m.score_level;

