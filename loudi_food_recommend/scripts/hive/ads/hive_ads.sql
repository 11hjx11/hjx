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

-- 创建ADS层数据库
CREATE DATABASE IF NOT EXISTS ads COMMENT '应用数据层' LOCATION '/user/loudi/ads';

-- ==================== ADS层（应用数据层） ====================

-- 使用ADS数据库
USE ads;

-- 1. 商户排行榜
CREATE TABLE IF NOT EXISTS ads.ads_merchant_ranking (
    merchant_id STRING COMMENT '商户ID',
    merchant_name STRING COMMENT '商户名称',
    area STRING COMMENT '区域',
    score DOUBLE COMMENT '评分',
    total_orders DOUBLE COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总金额',
    total_views DOUBLE COMMENT '总浏览量',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    order_ranking INT COMMENT '订单量排名',
    score_ranking INT COMMENT '评分排名',
    composite_ranking INT COMMENT '综合排名'
) COMMENT '商户排行榜'
PARTITIONED BY (dt STRING COMMENT '分区日期')
STORED AS PARQUET;

-- 2. 用户价值分析
CREATE TABLE IF NOT EXISTS ads.ads_user_value_analysis (
    user_id STRING COMMENT '用户ID',
    age INT COMMENT '年龄',
    gender STRING COMMENT '性别',
    total_amount DOUBLE COMMENT '总消费金额',
    total_orders DOUBLE COMMENT '总订单数',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    total_merchants DOUBLE COMMENT '消费商户数',
    activity_level STRING COMMENT '活跃度等级',
    value_level STRING COMMENT '价值等级',
    preferred_cuisine STRING COMMENT '偏好菜系',
    preferred_price_range STRING COMMENT '偏好价格区间'
) COMMENT '用户价值分析'
PARTITIONED BY (dt STRING COMMENT '分区日期')
STORED AS PARQUET;

-- 3. 区域商业分析
CREATE TABLE IF NOT EXISTS ads.ads_area_business_analysis (
    area STRING COMMENT '区域',
    total_merchants DOUBLE COMMENT '总商户数',
    total_orders DOUBLE COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总金额',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    avg_merchant_score DOUBLE COMMENT '平均商户评分',
    avg_price DOUBLE COMMENT '平均价格',
    popular_scene STRING COMMENT '热门场景',
    order_density DOUBLE COMMENT '订单密度（订单数/商户数）',
    revenue_per_merchant DOUBLE COMMENT '商户平均收入'
) COMMENT '区域商业分析'
PARTITIONED BY (dt STRING COMMENT '分区日期')
STORED AS PARQUET;

-- 4. 智能推荐模型输入
CREATE TABLE IF NOT EXISTS ads.ads_recommendation_input (
    user_id STRING COMMENT '用户ID',
    merchant_id STRING COMMENT '商户ID',
    user_age_group STRING COMMENT '用户年龄分组',
    user_level STRING COMMENT '用户等级',
    merchant_price_level STRING COMMENT '商户价格等级',
    merchant_score_level STRING COMMENT '商户评分等级',
    price_match INT COMMENT '价格匹配度',
    scene_match INT COMMENT '场景匹配度',
    interaction_count DOUBLE COMMENT '交互次数',
    total_orders DOUBLE COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总金额',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    match_score DOUBLE COMMENT '综合匹配分数'
) COMMENT '智能推荐模型输入'
PARTITIONED BY (dt STRING COMMENT '分区日期')
STORED AS PARQUET;

-- 5. 业务概览指标
CREATE TABLE IF NOT EXISTS ads.ads_business_overview (
    total_users DOUBLE COMMENT '总用户数',
    total_merchants DOUBLE COMMENT '总商户数',
    total_orders DOUBLE COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总金额',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    avg_merchant_score DOUBLE COMMENT '平均商户评分',
    active_user_ratio DOUBLE COMMENT '活跃用户占比',
    order_conversion_rate DOUBLE COMMENT '订单转化率',
    repeat_purchase_rate DOUBLE COMMENT '复购率'
) COMMENT '业务概览指标'
PARTITIONED BY (dt STRING COMMENT '分区日期')
STORED AS PARQUET;


-- ==================== ETL流程：DWS -> ADS ====================

-- 使用ADS数据库
USE ads;

-- 1. 生成商户排行榜数据
INSERT OVERWRITE TABLE ads.ads_merchant_ranking PARTITION(dt)
SELECT 
    merchant_id,
    merchant_name,
    area,
    score,
    total_orders,
    total_amount,
    total_views,
    avg_order_amount,
    ROW_NUMBER() OVER (ORDER BY total_orders DESC) AS order_ranking,
    ROW_NUMBER() OVER (ORDER BY score DESC) AS score_ranking,
    ROW_NUMBER() OVER (ORDER BY (total_orders * 0.6 + score * 20) DESC) AS composite_ranking,
    CURRENT_DATE() AS dt
FROM dws.dws_merchant_stats
WHERE dt = CURRENT_DATE();

-- 2. 生成用户价值分析数据
INSERT OVERWRITE TABLE ads.ads_user_value_analysis PARTITION(dt)
SELECT 
    u.user_id,
    u.age,
    u.gender,
    u.total_amount,
    u.total_orders,
    u.avg_order_amount,
    COUNT(DISTINCT m.merchant_id) AS total_merchants,
    CASE 
        WHEN u.total_orders >= 10 THEN '高活跃'
        WHEN u.total_orders >= 5 THEN '中活跃'
        ELSE '低活跃'
    END AS activity_level,
    CASE 
        WHEN u.total_amount >= 1000 THEN '高价值'
        WHEN u.total_amount >= 500 THEN '中价值'
        ELSE '低价值'
    END AS value_level,
    u.preferred_cuisine,
    u.preferred_price_range,
    CURRENT_DATE() AS dt
FROM dws.dws_user_stats u
LEFT JOIN dws.dws_user_merchant_match m ON u.user_id = m.user_id
WHERE u.dt = CURRENT_DATE()
GROUP BY 
    u.user_id, u.age, u.gender, u.total_amount, u.total_orders, 
    u.avg_order_amount, u.preferred_cuisine, u.preferred_price_range;

-- 3. 生成区域商业分析数据
INSERT OVERWRITE TABLE ads.ads_area_business_analysis PARTITION(dt)
SELECT 
    a.area,
    a.total_merchants,
    a.total_orders,
    a.total_amount,
    a.avg_order_amount,
    a.avg_merchant_score,
    a.avg_price,
    a.popular_scene,
    ROUND(a.total_orders / NULLIF(a.total_merchants, 0), 2) AS order_density,
    ROUND(a.total_amount / NULLIF(a.total_merchants, 0), 2) AS revenue_per_merchant,
    CURRENT_DATE() AS dt
FROM dws.dws_area_stats a
WHERE a.dt = CURRENT_DATE();

-- 4. 生成智能推荐模型输入数据
INSERT OVERWRITE TABLE ads.ads_recommendation_input PARTITION(dt)
SELECT 
    m.user_id,
    m.merchant_id,
    m.user_age_group,
    m.user_level,
    m.merchant_price_level,
    m.merchant_score_level,
    m.price_match,
    m.scene_match,
    m.interaction_count,
    m.total_orders,
    m.total_amount,
    m.avg_order_amount,
    ROUND((m.price_match * 0.3 + m.scene_match * 0.3 + 
           m.interaction_count * 0.2 + m.total_orders * 0.2), 2) AS match_score,
    CURRENT_DATE() AS dt
FROM dws.dws_user_merchant_match m
WHERE m.dt = CURRENT_DATE();

-- 5. 生成业务概览指标数据
INSERT OVERWRITE TABLE ads.ads_business_overview PARTITION(dt)
SELECT 
    COUNT(DISTINCT u.user_id) AS total_users,
    COUNT(DISTINCT m.merchant_id) AS total_merchants,
    SUM(m.total_orders) AS total_orders,
    SUM(m.total_amount) AS total_amount,
    ROUND(SUM(m.total_amount) / NULLIF(SUM(m.total_orders), 0), 2) AS avg_order_amount,
    ROUND(AVG(m.score), 2) AS avg_merchant_score,
    ROUND(COUNT(DISTINCT CASE WHEN u.total_orders > 0 THEN u.user_id END) / 
          NULLIF(COUNT(DISTINCT u.user_id), 0), 4) AS active_user_ratio,
    ROUND(SUM(m.total_orders) / NULLIF(SUM(m.total_views), 0), 4) AS order_conversion_rate,
    ROUND(COUNT(DISTINCT CASE WHEN u.total_orders > 1 THEN u.user_id END) / 
          NULLIF(COUNT(DISTINCT CASE WHEN u.total_orders > 0 THEN u.user_id END), 0), 4) AS repeat_purchase_rate,
    CURRENT_DATE() AS dt
FROM dws.dws_user_stats u
CROSS JOIN dws.dws_merchant_stats m
WHERE u.dt = CURRENT_DATE()
  AND m.dt = CURRENT_DATE();

