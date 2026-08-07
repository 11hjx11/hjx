-- MySQL建表脚本 - 对应Hive ADS层表结构
-- 确保与Hive表结构一致，以便Sqoop工具正常使用

-- 创建数据库（如果不存在）
CREATE DATABASE IF NOT EXISTS ads DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 使用ads数据库
USE ads;

-- ==================== 创建与Hive ADS层对应的MySQL表结构 ====================

-- 1. 商户排行榜
CREATE TABLE IF NOT EXISTS ads_merchant_ranking (
    merchant_id VARCHAR(50) NOT NULL COMMENT '商户ID',
    merchant_name VARCHAR(100) COMMENT '商户名称',
    area VARCHAR(50) COMMENT '区域',
    score DOUBLE COMMENT '评分',
    total_orders DOUBLE COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总金额',
    total_views DOUBLE COMMENT '总浏览量',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    order_ranking INT COMMENT '订单量排名',
    score_ranking INT COMMENT '评分排名',
    composite_ranking INT COMMENT '综合排名',
    dt VARCHAR(10) NOT NULL COMMENT '分区日期',
    PRIMARY KEY (merchant_id, dt)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='商户排行榜';

-- 2. 用户价值分析
CREATE TABLE IF NOT EXISTS ads_user_value_analysis (
    user_id VARCHAR(50) NOT NULL COMMENT '用户ID',
    age INT COMMENT '年龄',
    gender VARCHAR(10) COMMENT '性别',
    total_amount DOUBLE COMMENT '总消费金额',
    total_orders DOUBLE COMMENT '总订单数',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    total_merchants DOUBLE COMMENT '消费商户数',
    activity_level VARCHAR(20) COMMENT '活跃度等级',
    value_level VARCHAR(20) COMMENT '价值等级',
    preferred_cuisine VARCHAR(50) COMMENT '偏好菜系',
    preferred_price_range VARCHAR(50) COMMENT '偏好价格区间',
    dt VARCHAR(10) NOT NULL COMMENT '分区日期',
    PRIMARY KEY (user_id, dt)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户价值分析';

-- 3. 区域商业分析
CREATE TABLE IF NOT EXISTS ads_area_business_analysis (
    area VARCHAR(50) NOT NULL COMMENT '区域',
    total_merchants DOUBLE COMMENT '总商户数',
    total_orders DOUBLE COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总金额',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    avg_merchant_score DOUBLE COMMENT '平均商户评分',
    avg_price DOUBLE COMMENT '平均价格',
    popular_scene VARCHAR(50) COMMENT '热门场景',
    order_density DOUBLE COMMENT '订单密度（订单数/商户数）',
    revenue_per_merchant DOUBLE COMMENT '商户平均收入',
    dt VARCHAR(10) NOT NULL COMMENT '分区日期',
    PRIMARY KEY (area, dt)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='区域商业分析';

-- 4. 智能推荐模型输入
CREATE TABLE IF NOT EXISTS ads_recommendation_input (
    user_id VARCHAR(50) NOT NULL COMMENT '用户ID',
    merchant_id VARCHAR(50) NOT NULL COMMENT '商户ID',
    user_age_group VARCHAR(20) COMMENT '用户年龄分组',
    user_level VARCHAR(20) COMMENT '用户等级',
    merchant_price_level VARCHAR(20) COMMENT '商户价格等级',
    merchant_score_level VARCHAR(20) COMMENT '商户评分等级',
    price_match INT COMMENT '价格匹配度',
    scene_match INT COMMENT '场景匹配度',
    interaction_count DOUBLE COMMENT '交互次数',
    total_orders DOUBLE COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总金额',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    match_score DOUBLE COMMENT '综合匹配分数',
    dt VARCHAR(10) NOT NULL COMMENT '分区日期',
    PRIMARY KEY (user_id, merchant_id, dt)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='智能推荐模型输入';

-- 5. 业务概览指标
CREATE TABLE IF NOT EXISTS ads_business_overview (
    total_users DOUBLE COMMENT '总用户数',
    total_merchants DOUBLE COMMENT '总商户数',
    total_orders DOUBLE COMMENT '总订单数',
    total_amount DOUBLE COMMENT '总金额',
    avg_order_amount DOUBLE COMMENT '平均订单金额',
    avg_merchant_score DOUBLE COMMENT '平均商户评分',
    active_user_ratio DOUBLE COMMENT '活跃用户占比',
    order_conversion_rate DOUBLE COMMENT '订单转化率',
    repeat_purchase_rate DOUBLE COMMENT '复购率',
    dt VARCHAR(10) NOT NULL COMMENT '分区日期',
    PRIMARY KEY (dt)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='业务概览指标';

