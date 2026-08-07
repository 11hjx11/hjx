import pandas as pd
try:
    from sqlalchemy import create_engine
except ImportError:
    pass

# 从 config 读取数据库配置（避免硬编码凭据）
try:
    from config import DB_CONFIG
except ImportError:
    DB_CONFIG = None

# 数据库连接（带容错处理，连接失败时回退到 mock 数据）
def get_db_connection():
    if not DB_CONFIG:
        return None
    try:
        url = (
            f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        engine = create_engine(url)
        # Test connection
        engine.connect().close()
        return engine
    except Exception as e:
        print(f"Warning: Cannot connect to database ({e}). Using mock data instead.")
        return None

# 加载用户数据（支持mock回退）
def load_user_data():
    engine = get_db_connection()
    if engine:
        try:
            query = "SELECT user_id, preferred_cuisine, preferred_price_range, age, gender FROM ads_user_value_analysis WHERE dt = '2026-02-14'"
            df = pd.read_sql(query, engine)
            return df
        except Exception as e:
            print(f"Warning: Failed to load user data from database: {e}")
    
    # Mock data fallback
    return pd.DataFrame({
        'user_id': ['user001', 'user002', 'user003', 'user004', 'user005'],
        'preferred_cuisine': ['川菜', '粤菜', '湘菜', '鲁菜', '火锅'],
        'preferred_price_range': ['中档', '高档', '经济型', '中档', '高档'],
        'age': [25, 30, 22, 28, 35],
        'gender': ['男', '女', '男', '女', '男']
    })

# 加载商户数据（支持mock回退）
def load_merchant_data():
    engine = get_db_connection()
    if engine:
        try:
            query = "SELECT merchant_id, merchant_name, area, score, total_orders, avg_order_amount FROM ads_merchant_ranking WHERE dt = '2026-02-14'"
            df = pd.read_sql(query, engine)
            return df
        except Exception as e:
            print(f"Warning: Failed to load merchant data from database: {e}")
    
    # Mock data fallback
    return pd.DataFrame({
        'merchant_id': ['m001', 'm002', 'm003', 'm004', 'm005', 'm006', 'm007', 'm008'],
        'merchant_name': ['川味居', '粤香楼', '湘味轩', '鲁菜馆', '老火锅', '烧烤吧', '西餐厅', '日料店'],
        'area': ['朝阳区', '海淀区', '东城区', '西城区', '朝阳区', '海淀区', '东城区', '西城区'],
        'score': [4.8, 4.7, 4.6, 4.5, 4.9, 4.4, 4.3, 4.2],
        'total_orders': [1200, 980, 850, 720, 1500, 680, 550, 420],
        'avg_order_amount': [85, 120, 65, 95, 75, 110, 200, 150]
    })

# 加载用户-商户交互数据（支持mock回退）
def load_interaction_data():
    engine = get_db_connection()
    if engine:
        try:
            query = "SELECT user_id, merchant_id, interaction_count, total_orders, match_score FROM ads_recommendation_input WHERE dt = '2026-02-14'"
            df = pd.read_sql(query, engine)
            return df
        except Exception as e:
            print(f"Warning: Failed to load interaction data from database: {e}")
    
    # Mock data fallback
    return pd.DataFrame({
        'user_id': ['user001', 'user001', 'user002', 'user002', 'user003', 'user004', 'user005'],
        'merchant_id': ['m001', 'm005', 'm002', 'm006', 'm003', 'm004', 'm007'],
        'interaction_count': [5, 3, 2, 1, 4, 2, 3],
        'total_orders': [5, 3, 2, 1, 4, 2, 3],
        'match_score': [0.95, 0.85, 0.92, 0.88, 0.90, 0.87, 0.86]
    })

# 加载区域数据（支持mock回退）
def load_area_data():
    engine = get_db_connection()
    if engine:
        try:
            query = "SELECT area, total_merchants, total_orders, total_amount, avg_merchant_score FROM ads_area_business_analysis WHERE dt = '2026-02-14'"
            df = pd.read_sql(query, engine)
            return df
        except Exception as e:
            print(f"Warning: Failed to load area data from database: {e}")
    
    # Mock data fallback
    return pd.DataFrame({
        'area': ['朝阳区', '海淀区', '东城区', '西城区', '丰台区'],
        'total_merchants': [256, 189, 145, 178, 134],
        'total_orders': [45200, 32100, 28500, 35600, 22300],
        'total_amount': [3860000, 2580000, 1950000, 2890000, 1560000],
        'avg_merchant_score': [4.6, 4.5, 4.4, 4.5, 4.3]
    })

# 加载业务概览数据（支持mock回退）
def load_business_overview():
    engine = get_db_connection()
    if engine:
        try:
            query = "SELECT * FROM ads_business_overview WHERE dt = '2026-02-14'"
            df = pd.read_sql(query, engine)
            return df
        except Exception as e:
            print(f"Warning: Failed to load business overview from database: {e}")
    
    # Mock data fallback
    return pd.DataFrame([{
        'total_users': 12580,
        'total_merchants': 2345,
        'total_orders': 452300,
        'total_amount': 38600000,
        'avg_order_amount': 85.3,
        'avg_merchant_score': 4.5
    }])