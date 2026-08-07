from fastapi import APIRouter
from utils.database import load_user_data, load_merchant_data, load_area_data, load_business_overview
from utils.recommendation import ensure_numeric

router = APIRouter()

# 获取数据可视化数据
@router.get("/data/overview")
def get_overview_data():
    business_df = load_business_overview()
    return business_df.to_dict('records')[0]

# 获取区域数据
@router.get("/data/areas")
def get_area_data():
    area_df = load_area_data()
    return area_df.to_dict('records')

# 获取热门商户
@router.get("/data/top_merchants")
def get_top_merchants():
    merchant_df = load_merchant_data()
    merchant_df = ensure_numeric(merchant_df, ['score', 'total_orders'])
    top_merchants = merchant_df.sort_values(['score', 'total_orders'], ascending=False).head(10)
    return top_merchants.to_dict('records')

# 获取用户偏好分布
@router.get("/data/user_preferences")
def get_user_preferences():
    user_df = load_user_data()
    cuisine_dist = user_df['preferred_cuisine'].value_counts().to_dict()
    price_dist = user_df['preferred_price_range'].value_counts().to_dict()
    return {"cuisine": cuisine_dist, "price": price_dist}