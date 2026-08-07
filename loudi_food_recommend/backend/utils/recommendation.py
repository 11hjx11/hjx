import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# 菜系关键词映射：用于在商户名中识别菜系（商户数据无 cuisine 字段时的容错匹配）
CUISINE_KEYWORDS = {
    '川菜': ['川', '辣', '麻辣', '蜀', '巴蜀', '天府'],
    '粤菜': ['粤', '广东', '茶餐厅', '港', '点心', '烧腊'],
    '湘菜': ['湘', '湖南'],
    '鲁菜': ['鲁', '山东', '齐鲁'],
    '苏菜': ['苏', '江苏', '淮扬', '金陵'],
    '火锅': ['火锅', '麻辣烫', '串串', '锅'],
    '烧烤': ['烧烤', '烤肉', '串', '烤'],
    '西餐': ['西餐', '牛排', '法餐', '意面', '西式'],
    '日料': ['日料', '寿司', '拉面', '日本', '居酒屋', '刺身'],
    '韩料': ['韩料', '韩式', '泡菜', '韩国'],
}


def ensure_numeric(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """确保指定列是数值类型（返回副本，不修改原 DataFrame）"""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    return df


# 创建用户-商户交互矩阵
def create_user_merchant_matrix(interaction_df):
    # 用副本避免修改入参（副作用）
    df = interaction_df.copy()
    df['interaction_count'] = pd.to_numeric(df['interaction_count'], errors='coerce').fillna(0)

    # 创建用户-商户矩阵
    user_merchant_matrix = df.pivot_table(
        index='user_id',
        columns='merchant_id',
        values='interaction_count',
        fill_value=0
    )
    return user_merchant_matrix

# 协同过滤推荐
def collaborative_filtering(user_id, user_merchant_matrix, interaction_df, merchant_df, n_recommendations=5):
    if user_id not in user_merchant_matrix.index:
        # 新用户，基于内容推荐
        return content_based_recommendation(merchant_df, n_recommendations)

    # 计算用户相似度
    user_index = user_merchant_matrix.index.get_loc(user_id)
    user_vector = user_merchant_matrix.iloc[user_index].values.reshape(1, -1)
    similarities = cosine_similarity(user_merchant_matrix, user_vector).flatten()

    # 找到最相似的用户
    similar_users = np.argsort(similarities)[::-1][1:6]  # 排除自己
    similar_user_ids = user_merchant_matrix.index[similar_users]

    # 获取相似用户喜欢的商户
    recommended_merchants = set()
    interaction_df_num = ensure_numeric(interaction_df, ['interaction_count'])
    for similar_user in similar_user_ids:
        user_interactions = interaction_df_num[interaction_df_num['user_id'] == similar_user]
        top_merchants = user_interactions.sort_values('interaction_count', ascending=False).head(10)
        recommended_merchants.update(top_merchants['merchant_id'].tolist())

    # 排除用户已经交互过的商户
    user_interacted = interaction_df[interaction_df['user_id'] == user_id]['merchant_id'].tolist()
    recommended_merchants = [m for m in recommended_merchants if m not in user_interacted]

    # 获取商户信息，按推荐度排序（相似度越高、交互越多越靠前）
    merchant_df_num = ensure_numeric(merchant_df, ['score', 'total_orders'])
    recommendations = merchant_df_num[
        merchant_df_num['merchant_id'].isin(recommended_merchants)
    ].sort_values(['score', 'total_orders'], ascending=False).head(n_recommendations)
    return recommendations

# 基于内容的推荐
def content_based_recommendation(merchant_df, n_recommendations=5):
    # 确保merchant_df不为空
    if merchant_df.empty:
        return merchant_df

    # 统一转 numeric（返回副本，不修改入参）
    merchant_df_num = ensure_numeric(merchant_df, ['score', 'total_orders'])

    # 基于商户评分和销量推荐
    recommendations = merchant_df_num.sort_values(['score', 'total_orders'], ascending=False).head(n_recommendations)
    return recommendations

# 基于标签的推荐
def tag_based_recommendation(cuisine, price_range, merchant_df, n_recommendations=5):
    # 确保merchant_df不为空
    if merchant_df.empty:
        return merchant_df

    # 统一转 numeric（返回副本）
    filtered = ensure_numeric(merchant_df, ['avg_order_amount', 'score', 'total_orders'])

    # 菜系过滤 - 用关键词映射匹配商户名（容错：商户数据无 cuisine 字段）
    if cuisine:
        keywords = CUISINE_KEYWORDS.get(cuisine, [cuisine])
        pattern = '|'.join(keywords)
        cuisine_filtered = filtered[filtered['merchant_name'].str.contains(pattern, case=False, na=False, regex=True)]
        if not cuisine_filtered.empty:
            filtered = cuisine_filtered

    # 价格区间过滤
    if price_range:
        if price_range == '经济型':
            filtered = filtered[filtered['avg_order_amount'] < 50]
        elif price_range == '中档':
            filtered = filtered[(filtered['avg_order_amount'] >= 50) & (filtered['avg_order_amount'] < 100)]
        elif price_range == '高档':
            filtered = filtered[filtered['avg_order_amount'] >= 100]

    # 如果过滤后为空，回退到基于评分和销量的推荐
    if filtered.empty:
        return ensure_numeric(merchant_df, ['score', 'total_orders', 'avg_order_amount']).sort_values(
            ['score', 'total_orders'], ascending=False
        ).head(n_recommendations)

    # 排序并返回
    recommendations = filtered.sort_values(['score', 'total_orders'], ascending=False).head(n_recommendations)
    return recommendations