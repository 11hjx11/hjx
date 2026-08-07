"""
推荐算法单元测试
覆盖：ensure_numeric、create_user_merchant_matrix、协同过滤、内容推荐、标签推荐、菜系关键词匹配

运行方式（在 backend/ 目录下）：
    pytest tests/test_recommendation.py -v
"""
import sys
import os
import pandas as pd
import pytest

# 将 backend 目录加入 sys.path，使测试可在任意目录运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.recommendation import (
    ensure_numeric,
    create_user_merchant_matrix,
    collaborative_filtering,
    content_based_recommendation,
    tag_based_recommendation,
    CUISINE_KEYWORDS,
)


# ============================================================
# 测试夹具：构造小型 mock 数据（不依赖数据库）
# ============================================================

@pytest.fixture
def merchant_df():
    """模拟商户数据（含字符串数值列，用于验证 numeric 转换）"""
    return pd.DataFrame({
        'merchant_id': ['m001', 'm002', 'm003', 'm004', 'm005', 'm006'],
        'merchant_name': ['川味居', '粤香楼', '湘味轩', '老火锅', '西餐厅', '麻辣烫小馆'],
        'area': ['朝阳区', '海淀区', '东城区', '朝阳区', '西城区', '丰台区'],
        # 故意用字符串类型，验证 ensure_numeric 能正确转换
        'score': ['4.8', '4.7', '4.6', '4.9', '4.3', '4.2'],
        'total_orders': ['1200', '980', '850', '1500', '550', '320'],
        'avg_order_amount': ['85', '120', '65', '75', '200', '35'],
    })


@pytest.fixture
def interaction_df():
    """模拟用户-商户交互数据"""
    return pd.DataFrame({
        'user_id': ['user001', 'user001', 'user002', 'user002', 'user003'],
        'merchant_id': ['m001', 'm004', 'm002', 'm005', 'm003'],
        'interaction_count': [5, 3, 2, 1, 4],
        'total_orders': [5, 3, 2, 1, 4],
        'match_score': [0.95, 0.85, 0.92, 0.88, 0.90],
    })


# ============================================================
# ensure_numeric 测试
# ============================================================

class TestEnsureNumeric:
    def test_converts_string_columns(self, merchant_df):
        """字符串数值列应被转为数值类型"""
        result = ensure_numeric(merchant_df, ['score', 'total_orders'])
        assert result['score'].dtype in ('float64', 'int64')
        assert result['total_orders'].dtype in ('float64', 'int64')
        assert result['score'].iloc[0] == 4.8

    def test_does_not_modify_original(self, merchant_df):
        """ensure_numeric 不应修改原 DataFrame"""
        original_type = merchant_df['score'].dtype
        ensure_numeric(merchant_df, ['score'])
        assert merchant_df['score'].dtype == original_type

    def test_missing_column_ignored(self, merchant_df):
        """不存在的列应被静默忽略"""
        result = ensure_numeric(merchant_df, ['nonexistent_column'])
        assert len(result) == len(merchant_df)

    def test_invalid_values_become_zero(self, merchant_df):
        """非法值应转为 0"""
        df = merchant_df.copy()
        df.loc[0, 'score'] = 'invalid'
        result = ensure_numeric(df, ['score'])
        assert result['score'].iloc[0] == 0


# ============================================================
# create_user_merchant_matrix 测试
# ============================================================

class TestCreateUserMerchantMatrix:
    def test_matrix_shape(self, interaction_df):
        """矩阵应为 user × merchant 的透视表"""
        matrix = create_user_merchant_matrix(interaction_df)
        assert matrix.shape[0] == 3  # 3 个用户
        assert matrix.shape[1] == 5  # 5 个商户

    def test_no_side_effect(self, interaction_df):
        """create_user_merchant_matrix 不应修改入参 DataFrame"""
        original_values = interaction_df['interaction_count'].tolist()
        create_user_merchant_matrix(interaction_df)
        assert interaction_df['interaction_count'].tolist() == original_values

    def test_fill_value_zero(self, interaction_df):
        """未交互的格子应填充为 0"""
        matrix = create_user_merchant_matrix(interaction_df)
        # user003 只和 m003 交互过，其他应为 0
        assert matrix.loc['user003', 'm001'] == 0
        assert matrix.loc['user003', 'm003'] == 4


# ============================================================
# content_based_recommendation 测试
# ============================================================

class TestContentBasedRecommendation:
    def test_returns_top_n(self, merchant_df):
        """应返回指定数量的推荐"""
        recs = content_based_recommendation(merchant_df, n_recommendations=3)
        assert len(recs) == 3

    def test_sorted_by_score(self, merchant_df):
        """应按 score 降序排列"""
        recs = content_based_recommendation(merchant_df, n_recommendations=5)
        scores = recs['score'].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_empty_df(self):
        """空 DataFrame 应返回空"""
        empty = pd.DataFrame()
        recs = content_based_recommendation(empty)
        assert recs.empty


# ============================================================
# collaborative_filtering 测试
# ============================================================

class TestCollaborativeFiltering:
    def test_new_user_fallback(self, merchant_df, interaction_df):
        """新用户应回退到内容推荐"""
        matrix = create_user_merchant_matrix(interaction_df)
        recs = collaborative_filtering('new_user', matrix, interaction_df, merchant_df)
        assert not recs.empty
        assert len(recs) <= 5

    def test_existing_user(self, merchant_df, interaction_df):
        """老用户应返回推荐（排除已交互商户）"""
        matrix = create_user_merchant_matrix(interaction_df)
        recs = collaborative_filtering('user001', matrix, interaction_df, merchant_df)
        # user001 已交互 m001/m004，推荐不应包含
        assert 'm001' not in recs['merchant_id'].values
        assert 'm004' not in recs['merchant_id'].values


# ============================================================
# tag_based_recommendation 测试（重点验证菜系匹配修复）
# ============================================================

class TestTagBasedRecommendation:
    def test_cuisine_match_sichuan(self, merchant_df):
        """川菜关键词应匹配到含'川'的商户名（修复前用'川菜'匹配会失效）"""
        recs = tag_based_recommendation('川菜', None, merchant_df)
        assert not recs.empty
        # 川味居 含 '川'
        assert '川味居' in recs['merchant_name'].values

    def test_cuisine_match_hotpot(self, merchant_df):
        """火锅关键词应匹配到含'火锅'/'锅'的商户名"""
        recs = tag_based_recommendation('火锅', None, merchant_df)
        assert not recs.empty
        assert '老火锅' in recs['merchant_name'].values

    def test_price_filter_low(self, merchant_df):
        """经济型过滤：人均 < 50"""
        recs = tag_based_recommendation(None, '经济型', merchant_df)
        if not recs.empty:
            assert (recs['avg_order_amount'] < 50).all()

    def test_price_filter_high(self, merchant_df):
        """高档过滤：人均 >= 100"""
        recs = tag_based_recommendation(None, '高档', merchant_df)
        if not recs.empty:
            assert (recs['avg_order_amount'] >= 100).all()

    def test_no_match_fallback(self, merchant_df):
        """菜系无匹配时应回退到评分排序"""
        recs = tag_based_recommendation('外星菜', None, merchant_df)
        assert not recs.empty

    def test_empty_df(self):
        """空 DataFrame 应返回空"""
        empty = pd.DataFrame()
        recs = tag_based_recommendation('川菜', '中档', empty)
        assert recs.empty


# ============================================================
# CUISINE_KEYWORDS 映射完整性测试
# ============================================================

class TestCuisineKeywords:
    def test_all_cuisines_have_keywords(self):
        """每个菜系至少有一个关键词"""
        for cuisine, keywords in CUISINE_KEYWORDS.items():
            assert len(keywords) > 0, f"菜系 {cuisine} 关键词为空"

    def test_known_cuisines_present(self):
        """常见菜系应存在于映射中"""
        required = ['川菜', '粤菜', '湘菜', '火锅', '烧烤', '西餐', '日料']
        for c in required:
            assert c in CUISINE_KEYWORDS, f"缺少菜系: {c}"
