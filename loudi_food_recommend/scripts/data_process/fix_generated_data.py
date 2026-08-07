# fix_generated_data.py - 修复生成的娄底餐饮数据中的异常

import pandas as pd
import numpy as np


def fix_generated_data(input_file="loudi_diverse_generated_data.csv", output_file="loudi_improved_generated_data.csv"):
    """
    修复生成的餐饮数据中的各种异常问题
    """
    # 读取生成的数据
    df = pd.read_csv(input_file, encoding="UTF-8")

    # 1. 修复价格异常
    def fix_price(row):
        category = row['category']
        price = row['avg_price']

        # 价格区间规则（根据品类）
        price_ranges = {
            "湘菜": (40, 120),
            "早餐米粉店": (8, 25),
            "新化菜": (35, 100),
            "本地小吃": (15, 60),
            "包子": (4, 15),
            "粥": (3, 12),
            "火锅烤肉": (50, 150),
            "炸串小吃": (20, 80),
            "西餐厅": (60, 200),
            "日韩料理": (50, 180),
            "咖啡馆": (20, 80),
            "甜品店": (10, 50),
            "快餐连锁": (15, 40),
            "自助餐厅": (60, 200),
            "烤鱼店": (50, 120),
            "小龙虾店": (60, 150),
            "湘式快餐": (15, 35),
            "中式快餐": (12, 30),
            "东南亚菜": (50, 150),
            "海鲜餐厅": (80, 250),
            "素食餐厅": (25, 80),
            "特色主题餐厅": (40, 150),
            "烧烤店": (30, 100),
            "面馆": (12, 35)
        }

        if category in price_ranges:
            min_price, max_price = price_ranges[category]
            if price < min_price:
                return int(min_price)
            elif price > max_price:
                return int(max_price)
            else:
                # 在区间内的价格转换为整数
                return int(price)
        # 不在规则内的价格也转换为整数
        return int(price)

    df['avg_price'] = df.apply(fix_price, axis=1)

    # 2. 修复品类-菜品匹配
    def fix_dish_feature(row):
        category = row['category']
        dish = row['dish_feature']

        # 品类对应的菜品关键词
        dish_keywords = {
            "湘菜": ["剁椒鱼头", "小炒黄牛肉", "辣椒炒肉", "酸豆角肉末", "擂辣椒皮蛋"],
            "新化菜": ["新化三合汤", "向东街米粉", "糁子粑", "水车鱼冻", "白溪豆腐"],
            "早餐米粉店": ["肉片粉", "牛肉粉", "三鲜粉", "肉丝粉", "猪油拌粉"],
            "正餐米粉店": ["炒粉", "牛肉粉", "三鲜粉", "排骨粉", "新化粗粉"],
            "本地小吃": ["糖油粑粑", "嗦螺", "葱油饼", "刮凉粉", "甜酒冲蛋"],
            "包子": ["鲜肉包", "红糖馒头", "豆沙包", "奶黄包", "青菜包"],
            "粥": ["皮蛋瘦肉粥", "南瓜粥", "小米粥", "白粥", "黑米粥"],
            "火锅烤肉": ["麻辣火锅", "烤羊肉串", "清汤火锅", "烤生蚝", "鸳鸯火锅"],
            "炸串小吃": ["炸里脊肉", "臭豆腐", "炸土豆", "炸年糕", "炸香蕉"],
            "烧烤店": ["烤羊肉串", "烤牛肉串", "烤五花肉", "烤鸡翅", "烤茄子"],
            "西餐厅": ["牛排", "披萨", "意面", "沙拉", "烤羊排"],
            "日韩料理": ["寿司", "刺身", "石锅拌饭", "烤肉", "拉面"],
            "咖啡馆": ["拿铁", "卡布奇诺", "美式咖啡", "蛋糕", "三明治"],
            "甜品店": ["蛋糕", "奶茶", "冰淇淋", "布丁", "糖水"],
            "快餐连锁": ["汉堡", "薯条", "炸鸡", "可乐", "三明治"],
            "自助餐厅": ["烤肉自助", "火锅自助", "海鲜自助", "中西结合自助", "甜点自助"],
            "烤鱼店": ["烤鱼", "泡椒烤鱼", "麻辣烤鱼", "蒜香烤鱼", "酸菜烤鱼"],
            "小龙虾店": ["麻辣小龙虾", "十三香小龙虾", "蒜蓉小龙虾", "清蒸小龙虾", "椒盐小龙虾"],
            "湘式快餐": ["盖浇饭", "小炒肉饭", "酸辣土豆丝", "青菜", "汤"],
            "中式快餐": ["米饭套餐", "炒菜", "汤", "面条", "炒饭"],
            "东南亚菜": ["冬阴功汤", "咖喱鸡", "菠萝饭", "沙爹", "青木瓜沙拉"],
            "海鲜餐厅": ["海鲜拼盘", "清蒸鱼", "椒盐虾", "蒜蓉粉丝扇贝", "鲍鱼"],
            "素食餐厅": ["蔬菜沙拉", "素炒时蔬", "豆腐煲", "菌菇汤", "素食套餐"],
            "特色主题餐厅": ["特色菜1", "特色菜2", "特色菜3", "特色菜4", "特色菜5"],
            "面馆": ["拉面", "刀削面", "炸酱面", "牛肉面", "羊肉面"]
        }

        # 先检查菜名是否需要修复
        if dish == "向东街":
            # 如果品类是新化菜，使用向东街米粉；否则使用其他合适的菜品
            if category == "新化菜":
                return "向东街米粉"
            elif category in dish_keywords:
                return np.random.choice(dish_keywords[category])
            else:
                return dish

        if category in dish_keywords:
            # 检查菜品是否包含对应品类的关键词
            for keyword in dish_keywords[category]:
                if keyword in dish:
                    return dish
            # 如果不匹配，随机选择一个匹配的菜品
            return np.random.choice(dish_keywords[category])
        return dish

    df['dish_feature'] = df.apply(fix_dish_feature, axis=1)

    # 3. 修复场景-营业时间匹配
    def fix_business_hours(row):
        category = row['category']
        scene = row['main_scene']

        # 特定品类的营业时间限制
        category_hour_restrictions = {
            "早餐米粉店": ["06:00-10:00", "06:30-10:30", "07:00-11:00"],
            "包子": ["06:00-10:00", "06:30-10:30", "07:00-11:00"],
            "粥": ["06:00-10:00", "06:30-10:30", "07:00-11:00"],
            "小龙虾店": ["16:00-02:00", "17:00-01:00", "18:00-00:00"],
            "烤鱼店": ["16:00-23:00", "17:00-22:30", "18:00-22:00"],
            "烧烤店": ["17:00-02:00", "18:00-01:00", "19:00-00:00"]
        }

        # 场景对应的营业时间模板
        hour_templates = {
            "早餐": ["06:00-10:00", "06:30-10:30", "07:00-11:00"],
            "午餐": ["10:30-14:30", "11:00-14:00", "11:30-15:00"],
            "晚餐": ["16:30-21:30", "17:00-22:00", "17:30-22:30"],
            "夜宵": ["17:00-02:00", "18:00-01:00", "19:00-00:00"],
            "下午茶": ["14:00-17:00", "13:30-17:30"],
            "全天": ["10:00-22:00", "09:30-22:30", "10:30-21:30"]
        }

        # 先检查品类限制
        if category in category_hour_restrictions:
            return np.random.choice(category_hour_restrictions[category])

        # 根据场景关键词选择合适的营业时间
        if any(breakfast_keyword in scene for breakfast_keyword in ["早餐", "学生早餐", "工作日早餐", "周末早餐"]):
            return np.random.choice(hour_templates["早餐"])
        elif any(lunch_keyword in scene for lunch_keyword in ["午餐", "学生午餐", "工作日午餐", "周末午餐"]):
            return np.random.choice(hour_templates["午餐"])
        elif any(dinner_keyword in scene for dinner_keyword in ["晚餐", "学生晚餐", "工作日晚餐", "周末晚餐"]):
            return np.random.choice(hour_templates["晚餐"])
        elif any(night_keyword in scene for night_keyword in ["夜宵", "学生夜宵"]):
            return np.random.choice(hour_templates["夜宵"])
        elif "下午茶" in scene:
            return np.random.choice(hour_templates["下午茶"])
        else:  # 其他场景使用全天模板
            return np.random.choice(hour_templates["全天"])

    df['business_hours'] = df.apply(fix_business_hours, axis=1)

    # 4. 修复评分异常（确保评分在3.0-5.0之间，更符合实际业务）
    df['score'] = df['score'].apply(lambda x: max(3.0, min(5.0, x)))

    # 5. 修复菜品描述格式
    def fix_description_format(dish):
        # 确保菜品描述简洁
        if len(dish) > 50:
            return dish[:50] + "..."
        return dish

    df['dish_feature'] = df['dish_feature'].apply(fix_description_format)

    # 保存修复后的数据
    df.to_csv(output_file, index=False, encoding="UTF-8")

    print(f"数据修复完成！修复前：{len(df)}条记录")
    print(f"修复后的数据已保存到：{output_file}")


if __name__ == "__main__":
    fix_generated_data()
