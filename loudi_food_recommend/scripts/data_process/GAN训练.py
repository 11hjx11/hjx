# ====================== 第一步：导入所需库 ======================
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import warnings

warnings.filterwarnings("ignore")  # 忽略无关警告

# ====================== 第二步：定义业务规则 ======================
relation_rules = {
    "scene_to_time": {
        "早餐": ["06:00-10:00", "07:00-09:00"],
        "工作日早餐": ["06:00-10:00"],
        "午餐": ["11:00-14:00"],
        "晚餐": ["17:00-21:00"],
        "夜宵": ["18:00-23:00"],
        "工作日午餐": ["11:00-14:00"]
    },
    "category_to_dish": {
        "早餐米粉店": ["米粉", "粉"],
        "新化菜": ["三合汤", "向东街"],
        "湘菜": ["剁椒鱼头", "小炒黄牛肉", "辣椒炒肉"],
        "本地小吃": ["糖油粑粑", "炸串"]
    }
}

numeric_rules = {
    "category_to_price": {
        "湘菜": {"min": 40, "max": 90},
        "早餐米粉店": {"min": 10, "max": 30},
        "新化菜": {"min": 30, "max": 80},
        "本地小吃": {"min": 5, "max": 30}
    },
    "score": {"min": 0, "max": 5}
}


def check_business_rules(data):
    """业务规则校验函数"""
    errors = []
    # 校验场景-时间匹配
    scene = data.get("main_scene", "")
    time = data.get("business_hours", "")
    matched_scene = None
    for sk in relation_rules["scene_to_time"].keys():
        if sk in scene:
            matched_scene = sk
            break
    if matched_scene and time not in relation_rules["scene_to_time"][matched_scene]:
        errors.append(f"场景[{scene}]时间[{time}]违规")
    # 校验品类-菜品匹配
    category = data.get("category", "")
    dish = data.get("dish_feature", "")
    if category in relation_rules["category_to_dish"]:
        req_kw = relation_rules["category_to_dish"][category]
        if not any(k in dish for k in req_kw):
            errors.append(f"品类[{category}]菜品[{dish}]违规")
    # 校验价格阈值
    price = data.get("avg_price", 0)
    if category in numeric_rules["category_to_price"]:
        pr = numeric_rules["category_to_price"][category]
        if price < pr["min"] or price > pr["max"]:
            errors.append(f"品类[{category}]价格[{price}]违规")
    # 校验评分阈值
    score = data.get("score", 0)
    if score < 0 or score > 5:
        errors.append(f"评分[{score}]违规")
    return len(errors) == 0, "; ".join(errors)


# ====================== 第三步：数据预处理（核心修复：保留所有分类编码，添加品类分布校验） ======================
def preprocess_data(data_path, encoding="GB2312"):
    """
    数据预处理：
    1. 保留所有分类编码（去掉drop_first）
    2. 打印品类分布，校验有效数据
    3. 返回分类原始类别，用于反解析
    """
    # 1. 读取数据
    df = pd.read_csv(data_path, encoding=encoding)
    print(f"原始数据总行数：{len(df)}")

    # 2. 筛选核心字段
    core_fields = [
        "category", "area", "avg_price", "score", "main_scene", "business_hours", "dish_feature", "relation_tag"
    ]
    for f in core_fields:
        if f not in df.columns:
            df[f] = ""
    df_core = df[core_fields].dropna()
    print(f"有效数据总行数：{len(df_core)}")

    # 关键：打印品类分布，确认是否保留所有品类
    print("\n===== 训练数据品类分布 =====")
    if "category" in df_core.columns:
        print(df_core["category"].value_counts())
    else:
        raise ValueError("数据中无category字段！")

    if len(df_core) == 0:
        raise ValueError("有效数据为空，请检查CSV文件！")

    # 收集所有区域和场景（用于数据增强）
    all_area = df_core["area"].unique().tolist()
    all_scene = df_core["main_scene"].unique().tolist()

    # 定义菜品关键词（用于数据增强）
    dish_keywords = ["米粉", "三合汤", "剁椒鱼头", "糖油粑粑", "向东街", "小炒黄牛肉", "辣椒炒肉", "糖油粑粑", "炸串",
                     "酸辣粉", "牛肉粉", "排骨粉", "新化粗粉", "擂辣椒皮蛋", "香干炒肉", "干锅茶树菇", "红烧肉",
                     "腊味合蒸", "猪血丸子", "酸豆角肉末", "口水鸡", "麻婆豆腐", "鱼香肉丝", "宫保鸡丁", "回锅肉",
                     "梅菜扣肉", "糖醋排骨", "西湖牛肉羹", "蒜蓉粉丝蒸扇贝", "铁板牛肉", "清蒸鲈鱼", "红烧猪脚",
                     "椒盐虾", "蒜蓉空心菜", "清炒时蔬"]

    # 数据增强：增加训练数据的多样性
    print(f"\n开始数据增强...")
    augmented_data = []

    for _, row in df_core.iterrows():
        # 保留原始数据
        augmented_data.append(row.to_dict())

        # 对每条数据生成5个增强版本
        for _ in range(5):
            new_row = row.to_dict().copy()

            # 数值字段：添加小的随机扰动
            new_row["avg_price"] = max(0, row["avg_price"] + np.random.normal(0, 0.05))  # 价格扰动
            new_row["score"] = max(0, min(5, row["score"] + np.random.normal(0, 0.1)))  # 评分扰动

            # 分类字段：在相关类别中随机切换
            if np.random.random() < 0.3:  # 30%概率切换区域
                area_options = [a for a in all_area if a != row["area"]]
                if area_options:
                    new_row["area"] = np.random.choice(area_options)

            if np.random.random() < 0.2:  # 20%概率切换场景
                scene_options = [s for s in all_scene if s != row["main_scene"]]
                if scene_options:
                    new_row["main_scene"] = np.random.choice(scene_options)

            # 菜品特征：随机添加或替换菜品
            if np.random.random() < 0.4:  # 40%概率修改菜品
                original_dish = row["dish_feature"]
                dish_options = [kw for kw in dish_keywords if kw not in original_dish]
                if dish_options:
                    # 随机添加一个新菜品
                    new_dish = f"{original_dish}、{np.random.choice(dish_options)}"
                    new_row["dish_feature"] = new_dish

            augmented_data.append(new_row)

    # 转换为DataFrame
    df_core = pd.DataFrame(augmented_data)
    print(f"数据增强完成！增强后数据量：{len(df_core)}")
    print(f"增强后品类分布：")
    print(df_core["category"].value_counts())

    # 3. 数值字段归一化
    numeric_fields = ["avg_price", "score"]
    # 裁剪异常值
    df_core["avg_price"] = df_core["avg_price"].clip(0, 200)
    df_core["score"] = df_core["score"].clip(0, 5)
    # 归一化到0-1
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_core[numeric_fields] = scaler.fit_transform(df_core[numeric_fields])
    scaler_params = {
        "avg_price": {"min": scaler.data_min_[0], "max": scaler.data_max_[0]},
        "score": {"min": scaler.data_min_[1], "max": scaler.data_max_[1]}
    }

    # 4. 分类字段编码（核心修复：去掉drop_first，保留所有类别）
    cat_fields = ["category", "area", "main_scene"]  # 品类、区域、场景
    # 关键：handle_unknown="ignore" 处理未知类别，不丢弃任何类别
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    cat_encoded = ohe.fit_transform(df_core[cat_fields])
    # 保存原始类别（用于反解析和随机兜底）
    cat_categories = {f: list(ohe.categories_[i]) for i, f in enumerate(cat_fields)}
    # 生成编码列名
    cat_encoded_cols = ohe.get_feature_names_out(cat_fields).tolist()
    df_cat = pd.DataFrame(cat_encoded, columns=cat_encoded_cols, index=df_core.index)

    # 5. 营业时间数值化
    def time_to_num(time_str):
        if "-" in time_str:
            return int(time_str.split("-")[0].split(":")[0]) / 24  # 归一化到0-1
        return 0.0

    df_core["time_num"] = df_core["business_hours"].apply(time_to_num)

    # 6. 标签向量化
    df_core["tag_list"] = df_core["relation_tag"].apply(lambda x: x.split("|") if x.strip() else ["无标签"])
    all_tags = list(set([t for tags in df_core["tag_list"] for t in tags]))
    print(f"\n唯一标签总数：{len(all_tags)}")
    # 标签独热编码
    tag_encoded = np.zeros((len(df_core), len(all_tags)))
    for i, tags in enumerate(df_core["tag_list"]):
        for t in tags:
            if t in all_tags:
                tag_encoded[i, all_tags.index(t)] = 1
    df_tag = pd.DataFrame(tag_encoded, columns=[f"tag_{t}" for t in all_tags], index=df_core.index)

    # 7. 菜品特征编码
    def dish_to_vec(dish_str):
        return [1 if kw in dish_str else 0 for kw in dish_keywords]

    dish_encoded = np.array([dish_to_vec(d) for d in df_core["dish_feature"]])
    df_dish = pd.DataFrame(dish_encoded, columns=[f"dish_{kw}" for kw in dish_keywords], index=df_core.index)

    # 8. 合并所有特征（顺序：基础数值→分类编码→时间→标签→菜品）
    basic_features = df_core[["avg_price", "score"]]  # 2列
    df_features = pd.concat([basic_features, df_cat, df_core[["time_num"]], df_tag, df_dish], axis=1)
    train_data = df_features.values.astype(np.float32)
    feature_dim = train_data.shape[1]
    print(f"\n训练数据特征维度：{feature_dim}")

    return train_data, scaler_params, feature_dim, ohe, cat_categories, all_tags, dish_keywords


# ====================== 第四步：WGAN-GP模型（优化小数据训练稳定性） ======================
class Generator(nn.Module):
    """生成器：优化结构以适应更多数据类型"""

    def __init__(self, latent_dim, feature_dim, hidden_dim=512):  # 进一步增加隐藏层大小到512
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, feature_dim),
            nn.Tanh()  # 输出归一化到[-1,1]
        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """判别器：优化结构以适应更多数据类型"""

    def __init__(self, feature_dim, hidden_dim=512):  # 进一步增加隐藏层大小到512
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),  # 降低dropout率以保留更多特征信息
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # 无Sigmoid
        )

    def forward(self, x):
        return self.model(x)


def gradient_penalty(discriminator, real_data, fake_data, device):
    """WGAN-GP梯度惩罚：提升训练稳定性"""
    alpha = torch.rand(real_data.size(0), 1).to(device)
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    d_interpolated = discriminator(interpolated)
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


# ====================== 第五步：训练WGAN-GP（核心修复：重新生成fake_data，解决反向传播图释放问题） ======================
def train_wgan_gp(train_data, feature_dim, latent_dim=256, epochs=150, batch_size=8, lr=0.0002):
    """
    训练函数：
    1. 更小的batch_size（4），适配小数据
    2. 调整epochs为150，增加训练轮次
    3. 减少判别器训练次数（从5改为3），避免判别器过强
    4. 核心修复：生成器训练时重新生成fake_data，避免复用已释放图的张量
    """
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用训练设备：{device}")

    # 自适应batch_size（确保至少1个批次）
    num_samples = len(train_data)
    if batch_size > num_samples:
        batch_size = num_samples
        print(f"自动调整batch_size为：{batch_size}（匹配数据量）")
    # 数据加载器
    dataset = TensorDataset(torch.tensor(train_data).to(device))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    print(f"训练批次总数：{len(dataloader)}")

    if len(dataloader) == 0:
        raise ValueError("无可用训练批次，请检查数据量！")

    # 初始化模型和优化器
    generator = Generator(latent_dim, feature_dim).to(device)
    discriminator = Discriminator(feature_dim).to(device)
    # Adam优化器，betas=(0.5, 0.9) 适配WGAN
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.9))

    print("\n开始训练WGAN-GP...")
    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        total_loss_D = 0.0
        total_loss_G = 0.0

        for i, (real_data,) in enumerate(dataloader):
            batch_size_current = real_data.size(0)

            # --------------------- 训练判别器 ---------------------
            # 训练判别器（从5次改为3次，避免判别器过强）
            for _ in range(3):
                optimizer_D.zero_grad()
                # 真实数据损失
                real_output = discriminator(real_data)
                loss_real = -torch.mean(real_output)
                # 生成数据损失（临时生成fake_data，仅用于判别器训练）
                noise = torch.randn(batch_size_current, latent_dim).to(device)
                fake_data_d = generator(noise)  # 判别器专用的fake_data
                fake_output = discriminator(fake_data_d.detach())
                loss_fake = torch.mean(fake_output)
                # 梯度惩罚（使用判别器的fake_data）
                gp = gradient_penalty(discriminator, real_data, fake_data_d, device)
                # 判别器总损失
                loss_D = loss_real + loss_fake + 10 * gp  # 10是梯度惩罚权重
                loss_D.backward()
                optimizer_D.step()
                total_loss_D += loss_D.item()

            # --------------------- 训练生成器 ---------------------
            # 核心修复：重新生成fake_data，不再复用判别器的fake_data
            optimizer_G.zero_grad()
            # 重新生成噪声和fake_data，用于生成器训练
            noise_g = torch.randn(batch_size_current, latent_dim).to(device)
            fake_data_g = generator(noise_g)  # 生成器专用的fake_data
            fake_output_g = discriminator(fake_data_g)
            loss_G = -torch.mean(fake_output_g)
            loss_G.backward()  # 此时计算图是新的，不会报错
            optimizer_G.step()
            total_loss_G += loss_G.item()

        # 每10轮打印日志
        if (epoch + 1) % 10 == 0:
            avg_loss_D = total_loss_D / (len(dataloader) * 3)  # 3次判别器训练
            avg_loss_G = total_loss_G / len(dataloader)
            print(f"Epoch [{epoch + 1}/{epochs}] | Avg Loss D: {avg_loss_D:.4f} | Avg Loss G: {avg_loss_G:.4f}")

    # 保存模型
    torch.save(generator.state_dict(), "generator_wgan_gp_fixed.pth")
    print("\n训练完成！模型已保存为：generator_wgan_gp_fixed.pth")
    return generator, device


# ====================== 第六步：生成数据（核心修复：优化分类反解析，强制覆盖所有品类） ======================
def generate_diverse_data(generator, device, scaler_params, ohe, cat_categories, all_tags, dish_keywords, n_samples=100,
                          latent_dim=32):
    """
    生成数据：
    1. 修复分类反解析逻辑，支持所有品类
    2. 强制让生成的品类覆盖原始数据的所有类别
    3. 优化菜品匹配，根据品类生成对应菜品
    """
    generator.eval()
    with torch.no_grad():  # 禁用梯度计算，节省内存
        noise = torch.randn(n_samples, latent_dim).to(device)
        generated_features = generator(noise).cpu().numpy()

    generated_data = []
    # 特征维度索引计算（确保精准提取）
    cat_dim = len(ohe.get_feature_names_out())  # 分类编码维度
    time_idx = 2 + cat_dim  # 时间特征索引（基础数值2列 + 分类编码）
    tag_idx = time_idx + 1  # 标签特征索引（时间1列）
    dish_idx = tag_idx + len(all_tags)  # 菜品特征索引（标签len(all_tags)列）

    # 原始数据的所有品类（用于强制覆盖）
    all_category = cat_categories["category"]
    all_area = cat_categories["area"]
    all_scene = cat_categories["main_scene"]

    # 强制品类多样性：循环遍历所有品类，确保每个品类都有足够的样本
    category_list = []
    for cat in all_category:
        # 为每个品类分配大致相等的样本数
        category_list.extend([cat] * (n_samples // len(all_category)))
    # 补充剩余样本
    while len(category_list) < n_samples:
        category_list.append(np.random.choice(all_category))
    # 打乱顺序
    np.random.shuffle(category_list)

    for i in range(n_samples):
        feat = generated_features[i]

        # 1. 反解析数值字段（价格、评分）
        # 确保价格非负
        price_scale = scaler_params["avg_price"]["max"] - scaler_params["avg_price"]["min"]
        price = max(0, feat[0] * price_scale + scaler_params["avg_price"]["min"])
        # 确保评分在0-5之间
        score = max(0, min(5, feat[1] * (scaler_params["score"]["max"] - scaler_params["score"]["min"]) +
                           scaler_params["score"]["min"]))
        price = round(price, 2)
        score = round(score, 1)

        # 2. 强制品类多样性：直接使用预分配的品类列表，并考虑相关性
        category = category_list[i]

        # 基于品类选择更相关的区域
        area_category_correlation = {
            "湘菜": ["娄星区", "新化县", "双峰县", "冷水江市"],
            "新化菜": ["新化县", "娄星区"],
            "早餐米粉店": ["娄星区", "新化县"],
            "正餐米粉店": ["娄星区", "双峰县"],
            "火锅烤肉": ["娄星区", "新化县"],
            "炸串小吃": ["娄星区", "冷水江市"],
            "本地小吃": ["双峰县", "娄星区"],
            "包子": ["娄星区"],
            "粥": ["娄星区"]
        }
        # 基于品类选择更相关的场景
        scene_category_correlation = {
            "湘菜": ["周末聚餐", "亲子聚餐", "工作日午餐", "学生用餐", "商务聚餐"],
            "新化菜": ["工作日午餐", "周末聚餐", "早餐", "亲子聚餐"],
            "早餐米粉店": ["工作日早餐", "学生早餐"],
            "正餐米粉店": ["学生用餐", "工作日午餐"],
            "火锅烤肉": ["夜宵", "周末聚餐"],
            "炸串小吃": ["学生夜宵", "夜宵"],
            "本地小吃": ["学生用餐", "工作日午餐"],
            "包子": ["工作日早餐"],
            "粥": ["学生早餐"]
        }

        # 根据相关性选择区域和场景
        area = np.random.choice(area_category_correlation.get(category, all_area))
        main_scene = np.random.choice(scene_category_correlation.get(category, all_scene))

        # 3. 反解析营业时间（根据场景匹配）
        time_num = feat[time_idx]
        hour = int(time_num * 24)
        # 匹配场景对应的时间区间
        matched_time = None
        for sk in relation_rules["scene_to_time"].keys():
            if sk in main_scene:
                matched_time = np.random.choice(relation_rules["scene_to_time"][sk])
                break
        business_hours = matched_time if matched_time else np.random.choice(
            ["06:00-10:00", "11:00-14:00", "17:00-21:00"])

        # 4. 反解析菜品（根据品类匹配对应关键词）
        dish_feat = feat[dish_idx:dish_idx + len(dish_keywords)]
        # 菜品候选集（扩展）
        dish_candidates = {
            "米粉": ["新化粗粉", "三鲜粉", "向东街米粉", "牛肉粉", "排骨粉", "酸辣粉"],
            "三合汤": ["新化三合汤", "牛肉三合汤", "猪肉三合汤", "三合汤米粉"],
            "剁椒鱼头": ["剁椒鱼头", "双色剁椒鱼头", "清蒸鱼头", "砂锅鱼头"],
            "糖油粑粑": ["糖油粑粑", "葱油粑粑", "油炸粑粑", "糯米粑粑"],
            "向东街": ["向东街米粉", "向东街老粉馆", "向东街特色粉"],
            "小炒黄牛肉": ["小炒黄牛肉", "湘味黄牛肉", "家常黄牛肉"],
            "辣椒炒肉": ["辣椒炒肉", "农家小炒肉", "经典辣椒炒肉"],
            "其他": ["本地特色菜", "家常菜", "传统美食"]
        }
        # 根据品类选菜品关键词
        if category in relation_rules["category_to_dish"]:
            req_kw = relation_rules["category_to_dish"][category]
            # 从需求关键词中选概率最高的
            kw_scores = [dish_feat[dish_keywords.index(kw)] for kw in req_kw if kw in dish_keywords]
            if kw_scores:
                target_kw = req_kw[np.argmax(kw_scores)]
            else:
                target_kw = req_kw[0]
            # 选对应菜品
            dish_feature = np.random.choice(dish_candidates.get(target_kw, [target_kw]))
        else:
            dish_feature = np.random.choice(dish_keywords)

        # 5. 构建数据并修正业务规则
        data = {
            "category": category,
            "area": area,
            "avg_price": price,
            "score": score,
            "main_scene": main_scene,
            "business_hours": business_hours,
            "dish_feature": dish_feature,
            "relation_tag": f"{category}-{main_scene}-{business_hours[:5]}"
        }

        # 规则修正：确保数据合规
        is_valid, _ = check_business_rules(data)
        if not is_valid:
            # 修正价格到品类合理区间
            if category in numeric_rules["category_to_price"]:
                pr = numeric_rules["category_to_price"][category]
                data["avg_price"] = round(np.random.uniform(pr["min"], pr["max"]), 2)
            # 修正菜品到品类对应关键词
            if category in relation_rules["category_to_dish"]:
                req_kw = relation_rules["category_to_dish"][category]
                data["dish_feature"] = np.random.choice(req_kw)

        generated_data.append(data)

    # 转换为DataFrame并筛选合规数据
    generated_df = pd.DataFrame(generated_data)
    # 只保留合规数据
    compliant_df = [d for d in generated_data if check_business_rules(d)[0]]
    compliant_df = pd.DataFrame(compliant_df)

    # 保存数据
    compliant_df.to_csv("loudi_diverse_generated_data.csv", index=False, encoding="UTF-8")

    # 打印生成结果信息
    print(f"\n===== 生成数据结果 =====")
    print(f"生成样本总数：{n_samples}")
    print(f"合规样本数：{len(compliant_df)}")
    print(f"合规样本占比：{len(compliant_df) / n_samples * 100:.2f}%")
    print(f"\n生成数据品类分布：")
    print(compliant_df["category"].value_counts())
    print(f"\n生成数据示例（前10条）：")
    print(compliant_df[["category", "area", "avg_price", "main_scene", "dish_feature"]].head(10))

    return compliant_df


# ====================== 主程序：一键运行 ======================
if __name__ == "__main__":
    # 替换为增强后的数据集路径
    DATA_PATH = "loudi_enhanced_restaurants.csv"

    # 1. 数据预处理
    train_data, scaler_params, feature_dim, ohe, cat_categories, all_tags, dish_keywords = preprocess_data(DATA_PATH,
                                                                                                           encoding="utf-8")

    # 2. 训练WGAN-GP模型
    generator, device = train_wgan_gp(
        train_data=train_data,
        feature_dim=feature_dim,
        latent_dim=256,  # 维度
        epochs=150,  # 训练轮数
        batch_size=8  # 批量大小
    )

    # 3. 生成多样化数据
    generate_diverse_data(
        generator=generator,
        device=device,
        scaler_params=scaler_params,
        ohe=ohe,
        cat_categories=cat_categories,
        all_tags=all_tags,
        dish_keywords=dish_keywords,
        n_samples=500,
        latent_dim=256
    )

    # 移除生成数据中的relation_tag字段
    generated_data = pd.read_csv('loudi_diverse_generated_data.csv')
    generated_data = generated_data.drop(columns=['relation_tag'])
    generated_data.to_csv('loudi_diverse_generated_data.csv', index=False, encoding='utf-8')
    