import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
# 读取数据
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# 看一下数据长什么样
print(train_data.head())
# 看一下数据的基本的统计信息
print(train_data.describe())
# 看一下缺失值和数据类型
print(train_data.info())

# 乘客等级
plt.figure(figsize=(8, 6))
sns.countplot(x="Pclass", hue="Survived", data=train_data, palette="viridis")
plt.title("Survival Count by Pclass")
plt.xlabel("Pclass")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# 性别
plt.figure(figsize=(8, 6))
sns.countplot(x="Sex", hue="Survived", data=train_data, palette="viridis")
plt.title("Survival Count by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# 计算年龄区间和对应的存活率
age_bins = range(0, 90, 10)  # 年龄区间，每十岁为一个区间
survival_rate = train_data.groupby(pd.cut(train_data['Age'], bins=age_bins))['Survived'].mean()

# 绘制直方图
plt.bar(survival_rate.index.astype(str), survival_rate.values, align='center')
plt.title('Survival Probability by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Probability')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# 堂兄弟/妹个数
plt.figure(figsize=(8, 6))
sns.countplot(x="SibSp", hue="Survived", data=train_data, palette="viridis")
plt.title("Survival Count by SibSp")
plt.xlabel("SibSp")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# 父母与小孩个数
plt.figure(figsize=(8, 6))
sns.countplot(x="Parch", hue="Survived", data=train_data, palette="viridis")
plt.title("Survival Count by Parch")
plt.xlabel("Parch")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# 计算票价区间和对应的存活率
fare_bins = range(0, 600, 50)  # 年龄区间，每十岁为一个区间
survival_rate = train_data.groupby(pd.cut(train_data['Fare'], bins=fare_bins))['Survived'].mean()

# 绘制直方图
plt.bar(survival_rate.index.astype(str), survival_rate.values, align='center')
plt.title('Survival Probability by Fare Group')
plt.xlabel('Fare Group')
plt.ylabel('Survival Probability')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# 登船港口
plt.figure(figsize=(8, 6))
sns.countplot(x="Embarked", hue="Survived", data=train_data, palette="viridis")
plt.title("Survival Count by Embarked")
plt.xlabel("Embarked")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()

# 缺失值处理
average_age_train = train_data['Age'].mean()
average_age_test = test_data['Age'].mean()
average_fare_test = test_data['Fare'].mean()
max_embarked = train_data['Embarked'].value_counts().idxmax()
train_data['Age'].fillna(average_age_train, inplace=True)
train_data['Embarked'].fillna(max_embarked, inplace=True)
test_data['Age'].fillna(average_age_test, inplace=True)
test_data['Fare'].fillna(average_fare_test, inplace=True)

train_data["FamilySize"] = train_data["SibSp"] + train_data["Parch"]
train_data["NameLength"] = train_data["Name"].apply(lambda x: len(x))
test_data["FamilySize"] = test_data["SibSp"] + test_data["Parch"]
test_data["NameLength"] = test_data["Name"].apply(lambda x: len(x))
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""
# 将称谓保存在titles中
train_titles = train_data["Name"].apply(get_title)
test_titles = test_data["Name"].apply(get_title)
# 将字符型的称谓转换为数值型
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    train_titles[train_titles == k] = v
    test_titles[test_titles == k] = v
train_data['Title'] = train_titles
test_data['Title'] = test_titles
# 将非数值的值替换为1的函数
def replace_non_numeric(value):
    if not isinstance(value, (int, float)):
        return 1
    return value
# 将test_data中title列的所有非数值的值替换为1
test_data['Title'] = test_data['Title'].apply(lambda x: replace_non_numeric(x))

# 选择特征
features = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize',
    'NameLength', 'Title'
]

# 准备特征和目标变量
X = train_data[features]
y = train_data['Survived']
# 从特征列表中移除不希望转换为 one-hot 编码的列
features_to_encode = [feature for feature in features if feature != 'Title']
# 将分类变量转换为虚拟变量（one-hot编码）
X = pd.get_dummies(X[features_to_encode])
# 将未转换的列添加回 DataFrame
X['Title'] = train_data['Title']
# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 创建 SVM 和 RF 模型
svm_model = SVC(kernel='linear', random_state=42)  # 选择线性核
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
# 创建 VotingClassifier 结合 SVM 和 RF
voting_model = VotingClassifier(estimators=[('svm', svm_model), ('rf', rf_model)], voting='hard')
# 创建 BaggingClassifier 结合 SVM 和 RF
bagging_model = BaggingClassifier(base_estimator=voting_model, n_estimators=10, random_state=42)
# 训练 BaggingClassifier
bagging_model.fit(X_train, y_train)
# 预测
predictions = bagging_model.predict(X_test)
# 评估性能
accuracy = accuracy_score(y_test, predictions)
print("Bagging 模型在测试集上的准确率：", accuracy)

# 准备特征和目标变量
X_test = test_data[features]
# 从特征列表中移除不希望转换为 one-hot 编码的列
features_to_encode = [feature for feature in features if feature != 'Title']
# 将分类变量转换为虚拟变量（one-hot编码）
X_test = pd.get_dummies(X_test[features_to_encode])
# 将未转换的列添加回 DataFrame
X_test['Title'] = test_data['Title']
predictions = bagging_model.predict(X_test)
output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission02.csv', index=False)
print("Your submission was successfully saved!")
