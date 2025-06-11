import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns  # 这里只为示例，画图后用 mlflow.log_figure 上传

# 1. 开启自动记录
mlflow.set_experiment("iris-classification")
mlflow.sklearn.autolog()

# 2. 载入数据
iris = datasets.load_iris()
X, y = iris.data, iris.target

# 3. 划分训练/测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. 定义模型与超参数网格
model = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 3, 5],
}

# 5. 十折交叉验证 + 网格搜索
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid = GridSearchCV(model, param_grid, cv=cv, scoring="accuracy")

with mlflow.start_run(run_name="rf-gridsearch"):
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    
    # 6. 在测试集上评估
    preds = best.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    
    # 7. 保存并 log 分类报告
    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, preds))
    mlflow.log_artifact("classification_report.txt")
    
    # 8. 绘制并 log 混淆矩阵
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", 
                xticklabels=iris.target_names, 
                yticklabels=iris.target_names)
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
