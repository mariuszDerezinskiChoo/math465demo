from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

# For this illustrative example, keep the number of trees
# and tree depth small 
N_ESTIMATORS = 10 
MAX_DEPTH = 5

def get_rss(model, x, y):
    predictions = model.predict(x)
    return mean_squared_error(y,predictions)

df = pd.read_csv("./cbb.csv")
df = df.drop(["ADJOE", "ADJDE","BARTHAG","WAB"],axis=1)
df = df.drop(["TEAM","CONF","POSTSEASON","SEED"],axis=1)

test = df['W']
train = df.drop('W',axis=1) # OneHotEncoder(sparse=False).fit_transform(df.drop('W',axis=1))

X_train, X_test, y_train, y_test = train_test_split(train,test, test_size=0.2)

regr = RandomForestRegressor(n_estimators=N_ESTIMATORS,max_depth=3,max_features='sqrt')
regr.fit(X_train,y_train)
print("Random Forest MSE: {:.2f}".format(get_rss(regr,X_test,y_test)))
for index, tree in enumerate(regr.estimators_):
    print("Decision Tree {} RSS: {:.2f}".format(index,get_rss(tree,X_test,y_test)))
    plt.figure(figsize=(20,12))
    plot_tree(tree,fontsize=16,feature_names=X_train.columns,filled=True)
    plt.savefig('tree_{}.png'.format(index), dpi=100)