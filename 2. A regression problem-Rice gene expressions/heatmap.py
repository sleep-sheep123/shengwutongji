

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dict = {"1-mer" : np.round(np.array([0.6193719258418464,0.6307226636398032,0.5909950813469542,0.6519107075293228,0.620506999621642,0.6401816118047673]),3),
        "2-mer" : np.round(np.array([0.7983352251229663,0.8316307226636398,0.7578509269769201,0.8191449110858873,0.6969353007945517,0.8248202799848657]),3),
        "3-mer" : np.round(np.array([0.903140370790768,0.9027620128641695,0.8486568293605751,0.8823306848278472,0.8221978204474784,0.896329928111994]),3),
        "4-mer" : np.round(np.array([0.9171396140749148,0.914491108588725,0.8864926220204313,0.8827090427544457,7325009458948165,0.9008702232311767]),3),
        "5-mer" : np.round(np.array([0.9277336360196746,0.8838441165342414,0.9107075293227394,0.8853575482406356,0.6954218690881574,0.9103291713961408]),3),
        }

df = pd.DataFrame(dict,index = ["svc","knn","LogisticregRession","RandomForest","DecisionTree","GBDT"])

# print (df)
# 设置绘图风格
plt.style.use('ggplot')
sns.set_style('whitegrid')
# 设置画板尺寸
plt.subplots(figsize = (30,20))

sns.heatmap(df, 
            cmap=sns.diverging_palette(20, 220, n=200), 
        #     mask = mask, # 数据显示在mask为False的单元格中
            annot=True, # 注入数据
            center = 0,  # 绘制有色数据时将色彩映射居中的值
           )
# Give title. 
plt.title("Heatmap of all the Features", fontsize = 30)
plt.show()