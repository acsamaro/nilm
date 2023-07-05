import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# apps=['kettle', 'dish washer', 'toaster']
apps=['dish washer', 'toaster']
# houses = ['test', 1,2,5]
houses = ['test', 1]
labels = ['recall', 'accuracy', 'precision', 'f1'] 
y = np.arange(len(labels))  # the label locations
height = 0.25

# deep learning
df = pd.read_csv('final_results_v3.csv', sep=',')
# df = pd.read_csv('final_results_ml.csv', sep=',')
df = df.round(decimals=4)

print(df)
# for app in apps:
#     print(app)
#     for house in houses:
#         if app in ['dish washer', 'toaster'] and house in [2,5]:
#             continue
#         ann_df = df[(df['tec'] == 'ANN') & (df['app']==app) & (df['house']== str(house))]
#         cnn_df = df[(df['tec'] == 'CNN') & (df['app']==app) & (df['house']==str(house))]
#         gru_df = df[(df['tec'] == 'GRU') & (df['app']==app) & (df['house']== str(house))]
        
#         fig, ax = plt.subplots(figsize=(12, 6))
#         bars1 = ax.barh(y+height, ann_df.iloc[0, [4,5,6,7]],height, label='ANN', color='#2596be')
#         bars2 = ax.barh(y, cnn_df.iloc[0, [4,5,6,7]],height, label='CNN', color='#76b5c5')
#         bars3 = ax.barh(y-height, gru_df.iloc[0, [4,5,6,7]],height, label='GRU', color='#063970')
        
#         # if house == 'test':
#         #     ax.set_title('Metrics from test')
#         # else:
#         #     ax.set_title('Metrics from house '+ str(house))
#         ax.set_yticks(y)
#         ax.set_yticklabels(labels)
#         ax.legend(loc='best', frameon=True, markerscale=2)

#         ax.bar_label(bars1, padding=-40, color='snow')
#         ax.bar_label(bars2, padding=-40, color='snow')
#         ax.bar_label(bars3, padding=-40, color='snow')

#         where_to_save = 'bar_plot_'+app+'_'+str(house)+'_dl.eps'
#         if app == 'dish washer':
#             where_to_save = 'bar_plot_dish_washer_'+str(house)+'_dl.eps'

#         plt.savefig(where_to_save, format='eps', dpi=1200)
#         plt.show()

#machine learning
height = 0.15
# df = pd.read_csv('final_results_ml.csv', sep=',')
for app in apps:
    print(app)
    for house in houses:
        lr_df = df[(df['tec'] == 'logisticRegression') & (df['app']==app) & (df['house']== str(house))]
        knn_df = df[(df['tec'] == 'KNN') & (df['app']==app) & (df['house']==str(house))]
        svm_df = df[(df['tec'] == 'SVM') & (df['app']==app) & (df['house']== str(house))]
        sgd_df = df[(df['tec'] == 'SGD') & (df['app']==app) & (df['house']== str(house))]
        rf_df = df[(df['tec'] == 'randomForest') & (df['app']==app) & (df['house']== str(house))]
        per_df = df[(df['tec'] == 'perceptron') & (df['app']==app) & (df['house']== str(house))]


        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.barh(y+2*height, lr_df.iloc[0, [4,5,6,7]],height, label='Logistic Regression', color='#2596be')
        bars2 = ax.barh(y+height, knn_df.iloc[0, [4,5,6,7]],height, label='KNN', color='#76b5c5')
        bars3 = ax.barh(y, svm_df.iloc[0, [4,5,6,7]],height, label='SVM', color='#063970')
        bars4 = ax.barh(y-height, sgd_df.iloc[0, [4,5,6,7]],height, label='SGD', color='#e28743')
        bars5 = ax.barh(y-2*height, rf_df.iloc[0, [4,5,6,7]],height, label='Random Forest', color='#eab676')
        bars6 = ax.barh(y-3*height, per_df.iloc[0, [4,5,6,7]],height, label='Perceptron', color='#873e23')

        # ax.set_xlabel('Metrics from house '+ str(house))
        # if house == 'test':
        #     ax.set_title('Métricas do teste')
        # else:
        #     ax.set_title('Métricas da casa '+ str(house))
        ax.set_yticks(y)
        ax.set_yticklabels(labels)

        ax.legend(loc='best', frameon=True, markerscale=2)
        ax.bar_label(bars1, padding=-40, color='snow')
        ax.bar_label(bars2, padding=-40, color='snow')
        ax.bar_label(bars3, padding=-40, color='snow')
        ax.bar_label(bars4, padding=-40, color='snow')
        ax.bar_label(bars5, padding=-40, color='snow')
        ax.bar_label(bars6, padding=-40, color='snow')


        # ax.bar_label(bars1, padding=10, color='black')
        # ax.bar_label(bars2, padding=10, color='black')
        # ax.bar_label(bars3, padding=10, color='black')
        # ax.bar_label(bars4, padding=10, color='black')
        # ax.bar_label(bars5, padding=10, color='black')
        # ax.bar_label(bars6, padding=10, color='black')


        where_to_save = 'bar_plot_'+app+'_'+str(house)+'_ml.eps'
        if app == 'dish washer':
            where_to_save = 'bar_plot_dish_washer_'+str(house)+'_ml.eps'
        plt.savefig(where_to_save, format='eps')
        plt.show()