
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt



data1 = pd.read_excel(r"F:\Project\final\project data.xlsx")


# Checking for null files
data1.isnull().sum()

data1.dropna()

data1.columns

data1 = data1.drop(["Patient_Gender","Patient_ID","Patient_Age","Test_Booking_Date","Sample_Collection_Date",
                    "Cut_off_Schedule","Cut_off_time_HH_MM","Test_Booking_Time_HH_MM",
                    "Scheduled_Sample_Collection_Time_HH_MM","Agent_ID","Mode_Of_Transport"], axis = 1)


# Identify duplicates records in the data

duplicate = data1.duplicated()
duplicate
sum(duplicate)

# Removing Duplicates
data2 = data1.drop_duplicates()

# Visualization for correlation
sns.heatmap(data2.corr(),cbar=True,cmap='Blues')


# Separating numerical columns and categorical columns

num_columns = data2[['Agent_Location_KM', 'Time_Taken_To_Reach_Patient_MM', 'Time_For_Sample_Collection_MM',
                       'Lab_Location_KM', 'Time_Taken_To_Reach_Lab_MM']]

cat_columns = data2[['Test_Name', 'Sample',
                       'Way_Of_Storage_Of_Sample', 'Traffic_Conditions',
                       'Reached_On_Time']]

# Identifying the outliers

sns.boxplot(np.log(num_columns['Agent_Location_KM']))
sns.boxplot(np.log(num_columns['Time_Taken_To_Reach_Patient_MM']))
sns.boxplot(np.log(num_columns['Time_For_Sample_Collection_MM']))
sns.boxplot(np.log(num_columns['Lab_Location_KM']))
sns.boxplot(np.log(num_columns['Time_Taken_To_Reach_Lab_MM']))

# Treating the outliers

from feature_engine.outliers import Winsorizer

w = Winsorizer(capping_method='iqr', fold = 1.5, tail = 'both', 
              variables=['Agent_Location_KM', 'Time_Taken_To_Reach_Patient_MM', 'Time_For_Sample_Collection_MM',
                        'Lab_Location_KM', 'Time_Taken_To_Reach_Lab_MM'])

num = w.fit_transform(num_columns[['Agent_Location_KM', 'Time_Taken_To_Reach_Patient_MM', 'Time_For_Sample_Collection_MM',
                        'Lab_Location_KM', 'Time_Taken_To_Reach_Lab_MM']])

num.describe()

#converting into binary
lb = LabelEncoder()
cat_columns["Test_Name"] = lb.fit_transform(data2["Test_Name"])
cat_columns["Sample"] = lb.fit_transform(data2["Sample"])
cat_columns["Way_Of_Storage_Of_Sample"] = lb.fit_transform(data2["Way_Of_Storage_Of_Sample"])
cat_columns["Traffic_Conditions"] = lb.fit_transform(data2["Traffic_Conditions"])

cat_columns.info()


plt.figure(figsize=(15, 7))
plt.subplot(2,2,1)
sns.distplot(cat_columns['Test_Name'])

plt.figure(figsize=(15, 7))
plt.subplot(2,2,1)
sns.distplot(cat_columns['Sample'])

plt.figure(figsize=(15, 7))
plt.subplot(2,2,1)
sns.distplot(cat_columns['Way_Of_Storage_Of_Sample'])

plt.figure(figsize=(15, 7))
plt.subplot(2,2,1)
plt.hist(cat_columns['Traffic_Conditions'])


# Combining the numerical and categoricsl columns
data = pd.concat([num, cat_columns], axis = 1)


# rearranging the columns
data = data.iloc[: , [5,6,7,8,0,1,2,3,4,9]]

# apply normalization technique
##column = ['Test_Name','Sample','Way_Of_Storage_Of_Sample','Test_Booking_Time_HH_MM','Scheduled_Sample_Collection_Time_HH_MM','Cut_off Schedule','Cut_off_time_HH_MM','Agent_ID','Traffic_Conditions','Agent_Location_KM','Time_Taken_To_Reach_Patient_MM','Time_For_Sample_Collection_MM','Lab_Location_KM','Time_Taken_To_Reach_Lab_MM']
##data[column] = (data[column] - data[column].min()) / (data[column].max() - data[column].min())	
#Since there is not much difference in accuracy, there is no need to normalize the data

data.describe()

data['Reached_On_Time'].unique()
data['Reached_On_Time'].value_counts()


predictors = data.drop('Reached_On_Time', axis=1)
target = data['Reached_On_Time']

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.3, random_state=2)

# Adaboost algorithm

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.05, n_estimators = 500)

ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train))

final_df = pd.concat([predictors,target], axis=1)

input_data = (9,0,1,2,2,6,3,3,9)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = ada_clf.predict(input_data_reshaped)
print(prediction)

if prediction == 1:
    print("Reached on time")
else:
    print("Wil not reach on time")


import pickle
filename = "train.pkl"
pickle.dump(ada_clf, open(filename,"wb"))

filename1 = "final.pkl"
pickle.dump(data1, open(filename1,"wb"))



