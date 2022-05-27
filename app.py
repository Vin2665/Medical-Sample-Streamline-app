
import pickle
import streamlit as st
import numpy as np

dt = pickle.load(open('final.pkl','rb'))
ada_clf = pickle.load(open('train.pkl','rb'))

st.title("Medical Sample Collection Process Streamline")


#Test_Name
Test_Name = st.selectbox('Test_Name', dt["Test_Name"].unique())
if Test_Name == "Acute kidney profile":
    Test_Name = 0
elif Test_Name == "HbA1c":
    Test_Name = 5
elif Test_Name == "Vitamin D-25Hydroxy":
    Test_Name = 9
elif Test_Name == "TSH":
    Test_Name = 8
elif Test_Name == "Lipid Profile":
    Test_Name = 6
elif Test_Name == "Complete Urinalysis":
    Test_Name = 2
elif Test_Name == "RTPCR":
    Test_Name = 7
elif Test_Name == "H1N1":
    Test_Name = 4
elif Test_Name == "Fasting blood sugar":
    Test_Name = 3
else:
    Test_Name = 1

#Sample
Sample = st.radio('Sample', dt["Sample"].unique())
if Sample == "Blood":
    Sample = 0
elif Sample == "Urine":
    Sample = 2
else:
    Sample = 1

#Way_Of_Storage_Of_Sample
Way_Of_Storage_Of_Sample = st.radio('Way_Of_Storage_Of_Sample', dt["Way_Of_Storage_Of_Sample"].unique())
if Way_Of_Storage_Of_Sample == "Advanced":
    Way_Of_Storage_Of_Sample = 0
else:
    Way_Of_Storage_Of_Sample = 1


#Traffic_Conditions
Traffic_Conditions = st.radio('Traffic_Conditions', dt['Traffic_Conditions'].unique())
if Traffic_Conditions == "Low Traffic":
    Traffic_Conditions = 1
elif Traffic_Conditions == "Medium Traffic":
    Traffic_Conditions = 2
else:
    Traffic_Conditions = 0

#Agent_Location_KM
Agent_Location_KM = st.number_input('Agent_Location_KM')

#Time_Taken_To_Reach_Patient_MM
Time_Taken_To_Reach_Patient_MM = st.number_input('Time_Taken_To_Reach_Patient_MM')

#Time_For_Sample_Collection_MM
Time_For_Sample_Collection_MM = st.number_input('Time_For_Sample_Collection_MM')

#Lab_Location_KM
Lab_Location_KM = st.number_input('Lab_Location_KM')

#Time_Taken_To_Reach_Lab_MM
Time_Taken_To_Reach_Lab_MM = st.number_input('Time_Taken_To_Reach_Lab_MM')

if st.button('Predict Result'):

    query = np.array([Test_Name,Sample,Way_Of_Storage_Of_Sample,
                      Traffic_Conditions,Agent_Location_KM,
                      Time_Taken_To_Reach_Patient_MM,
                      Time_For_Sample_Collection_MM,Lab_Location_KM,
                      Time_Taken_To_Reach_Lab_MM])
    query = query.reshape(1,9)

    result = ada_clf.predict(query)

    if result == 'Y':
        st.header("Sample Reached On Time ?\n YES")
    else:
        st.header("Sample Reached On Time ?\n NO")