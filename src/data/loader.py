import pandas as pd
import os

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file at {path} does not exist.")
    
    data = pd.read_csv(path)
    return data

if __name__ == "__main__":
    data = load_data("D:\\agentic AI\\MODELS\\ACM-SIG-AI-TELCO-CHURN\\data\\raw\\WA_Fn-UseC_-Telco-Customer-Churn.csv")
    #Telco Customer Churn dataset from Kaggle
    print(data.head(5))
    print("Number of rows:", data.shape[0])
    print("Number of columns:", data.shape[1])
    print('information about the dataset : \n')
    print(data.info())
    print(data.index)
