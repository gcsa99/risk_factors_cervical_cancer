import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

#ok

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/risk_factors_cervical_cancer_clear.csv'
    output_file = '0-Datasets/risk_factors_cervical_cancer_normalized_Biopsy.csv'
    names = ['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies','Smokes','Smokes (years)','Hormonal Contraceptives','Hormonal Contraceptives (years)','IUD','IUD (years)',
         'STDs','STDs (number)','STDs:HPV','STDs: Number of diagnosis','Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Biopsy']
    features = ['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies','Smokes','Smokes (years)','Hormonal Contraceptives','Hormonal Contraceptives (years)','IUD','IUD (years)',
         'STDs','STDs (number)','STDs:HPV','STDs: Number of diagnosis','Dx:Cancer','Dx:CIN','Dx:HPV','Dx']
    target = 'Biopsy'
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                     names = names) # Nome das colunas                      
    ShowInformationDataFrame(df,"Dataframe original")

    # Separating out the features
    x = df.loc[:, features].values
    
    # Separating out the target
    y = df.loc[:,[target]].values

    # Z-score normalization
    x_zcore = StandardScaler().fit_transform(x)
    normalized1Df = pd.DataFrame(data = x_zcore, columns = features)
    normalized1Df = pd.concat([normalized1Df, df[[target]]], axis = 1)
    ShowInformationDataFrame(normalized1Df,"Dataframe Z-Score Normalized")
    normalized1Df.to_csv(output_file, header=False, index=False)

    # # Mix-Max normalization
    # x_minmax = MinMaxScaler().fit_transform(x)
    # normalized2Df = pd.DataFrame(data = x_minmax, columns = features)
    # normalized2Df = pd.concat([normalized2Df, df[[target]]], axis = 1)
    # ShowInformationDataFrame(normalized2Df,"Dataframe Min-Max Normalized")

    

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")   

if __name__ == "__main__":
    main()