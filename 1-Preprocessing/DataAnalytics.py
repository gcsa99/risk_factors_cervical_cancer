import pandas as pd
import numpy as np

#ok

def main():
    # Faz a leitura do arquivo
    input_file = '0-Datasets/risk_factors_cervical_cancer_clear.csv'
    #output_file = '0-Datasets/risk_factors_cervical_cancer_DataAnalytics.csv'
    names = ['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies','Smokes','Smokes (years)','Hormonal Contraceptives','Hormonal Contraceptives (years)','IUD','IUD (years)',
         'STDs','STDs (number)','STDs:HPV','STDs: Number of diagnosis','Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Biopsy']
    features = ['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies','Smokes','Smokes (years)','Hormonal Contraceptives','Hormonal Contraceptives (years)','IUD','IUD (years)',
         'STDs','STDs (number)','STDs:HPV','STDs: Number of diagnosis','Dx:Cancer','Dx:CIN','Dx:HPV','Dx','Biopsy']
    
    df = pd.read_csv(input_file,    # Nome do arquivo com dados
                    usecols = features,
                    names = names) # Nome das colunas

    #media 
    print('Média')
    print(df.mean())
    print('\n\n')

    #median
    print('Mediana')
    print(df.median())
    print('\n\n')

    #quatil
    """ print('Quantil')
    print(df.quantile())
    print('\n') """
    print('Quantil 25%')
    print(df.quantile(q=0.25))
    print('\n')
    print('Quantil 75%')
    print(df.quantile(q=0.75))
    print('\n\n')

    #moda
    print('Moda')
    print(df.mode())
    print('\n\n')


    #Medidas de dispersãp
    # Amplitude
    print('Amplitude')
    ampl = df.max() - df.min()
    print(ampl)
    print('\n\n')

    #Variância
    print('Variância')
    print(df.var())
    print('\n\n')

    #Desvio Padrão
    print('Desvio padrão')
    print(df.std())
    print('\n\n')

    #Desvio absoluto
    print('Desvio absoluto')
    print(df.mad())
    print('\n\n')

    #Covariância e Correlação
    print('Covariância')
    print(df.cov())
    print('\n')
    print('Correlação')
    print(df.corr())
    print('\n')

if __name__ == "__main__":
    main()