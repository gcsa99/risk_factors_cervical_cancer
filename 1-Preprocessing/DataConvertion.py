import pandas as pd
#from sklearn.preprocessing import MinMaxScaler

def main():
  input_file = '0-Datasets/risk_factors_cervical_cancer.csv'
  output_file = '0-Datasets/risk_factors_cervical_cancer_clear.csv'
  names = ['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies','Smokes','Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives','Hormonal Contraceptives (years)','IUD','IUD (years)',
         'STDs','STDs (number)','STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease',
         'STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV','STDs: Number of diagnosis','STDs: Time since first diagnosis','STDs: Time since last diagnosis','Dx:Cancer',
         'Dx:CIN','Dx:HPV','Dx','Hinselmann','Schiller','Citology','Biopsy'] 
  features = ['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies','Smokes','Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives','Hormonal Contraceptives (years)','IUD','IUD (years)',
         'STDs','STDs (number)','STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease',
         'STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV','STDs: Number of diagnosis','Dx:Cancer',
         'Dx:CIN','Dx:HPV','Dx','Hinselmann','Schiller','Citology','Biopsy']
  
  df = pd.read_csv(  input_file,          # Nome do arquivo com dados
                     names = names,       # Nome das colunas 
                     usecols = features,  # Define as colunas que serão utilizadas
                     na_values='unknown')
  print(df.head(10))

  #identify all categorical variables
  cat_columns = df.select_dtypes(['object']).columns
  
  #convert all categorical variables to numeric
  df[cat_columns] = df[cat_columns].apply(lambda x: pd.factorize(x)[0])
  print(df.head(10))

  df.to_csv(output_file, header=False, index=False) 

if __name__ == "__main__":
  main()