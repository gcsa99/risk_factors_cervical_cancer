from math import ceil
from tracemalloc import stop
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def main():
    input_file = '0-Datasets/risk_factors_cervical_cancer_clear.csv'
    names = ['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies','Smokes','Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives','Hormonal Contraceptives (years)','IUD','IUD (years)',
         'STDs','STDs (number)','STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease',
         'STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV','STDs: Number of diagnosis','STDs: Time since first diagnosis','STDs: Time since last diagnosis','Dx:Cancer',
         'Dx:CIN','Dx:HPV','Dx','Hinselmann','Schiller','Citology','Biopsy'] 
    df = pd.read_csv(input_file, names = names)                    

    # Distribuição de Frequencia Idade #
    #Atributo idade 
    df_age = (df['Age'])
    array_age = df_age.tolist()
    #print(array_age)
    print(df_age)
    
    #Idade Mínima e Máxima 
    age_min = int(df.min()[['Age']])
    print(age_min)
    age_max = int(df.max()[['Age']])
    print(age_max)

    #Definir o número de classess
    number_classes =  6

    #Calcular a amplitude de classe
    range = ceil((age_max - age_min)/number_classes)
    print ("range de teste", range)

    #Definir os limites inferiores e superiores das classes
    frequencias = []
    valor = age_min
    while valor < age_max:
        frequencias.append('{} - {}'.format(round(valor,1),round(valor+range,1)))
        valor += range

    print('frequencias', frequencias)

    #Rotular os valores dos atributos de acordo com sua classe
    freq_abs = pd.cut(df_age, bins=[18,31,44,57,70,83]) # Discretização dos valores em k faixas, rotuladas pela lista criada anteriormente
    print("teste frenquencia abs", freq_abs)

    #quantidade de atributos idade que tem em cada classe
    qtd_atr = pd.value_counts(freq_abs) 
    print('Quantidade de atributos em cada classe\n', qtd_atr)

    #Histograma do atributo idade
    bin = []
    for number in frequencias:
        bin.append(int(number[0:3]))

    last_range = frequencias.pop()

    bin.append(int(last_range[5:7]))
 
    plt.xlabel("Idade")
    plt.ylabel("Distribuição da idade")
    plt.title("Histograma de Distribuição de idade")
    plt.xlim(35, 89)
    plt.xticks(bin)
    plt.hist(array_age, bins=bin, edgecolor='black')
    plt.savefig('0-Datasets/histogram.png', format='png')
    plt.show()

def ShowInformationDataFrame(df, message=""):
    print(message+"\n")
    print(df.info())
    print(df.describe())
    print(df.head(10))
    print("\n")
    
if __name__ == "__main__":
    main()