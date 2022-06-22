import pandas as pd
import statistics

def main():
    input_file = '0-Datasets/risk_factors_cervical_cancer_clear.csv'
    names = ['Age','Number of sexual partners','First sexual intercourse','Num of pregnancies','Smokes','Smokes (years)','Smokes (packs/year)','Hormonal Contraceptives','Hormonal Contraceptives (years)','IUD','IUD (years)',
         'STDs','STDs (number)','STDs:condylomatosis','STDs:cervical condylomatosis','STDs:vaginal condylomatosis','STDs:vulvo-perineal condylomatosis','STDs:syphilis','STDs:pelvic inflammatory disease',
         'STDs:genital herpes','STDs:molluscum contagiosum','STDs:AIDS','STDs:HIV','STDs:Hepatitis B','STDs:HPV','STDs: Number of diagnosis','STDs: Time since first diagnosis','STDs: Time since last diagnosis','Dx:Cancer',
         'Dx:CIN','Dx:HPV','Dx','Hinselmann','Schiller','Citology','Biopsy'] 

    df = pd.read_csv(input_file, names = names)  

    Age = df['Num of pregnancies'].tolist()
    sortedAge = sorted(df['Num of pregnancies'].tolist())

    meanAge = df['Num of pregnancies'].mean()
    modeAge = df['Num of pregnancies'].mode()
    medianAge = statistics.median(Age)
    avgPointAge = (sortedAge[0] + sortedAge[len(sortedAge)-1])/2
    weightedAverageAge = (df['Num of pregnancies'].sum())/len(Age)
    #geometricMeanAge = statistics.geometric_mean(Age)
    harmonicMeanAge = statistics.harmonic_mean(Age)

    print("Média de gravidezes de cada mulher de acordo com a idade:")
    print("Média = " + str(meanAge))
    print("Moda = " + str(modeAge[0]))
    print("Mediana = " + str(medianAge))
    print("Ponto Médio = " + str(avgPointAge))
    print("Média Ponderada = " + str(weightedAverageAge))
    #print("Média Geométrica = " + str(geometricMeanAge))
    print("Média Harmônica = " + str(harmonicMeanAge))

if __name__ == "__main__":
    main()