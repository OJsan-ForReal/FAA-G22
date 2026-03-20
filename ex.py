import numpy as np
import pandas as pd
"""
            Insuficiente: exame Score 0 a 49
            Suficiente:   exame Score 50 a 69
            Bom:          exame Score 70 a 89
            Excelente:    exame Score 90 a 100

"""



df = pd.DataFrame({
    'exame_score': [50.16, 10.3, 9.9, 80.5, 90.0]
})
print(type(df['exame_score'][0]))
def categorizar_exame_score(score):
    if score < 50:
        return 'Insuficiente'
    elif 50 <= score < 70:
        return 'Suficiente'
    elif 70 <= score < 90:
        return 'Bom'
    else:
        return 'Excelente'
    
df["exame_score"] = df['exame_score'].apply(categorizar_exame_score)
print(df)
