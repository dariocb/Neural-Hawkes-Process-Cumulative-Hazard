# Loads the raw data, processes it and then saves the processed file.

import pandas as pd
import numpy as np
import datetime


def process_data() -> pd.DataFrame:
    df = pd.read_excel('./data/raw/TEXTOS_Consultas_Psiquiatría.xlsx')


    df = df[~(df.Sexo=='D')]
    df.loc[df.Sexo=='H', 'Sexo'] = 1
    df.loc[df.Sexo=='M', 'Sexo'] = 0

    df['idea_planif_suicida_reciente'] = None

    df.loc[((df['Ideación suicida reciente']==0) |
            (df['Ideación suicida reciente']==None)) & 
            (df['Planificación suicida reciente']==1), 'Ideación suicida reciente'] = 1 

    df.loc[(df['Ideación suicida reciente']==1) & (df['Planificación suicida reciente']==1), 'idea_planif_suicida_reciente'] = 2
    df.loc[(df['Ideación suicida reciente']==1) & (df['Planificación suicida reciente']==0), 'idea_planif_suicida_reciente'] = 1
    df.loc[(df['Ideación suicida reciente']==0) & (df['Planificación suicida reciente']==0), 'idea_planif_suicida_reciente'] = 0

    df.drop(['Ideación suicida reciente', 'Planificación suicida reciente'], axis=1, inplace=True)

        
    def fix_date(x):
        month, year = x.split('/')
        if year > '02':
            year = '19' + year
        else:
            year = '20' + year
        return int(year), int(month)

    df.sort_values('Fecha consulta', inplace=True)
    df['Fecha de Nacimiento'] = df['Fecha de Nacimiento'].apply(lambda x: datetime.date(*fix_date(x), 1) if x is not None else None)
    
    return df