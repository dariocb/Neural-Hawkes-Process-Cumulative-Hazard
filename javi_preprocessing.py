from data_processing import process_data
import pandas as pd
import numpy as np

from datetime import datetime
import spacy
import re

import matplotlib.pyplot as plt

def drop_nullID(data):
    """
    Column: ID
    Removes the entries in which the id is null
    
    data: pandas dataframe
    """
    aux=np.where(data['ID']=='#NULL!')[0]
    data = data.drop(aux,axis=0) #Drop rows
    data = data.reset_index(drop=True)
    return data

def get_expanded_year(year):
    """"
    It receives a two digit number(string) representing the year and returns a 4 digit number (int)
    
    """
    if year>'23':
        year='19'+year
    else:
        year='20'+year
    return int(year)

def get_age(data): 
    """
    Computes the age of each patient at the date of the visit
    
    data: pandas dataframe 
    return: List wth the ages of each patient at the visit date
    
    """
    fecha_nacimiento = data.loc[:, 'Fecha de Nacimiento']
    fecha_consulta = data.loc[:, 'Fecha consulta']
    ages = []
    if len(fecha_consulta) == len(fecha_nacimiento):
        for i in fecha_nacimiento.index:
            # birthdate of the patient
            month = fecha_nacimiento[i].strftime('%m')
            year = fecha_nacimiento[i].strftime('%y')
            year = get_expanded_year(year)

            # Date of the visit
            consulta_month=fecha_consulta[i].strftime('%m')
            if len(consulta_month)<2: #if the month is represented with a single number (7) instead of 2 (07)
                consulta_month='0'+consulta_month
            consulta_year = fecha_consulta[i].strftime('%y')
            consulta_year = get_expanded_year(consulta_year)
            
            # Claculate age
            age=consulta_year-year
            if consulta_month<month:
                age-=1
            ages.append(age)

    else:
        print('Error: Different sizes')
    return ages

def remove_negative_ages(data,ages):
    """There are some records in which the datebirth is at least 2 years posterior to the date of the visit to the hospital, which make no sense.
    It is not related to the pregnancy because the difference in all them is morte than 2 years
    So the objective of this function is to remove those records
    
    data: Pandas dataframe to be modified
    ages: list wth the ages of each patient at the visit date
    
    """
    aux=np.where(pd.Series(ages)<0)[0] #61 registros con edades negativas
    fecha_nacimiento=data.loc[:,'Fecha de Nacimiento']
    i=fecha_nacimiento.index[aux] #take the index in our dataset
    data = data.drop(i,axis=0) #Drop rows
    
    return data

def lemmatize(text,nlp):
    if type(text)==str:
        doc =  nlp(text)
        tokens = []

        for token in doc:
            # Removal of special characters, punctuation and generic stop words
            if token.is_alpha and not token.is_stop and not token.like_num:
                # Lemmatization and lowercasing
                tokens.append(token.lemma_.lower())

        tokens = " ".join(tokens)
        return tokens

def remove_no(text):
    """
    Column: Antecedentes somáticos
    It removes all fields whose value is no
    
    text: string     
    """
    if type(text)!=float:
        paragraphs=text.split('\n') #list with each paragraph
        text2='' #In this string we will store all the sentences without a 'SI'
        for line in paragraphs:
            if len(line.split(':'))>1:
                value=line.split(':')[1].strip() #se coge el valor despues de los dos puntos y le quita espcios en blanco con strip
                if 'No' not in value:
                    text2+=line+' '
            else:
                text2+=line+' '     
        return text2     

def remove_code(text):
    """
    Column: EJEI Diagnóstico Codificado CIE10
    look for the code pattern(i.e F98.1-) and remove it from the text
    
    text: string with codes
    return string without codes
    """
    if type(text)!=float: #For patients without nan
        patron = re.compile(r'[FTGRZEMNIA]\d+\.\d+\-')
        texto_sin_coincidencias = patron.sub('', text)  
        return texto_sin_coincidencias
    return text
    
if __name__=='__main__':
    df = process_data()
    df = drop_nullID(df)
    ages = get_age(df)
    df = pd.concat([df, pd.DataFrame(ages, columns=['Age'])], axis=1)
    df = remove_negative_ages(df, ages)
    df.loc[:, 'Fecha consulta'] = pd.to_datetime(df.loc[:, 'Fecha consulta'], format='%d/%m/%Y')
   
    nlp = spacy.load('es_core_news_md')
    nlp.disable_pipe("parser")

    df.loc[:, 'Antecedentes somáticos'] = df.loc[:, 'Antecedentes somáticos'].apply(lambda text:lemmatize(remove_no(text), nlp))
    df.loc[:, 'EJEI Diagnóstico Codificado CIE10'] = df.loc[:,'EJEI Diagnóstico Codificado CIE10'].apply(lambda text: lemmatize(remove_code(text), nlp) )
    df = df[['ID', 'Fecha consulta', 'Age', 'Antecedentes somáticos', 'EJEI Diagnóstico Codificado CIE10']]
    a=0
