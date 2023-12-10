import pandas as pd


path = './data/processed/'

df_data_processed = pd.read_csv(path+'Data_Processed.csv')
df_codigo_motivo_consulta = pd.read_csv(path+'codigo_motivo_consulta.csv')
df_codigo_motivo_consulta = df_codigo_motivo_consulta.drop('Unnamed: 0', axis=1)
df_merged = pd.merge(df_data_processed, df_codigo_motivo_consulta, on='Numero episodio', how='inner')
df_merged = df_merged.drop(['Planificación suicida reciente', 'Fecha de Nacimiento', 'Antecedentes psiquiátricos', 'Antecedentes somáticos'], 
                           axis=1)
df_merged['Historia familiar de intentos de suicidio'] = df_merged['Historia familiar de intentos de suicidio'].fillna('No')
df_merged['Ideación suicida reciente'] = df_merged['Ideación suicida reciente'].fillna('No')

df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].fillna('0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('-', '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('.', '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('?', '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("'", '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("np", '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('no', '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("NO", '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("No", '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("0.", '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('.0', '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("o", '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("+", '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("a", '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('h', '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("no contesta", '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('..', '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Ninguno.', '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Ninguno', '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('indeterminado', '0')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('1 en total', '1')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('1 gesto suicida a los 35 años', '1')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('1 intennto suicidio + varios gestos parasuicidas previos', '1')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('1 gesto', '1')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('1 + 1 sobreingesta dudosamente suicida', '1')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Un gesto parasuicida  a lo largo de la vida', '1')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace(">2", '2')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('2 o 3', '2')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('2-3', '2')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('dos conocidos', '2')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('2 intennto suicidio + varios gestos parasuicidas previos', '2')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('2 intento suicidio + varios gestos parasuicidas previos', '2')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("3-4", '3')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace(">3", '3')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('varios', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('varias', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Varios', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('varios gestos parasuicidas', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("Varios gestos autolesivos.", '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('varias sobreingestas medicamentosas.', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Varios (no precisa número)', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Varios (no precisa numero)', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Varias (no precisa numero)', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('varias (no precisa numero)', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('En juventud, varios episodios', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Múltiples.', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('multiples', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Multiples', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Múltiples', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('si', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Si', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('gestos parasuicidas', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Gestos autolesivos previos', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('4 aproximadamente', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('4 y gestos autolesivos', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('4-5', '4')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('mas de cinco', '5')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Más de cinco', '5')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('Más de 5', '5')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('5-6', '5')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('5 referidos por el paciente', '5')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace(">5", '5')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('5-8', '5')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("5-10", '5')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('mas de 5', '5')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('6-7 aproximadamente', '6')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('más de 6', '6')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('7 parasuicidios', '7')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("+10", '10')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('+ de 10', '10')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('>10', '10')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('VARIOS (10 O MAS)', '10')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace("Aproximadamente 20", '20')
df_merged['Número de intentos de suicidio previos'] = df_merged['Número de intentos de suicidio previos'].replace('20, útima hace más de 20 años', '20')
df_merged['Número de intentos de suicidio previos'] = pd.to_numeric(df_merged['Número de intentos de suicidio previos'], errors='coerce')

mapping = {'evaluación_revisión': 0, 'farmacoología_tratamiento': 1, 'alteración_sueño_alimentación': 2, 'transtorno_anímico': 3} 
mapping_binario = {'Si': 1, 'No': 0} 
mapping_true_false = {True: 1, False: 0} 
mapping_sexo = {'M': 0, 'H': 1} 

df_merged['codigo_motivo_consulta'] = df_merged['codigo_motivo_consulta'].map(mapping)
df_merged['Sexo'] = df_merged['Sexo'].map(mapping_sexo)
df_merged['Historia familiar de intentos de suicidio'] = df_merged['Historia familiar de intentos de suicidio'].map(mapping_binario)
df_merged['Ideación suicida reciente'] = df_merged['Ideación suicida reciente'].map(mapping_binario)
df_merged['urgencias'] = df_merged['urgencias'].map(mapping_true_false)
df_merged.set_index('Numero episodio', inplace=True)
df_merged.to_csv('./data/prepared_datasets/df_metadata.csv')

df_ejeI = pd.read_csv(path + 'Eje1.csv')
df_ejeI_labels = pd.read_csv(path + 'ejeI.csv')
df_ejeI_labels = df_ejeI_labels.drop('Unnamed: 0', axis=1)
df_ejeI_merged = pd.merge(df_ejeI, df_ejeI_labels, on='Numero episodio', how='inner')

mapping_ejeI = {'personalidad': 0, 'sustancias': 1, 'bipolaridad': 2,
                        'conducta': 3, 'paróxisticos': 4, 'depresión': 5,
                        'ánimo': 6, 'adaptación': 7, 'delirante': 8,
                        'tdah': 9, 'sueño_estrés': 10, 'maníacos': 11,
                        'psicósis': 12, 'obsesivo': 13, 'esquizofrenia': 14,
                        'humor': 15}

df_ejeI_merged['F'] = df_ejeI_merged['F'].fillna(False).map(mapping_true_false)
df_ejeI_merged['T'] = df_ejeI_merged['T'].fillna(False).map(mapping_true_false)
df_ejeI_merged['G'] = df_ejeI_merged['G'].fillna(False).map(mapping_true_false)
df_ejeI_merged['R'] = df_ejeI_merged['R'].fillna(False).map(mapping_true_false)
df_ejeI_merged['Z'] = df_ejeI_merged['Z'].fillna(False).map(mapping_true_false)
df_ejeI_merged['E'] = df_ejeI_merged['E'].fillna(False).map(mapping_true_false)
df_ejeI_merged['M'] = df_ejeI_merged['M'].fillna(False).map(mapping_true_false)
df_ejeI_merged['N'] = df_ejeI_merged['N'].fillna(False).map(mapping_true_false)
df_ejeI_merged['I'] = df_ejeI_merged['I'].fillna(False).map(mapping_true_false)
df_ejeI_merged['A'] = df_ejeI_merged['A'].fillna(False).map(mapping_true_false)
df_ejeI_merged['ejeI'] = df_ejeI_merged['ejeI'].map(mapping_ejeI)
df_ejeI_merged.set_index('Numero episodio', inplace=True)
df_ejeI_merged.to_csv('./data/prepared_datasets/df_ejeI.csv')

df_ejeII = pd.read_csv(path + 'Eje2.csv')
df_ejeII['F'] = df_ejeII['F'].fillna(False).map(mapping_true_false)
df_ejeII.set_index('Numero episodio', inplace=True)

df_ejeII.to_csv('./data/prepared_datasets/df_ejeII.csv')
a = 0
