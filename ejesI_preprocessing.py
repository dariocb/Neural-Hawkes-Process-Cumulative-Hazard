from initial_preprocessing import process_data

from matplotlib import pyplot as plt
import pandas as pd
import spacy

from gensim import models
from gensim.models.phrases import Phrases
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel

import pyLDAvis.gensim as gensimvis
import pyLDAvis

nlp = spacy.load('es_core_news_md')
nlp.disable_pipe('parser')

def lemmatize_and_preprocess(text):
    '''
    Preprocess the text and lemmatizes it

    text -> text to lemmatize
    '''
    doc = nlp(text)
    preprocessed_text =  ' '.join([token.lemma_ for token in doc if not token.is_punct  and token.is_alpha and not token.is_stop and not token.like_num])
    return preprocessed_text.lower()

def get_corpus(df):
    '''
    Corpus calculation of a dataframe of texts

    df -> dataframe to calculate the corpus
    '''
    corpus = []
    for i in df.index:
        doc = df['EJEI_lemmas'][i]
        corpus.append(doc.split())
    
    # N-grams detection:
    phrase_model = Phrases(corpus, min_count=2, threshold=20)
    corpus = [el for el in phrase_model[corpus]]
    return corpus

def explore_stop_words(D):
    '''
    Function for prining most common tokens

    D -> Dictionary
    '''
    tfidf_model = TfidfModel(dictionary=D)

    # Create list of tuples with token id and ndocs
    token_ndocs = [(D[id], ndocs) for id, ndocs in tfidf_model.dfs.items()]

    # Create dataframe
    df_mostcommon = pd.DataFrame(token_ndocs, columns=['token', 'ndocs'])
    df_mostcommon = df_mostcommon.sort_values(by='ndocs', ascending=False)
    df_mostcommon = df_mostcommon[df_mostcommon['ndocs'] > 1].sort_values(by='ndocs', ascending=False)

    most_common = df_mostcommon.nlargest(10, 'ndocs')
    print(most_common.head(25))

def remove_stopwords(custom_stop_words):
    '''
    Function for removing a list of words considered stop words from the dictionary

    custom_stop_words -> List of handpicked stopwords
    '''
    stop_word_ids = [D.token2id[word] for word in custom_stop_words if word in D.token2id]
    D.filter_tokens(bad_ids=stop_word_ids)
    return D
    
def clean_corpus(D, corpus):
    '''
    Function for cleaning the corpus based on a postprocesed dictionary

    D -> Clean adn prepared dictionary 
    corpus -> Original corpus to clean
    '''
    corpus_clean = []
    for sent in corpus: 
        aux = [token for token in sent if token in D.token2id.keys()]
        corpus_clean.append(aux)
    
    return corpus_clean

def train_lda_models(D, corpus, corpus_bow):
    '''
    Function for training lda model with different number of topics and saving the best one

    D -> Dictionary 
    Corpus -> Original corpus
    Corpus_bow -> Vectorized corpus
    '''
    best_coherence = -1
    best_lda_model = None

    results_df = pd.DataFrame(columns=["n_topics", "coherence"])

    for n_topics in range(4, 51, 2):
        print('Training model {} / 50 -----------------------------------------------------------------------------------------'.format(str(n_topics)))
        ldag = LdaModel(corpus=corpus_bow, id2word=D, num_topics=n_topics, random_state=42)
        coherencemodel = CoherenceModel(ldag, texts=corpus, dictionary=D, coherence='c_v')
        coherence_score = coherencemodel.get_coherence()

        results = pd.DataFrame(data=[[n_topics, coherence_score]], columns=["n_topics", "coherence"])
        results_df = pd.concat([results_df, results])

        # keep best model updated
        if coherence_score > best_coherence:
            best_coherence = coherence_score
            best_lda_model = ldag
    

    model_coherences = results_df.set_index('n_topics')['coherence'].to_dict()
    model_coherences = {str(key): value for key, value in model_coherences.items()}

    max_index = max(model_coherences, key=model_coherences.get)
    plt.figure(figsize=(16, 6))
    plt.plot(list(model_coherences.keys()), list(model_coherences.values()),  marker='o', linestyle='-')
    plt.xlabel('Model')
    plt.xticks(rotation=90)
    plt.ylabel('Coherence')
    plt.title('Coherence of the different models')
    max_x = list(model_coherences.keys()).index(max_index)
    max_y = model_coherences[max_index]
    plt.scatter(max_x, max_y, color='red', zorder=5)
    ax = plt.gca()
    ax.xaxis.get_major_ticks()[list(model_coherences.keys()).index(max_index)].label1.set_color('red')


    # Mostrar el gráfico
    plt.show()
    return best_lda_model
 
def get_topic(lda_model, labels_df, texts):
  # get the principal topic of each of the documents
  topics = []
  topics_dominant = []
  for i, text in enumerate(texts):
    tokens = text.split()
    bow = lda_model.id2word.doc2bow(tokens)

    # get the topic distribution
    doc_topics = lda_model.get_document_topics(bow)

    # # get the maximum probability topic
    max_prob = 0.0
    dominant_topic = -1
    for topic, prob in doc_topics:
      if prob > max_prob:
        max_prob = prob
        dominant_topic = topic

        # map dominant topic with the label
        dominant_topic_label = labels_df.loc[labels_df['Topic'] == dominant_topic, 'Label'].iloc[0]

    topics_dominant.append(dominant_topic_label)

  return topics_dominant

if __name__=='__main__':
    # load the dataframe and filter the columns we need
    df = process_data()
    df = df[['Numero episodio', 'EJEI Diagnóstico Codificado CIE10', 'EJEI Diagnóstico']]

    # join both columns into one
    df.loc[:, 'EJEI'] = df['EJEI Diagnóstico Codificado CIE10'].fillna('') + df['EJEI Diagnóstico'].fillna('')

    # lemmatize texts and get corpus 
    df['EJEI_lemmas'] = df['EJEI'].apply(lemmatize_and_preprocess)
    corpus = get_corpus(df)

    # get the dictionary and filter the extremes and some stopwords and obtain the clean corpus
    D = Dictionary(corpus)
    D.filter_extremes(no_below=2, no_above=.5)
    custom_stopwords = ['transtorno', 'depresivo', 'ansiedad', 'adaptativo', 'episodio', 'esquizofrenia', 'mixto', 
                        'bipolar', 'trastorno', 'distimia', 'dependencia', 'personalidad' 'tr', 'alcohol', 'recurrente', 
                        'síntoma', 'paranoidir', 'remisión', 'iv_problemas', 'paciente', 'problemas_grupo', 'base', 
                        'primario_apoyo', 'eje', 'ii', 'i', 'diagnóstico', 'cie', 'comportamiento', 'tept', 'tipo', 't', 
                        'leve', 'fin', 'rasgo', 'limite', 'referencia', 'vs', 'conducata', 'c', 'cluster', 'estudio', 
                        'predominio', 'adaptación', 'reacción', 'dato_descompensación', 'normal', 'efecto_secundario', 
                        'diagnosticado', 'alteracion', 'psicofármaco', 'momento', 'actual', 'estimulant', 'b', 'ci', 
                        'tp', 'c', 'estabilizado', 'tlp', 'anómalo', 'ajuste_tratamiento', 'limie', 'sd_depenmdencia', 
                        'posiblemente_ep', 'predominio', 'leve', 'residual', 't', 'especificar', 'actualmente', 
                        'generalizado', 'estable', 'resuelto', 'fase', 'parcial', 'reacción', 'código_cie', 'actualmente', 
                        'vital_adverso', 'adaptación', 'probable', 'personalidad', 'tr', 'psicótico'] 
    D = remove_stopwords(custom_stopwords)
    corpus = clean_corpus(D, corpus)

    # Perform the text vectorizations BoW
    corpus_bow = [D.doc2bow(doc) for doc in corpus]

    # train the model and add labels
    topicModel = train_lda_models(D, corpus, corpus_bow)
    topicModel.save('./models/EJEI.gensim')
    # modelo_lda = models.LdaModel.load("modelo_lda.gensim")
    vis = pyLDAvis.gensim.prepare(topicModel, corpus_bow, D)
    pyLDAvis.save_html(vis, './plots/lda_visualization_EJEI.html')
     
    labels_df = pd.DataFrame({'Topic': [i for i in range(0, 16)], 'Label': ['bipolaridad', 'esquizofrenia', 'sueño_estrés',
                                                                           'sustancias', 'maníacos', 'conducta', 
                                                                           'personalidad', 'ánimo', 'delirante', 'obsesivo', 
                                                                           'paróxisticos', 'adaptación', 'humor', 'psicósis',
                                                                           'depresión', 'tdah']})

    # add topic to the dataframe 
    df["ejeI"] = get_topic(topicModel, labels_df, df['EJEI_lemmas'])
    df = df[['Numero episodio', 'ejeI']]
    df.to_csv('./data/processed/ejeI.csv')