import os
import re
import glob
import string
import itertools
import threading
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

import sys
import math
import time
import shutil
from io import StringIO
from shutil import copyfile
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter


import io 
import nltk
from nltk.corpus import stopwords  

'''
nltk.download('stopwords')
nltk.data.path.append('/root/nltk_data')
from nltk.corpus import stopwords
'''

path = os.getcwd()
stop_words = set(stopwords.words(path + '/portuguese'))

def convert_pdf_to_string(file_path):

	output_string = StringIO()
	with open(file_path, 'rb') as in_file:
	    parser = PDFParser(in_file)
	    doc = PDFDocument(parser)
	    rsrcmgr = PDFResourceManager()
	    device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
	    interpreter = PDFPageInterpreter(rsrcmgr, device)
	    for page in PDFPage.create_pages(doc):
	        interpreter.process_page(page)
	return(output_string.getvalue())
               
def convert_title_to_filename(title):
    filename = title.lower()
    filename = filename.replace(' ', '_')
    return filename

def split_to_title_and_pagenum(table_of_contents_entry):
    title_and_pagenum = table_of_contents_entry.strip()
    
    title = None
    pagenum = None
    
    if len(title_and_pagenum) > 0:
        if title_and_pagenum[-1].isdigit():
            i = -2
            while title_and_pagenum[i].isdigit():
                i -= 1

            title = title_and_pagenum[:i].strip()
            pagenum = int(title_and_pagenum[i:].strip())
        
    return title, pagenum

def calculate_wcss(data):
        wcss = []
        for n in range(2, 6):
            kmeans = KMeans(n_clusters=n)
            kmeans.fit(X=data)
            wcss.append(kmeans.inertia_)   
        return wcss

def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]
    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator/denominator)   
    return distances.index(max(distances)) + 2

def GerarCluster(path):

  def GerarConv(path):
     ListadePDF = []
     if not os.path.exists(path + 'Conv'):
       os.makedirs(path + 'Conv')

     for filename in os.listdir(path):
        if filename.endswith('.pdf'):
          nome = Path(filename).name
          nome = nome[:-4]
          texto = convert_pdf_to_string(path + filename)
          f = open(path + "Conv/" + nome + ".txt", "w",encoding="utf-8")
          f.write(texto)
          f.close()
        elif filename.endswith('.txt'):
          nome = Path(filename).name
          with open(path + nome , encoding = 'latin-1') as f:
             f.close()
          copyfile(path + filename, path + 'Conv/' + filename)
     return 
  
  def Textos2(path):
    for filename in os.listdir(path):
        if filename.endswith('.txt'):
           with open(os.path.join(path, filename), encoding = "ISO-8859-1") as f:
                nome = Path(filename).name
                titulo.append(nome)
                linha = f.read()
                lista.append(linha)
    return
  def Textos(path):
    if not os.path.exists(path[:-5] + 'Novo/'):
       os.makedirs(path[:-5] + 'Novo/')

    for filename in os.listdir(path):
        if filename.endswith('.txt'):
           with open(os.path.join(path, filename), encoding = "latin-1", errors = 'ignore') as f:
              nome = f.name
              line = f.read()
              words = line.split() 
              for r in words: 
                  if not r in stop_words: 
                       appendFile = open(path[:-5] + "Novo/" + filename  ,'w',encoding="utf-8")
                       appendFile.write(" "+r) 
                       appendFile.close() 
    return

  GerarConv(path)
  lista = []
  titulo = []
  Sum_of_squared_distances = []
  Textos(path + 'Conv/')
  Textos2(path + 'Novo/')
  vectorizer = TfidfVectorizer(stop_words={'portuguese'})
  X = vectorizer.fit_transform(lista)
  sum_of_squares = calculate_wcss(X)
  n = optimal_number_of_clusters(sum_of_squares)
  true_k = n
  model = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=50)
  model.fit(X)
  labels=model.labels_
  tabela=pd.DataFrame(list(zip(titulo,labels)),columns=['Título','cluster'])
  tabela=tabela.sort_values(by=['cluster'])
  
  import shutil
  dir_path = path + 'Novo/'
  dir_path2 = path + 'Conv/'
  try:
     shutil.rmtree(dir_path)
     shutil.rmtree(dir_path2)
  except OSError as e:
     print("Error: %s : %s" % (dir_path, e.strerror))
  return tabela


if __name__ == '__main__':
  path = os.getcwd()
  done = False
  
  def animate():
   for c in itertools.cycle(['|', '/', '-', '\\']):
      if done:
          break
      sys.stdout.write('\rCarregando ' + c)
      sys.stdout.flush()
      time.sleep(0.1)
   sys.stdout.write('\rTerminado!     ')
  
  def work(path):
     Tabela = GerarCluster(path + "/Textos/")
     Tabela_Dict = Tabela.to_dict('list')
     listas_tabela = (list(Tabela_Dict.values()))
     res = {}
     for key in listas_tabela[0]:
        for value in listas_tabela[1]:
           res[key] = value
           listas_tabela[1].remove(value)
           break

     now = datetime.now()
     dt_string = now.strftime("%d_%m_%Y_%H_%M/")
     clusters = Tabela['cluster'].nunique()

     if not os.path.exists(path + '/Análise_' + dt_string ):
       os.mkdir(path + '/Análise_' + dt_string )
      
     for x in range(clusters):
       if not os.path.exists(path + '/Análise_' + dt_string + 'Cluster_' + str(x)):
         os.makedirs(path + '/Análise_' + dt_string + 'Cluster_' + str(x))
     for value in res:
       try:
         copyfile(path + '/Textos/' + value, path + '/Análise_' + dt_string + 'Cluster_' + str(res[value]) + '/' + value)
       except Exception:
         copyfile(path + '/Textos/' + value[:-4] + '.pdf', path + '/Análise_' + dt_string + 'Cluster_' + str(res[value]) +'/' + value[:-4] + '.pdf')
         
     DIR = (path + '/Análise_' + dt_string)
     os.chdir(path + '/Análise_' + dt_string)
     for directory in os.listdir(DIR):
        numero = len([name for name in os.listdir(DIR + '/' + directory) if os.path.isfile(os.path.join(DIR + '/' + directory, name))])
        os.rename(directory,str(numero) + '_arquivos_' + str(directory))
              
     done = True
     return done
        
  load = threading.Thread(target=animate)
  load.start() 
  worker = threading.Thread(target=work)
  done = work(path)
