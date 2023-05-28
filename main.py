import os
import PyPDF2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
import math

# Definir a lista de stopwords em português e pontuações que serão removidas dos documentos
stopwords = set(stopwords.words('portuguese'))
stopwords.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '-', '—', '…'])

# Pré-processa os documentos
def pre_processamento(text):
   # Tokenização
   tokens = word_tokenize(text.lower())

   # Remoção de stopwords
   tokens = [token for token in tokens if token not in stopwords]

   return tokens

# Gera o índice invertido dos documentos
def indice_invertido(docs_directory):
   indice = defaultdict(lambda: {'total_freq': 0, 'doc_freq': defaultdict(int)})

   # Percorre todos os arquivos PDF do diretório e os enumera (para o doc_id)
   for doc_id, filename in enumerate(sorted(os.listdir(docs_directory)), start=1):
      if filename.endswith('.pdf'):
         file_path = os.path.join(docs_directory, filename)

         # Lê o conteúdo do PDF
         with open(file_path, 'rb') as file:
               pdf_reader = PyPDF2.PdfReader(file)
               doc_content = ''
               for page in pdf_reader.pages:
                  doc_content += page.extract_text()

               # Pré-processa o conteúdo do documento
               tokens = pre_processamento(doc_content)

               # Atualiza o índice invertido
               for token in tokens:
                  indice[token]['total_freq'] += 1
                  indice[token]['doc_freq'][doc_id] += 1

   return indice

# Realiza os cálculos BM-25 OKAPI
def bm25(consulta, indice):
   rsvs = defaultdict(float)
   N = len(indice)
   # Calcula a média de tamanho dos documentos
   lave = sum([indice[token]['total_freq'] for token in indice]) / N
   k1 = 2.0 # Valor de K1 do enunciado
   b = 0.75 # Valor de B do enunciado

   # Pré-processa a consulta
   consulta_tokens = pre_processamento(consulta)

   # Calcula o rsv para cada documento relevante
   for token in consulta_tokens:
      if token in indice:
         n = len(indice[token]['doc_freq'])
         # Calcula o IDF
         idf = math.log(N/n)
         for doc_id, doc_freq in indice[token]['doc_freq'].items():
               tfd = doc_freq
               ld = indice[token]['total_freq']
               # Calcula o RSV
               rsv = idf * ((tfd * (k1 + 1)) / (k1 * (1 - b + b * (ld / lave))+ tfd))
               # Soma o RSV
               rsvs[doc_id] += rsv

   return rsvs

# Realiza o ranqueamento dos documentos
def ranqueamento(consulta, indice):
   rsvs = bm25(consulta, indice)

   # Ordena os documentos pelo rsv
   docs_rank = sorted(rsvs.items(), key=lambda x: x[1], reverse=True)

   return docs_rank

# Salva o índice invertido em um arquivo .txt
def salva_indice(indice, output_file):
   with open(output_file, 'w') as file:
      for term, term_info in indice.items():
         total_freq = term_info['total_freq']
         doc_freq = term_info['doc_freq']
         line = f"{term}, {total_freq}, "
         doc_freq_str = ', '.join([f"({doc_id}, {freq})" for doc_id, freq in doc_freq.items()])
         line += doc_freq_str + "\n"
         file.write(line)

# Diretório dos documentos
docs_directory = 'Docs'

# Constrói o índice invertido
indice = indice_invertido(docs_directory)

# Chama a função de salvar o índice invertido
salva_indice(indice, 'indice_invertido.txt')

# Solicita a consulta ao usuário
consulta = input("Digite sua consulta: ")

# Chama a funãção do ranqueamento
docs_rank = ranqueamento(consulta, indice)

# Verifica se há resultados
if len(docs_rank) > 0:
    # Exibe os documentos ranqueados
    for rank, (doc_id, rsv) in enumerate(docs_rank, start=1):
        print(f"🏅 Rank {rank} - Documento: {doc_id} - RSV: {rsv}")
else:
    print("Sem resultados")
