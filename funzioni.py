## FUNZIONI #########################################################################################################################################

from librerie import *


######  FUNZIONE FIND_PEAKS         
##
##  Questa funzione è progettata per trovare i massimi e i minimi locali in una sequenza di dati unidimensionale. È basata sulla libreria
##  scipy.signal, che fornisce strumenti per il trattamento dei segnali e l'analisi dei picchi nei dati.
##
##      INPUT:   - data: array NumPy unidimensionale contenente la sequenza di dati su cui si desidera trovare i picchi;
##      OUTPUT:  - peaks_min, peaks_max: indici dei picchi massimi e minimi

def find_peaks(data):

    peaks_max, _ = signal.find_peaks(data, prominence=1)   # Trova i massimi
    peaks_min, _ = signal.find_peaks(-data, prominence=1)  # Trova i minimi
    
    return peaks_max, peaks_min                            # Restituisce gli indici dei picchi massimi e minimi


######  FUNZIONE PATH_TO_PASER      
##
##  La funzione legge un file di testo specificato dal percorso (path) e converte i dati contenuti in quel file in un DataFrame di Pandas.
##  Il file di testo è un file di dati strutturato in un formato specifico, e la funzione estrae e elabora le informazioni in esso contenute.
##  Successivamente, calcola le grandezze derivate di distanza e velocità tra i punti.
##
##      INPUT:   - path: percorso del file specificato.
##      OUTPUT:  - df: dataFrame di Pandas (la funzione prende il file di dati strutturato dal percorso, estrae le informazioni significative 
##                     da esso e le organizza nel dataframe df).

def path_to_paser(path):   

    with open(path, "r") as f:          # Apre il file specificato dal percorso in modalità lettura
        righe = f.readlines()           # Legge tutte le righe del file e le memorizza in una lista di stringhe
    file_name = righe[0]                # Estrae il nome del file dalla prima riga
    number_points = righe[1]            # Estrae il numero di punti dalla seconda riga
    pattern = r'\d+'                    # Utilizza espressioni regolari per trovare tutti i numeri nella stringa 'number_points'

    numbers = re.findall(pattern, number_points)
    expected_numbers_of_points = [int(number) for number in numbers]      # Converte i numeri estratti in una lista di interi
    columns_name = righe[3]                                               # Estrae il nome delle colonne dalla quarta riga
    columns_name = re.split(r'\s+', columns_name.strip())                 # Suddivide la stringa 'columns_name' in una lista di nomi di colonne 
                                                                          # usando gli spazi come delimitatori
    columns_name = [word for word in columns_name if word]                # Rimuove eventuali elementi vuoti nella lista dei nomi delle colonne
    columns_name[2] = "ORIENTATION X"                                     # Modifica il nome di alcune colonne specifiche
    columns_name.insert(3, "ORIENTATION Y")
    points = []                                                           # Inizializza una lista vuota per memorizzare i punti
    
    for i in range(4, len(righe) - 1):                              # Itera sulle righe del file dal quinto all'ultima riga
        point = righe[i]
        pattern = r'\d+'                                            # Utilizza espressioni regolari per trovare tutti i numeri nella riga 'point'
        numbers = re.findall(pattern, point)
        extracted_numbers = [int(number) for number in numbers]     # Converte i numeri estratti in una lista di interi e aggiunge la lista dei punti
        points.append(extracted_numbers)
    
    df = pd.DataFrame(points, columns=columns_name)                       # Crea un DataFrame di Pandas utilizzando i punti e i nomi delle colonne
    df["DISTANCE"] = ((df["X"].diff())**2 + (df["Y"].diff())**2)**0.5     # Calcola la distanza tra due punti consecutivi usando le colonne "X" e "Y"
    df["TIME_DIFF"] = df["TIME"].diff()                                   # Calcola il tempo tra due punti consecutivi usando la colonna "TIME"
    df["VELOCITY"] = df["DISTANCE"] / df["TIME_DIFF"]                     # Calcola la velocità di ogni punto come la distanza divisa per il tempo

    return df                                                       


######  FUNZIONE NORMALIZE         
##
##  Questa funzione esegue la normalizzazione dei dati utilizzando il metodo Min-Max. La normalizzazione Min-Max è un processo che scala i
##  dati in modo che tutti i valori si trovino nell'intervallo compreso tra 0 e 1. Questo processo è utile quando si desidera portare i dati 
##  in uno specifico intervallo o quando si desidera ridurre la differenza di scala tra diverse colonne di dati. Questa funzione prende 
##  un array di dati e restituisce una versione normalizzata degli stessi, in cui tutti i valori si trovano nell'intervallo [0, 1]. Questo
##  è utile per garantire che i dati abbiano la stessa scala e per prepararli all'elaborazione o all'addestramento di modelli di machine 
##  learning.
##
##      INPUT:   - array: array contenente i dati (righe, 1);
##      OUTPUT:  - array: array dei dati normalizzati min-max.

def normalize(array):
    array = np.asarray(array)                           # Questa istruzione converte l'input array in un array NumPy, se non lo è già.
    min_val = np.min(array)                             # Calcola il valore minimo all'interno dell'array array utilizzando la funzione np.min.
                                                        # Questo valore minimo rappresenterà il minimo valore presente nei dati.
    max_val = np.max(array)                             # Calcola il valore massimo all'interno dell'array array utilizzando la funzione np.max.
                                                        # Questo valore massimo rappresenterà il massimo valore presente nei dati.
    normalized_array = (array - min_val) / (max_val - min_val)                           # Calcola la normalizzazione Min-Max
    return normalized_array



# Define the Gaussian function
def gaussian(x, *params):
    num_gaussians = len(params) // 3
    result = np.zeros_like(x)
    for i in range(num_gaussians):
        A = params[i * 3]
        mu = params[i * 3 + 1]
        sigma = params[i * 3 + 2]
        result += A * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return result


#def preprocessing(list_data, num_sub_seq, len_sub_seq):
    """
    :param list_data: lista con struttura (misure, n°timestamp)
    :param num_sub_seq: numero di sottosequenze da campionare
    :param len_sub_seq: lunghezza delle sottosequenze da campionare
    :return: lista di dati aveti subito il preprocessing
    :return: groundtruth modificato in base all'uso di concat o meno
    """

    numeric_data = np.asarray(list_data, dtype=np.float)
    numeric_data = numeric_data.T

    # Procediamo alla normalizzazione delle coordinate x ed y
    # Semplice normalizzazione min-max: si potrebbe usare anche quella di scikit-learn
    numeric_data[:, 0] = normalize(numeric_data[:, 0])
    numeric_data[:, 1] = normalize(numeric_data[:, 1])

    diff = numeric_data[1:, 0:2] - numeric_data[:-1, 0:2]

    product = numeric_data[1:, 3] * numeric_data[:-1, 3]

    numeric_data = np.ndarray((len(diff), 3))
    numeric_data[:, 0:2] = diff
    numeric_data[:, 2] = product

    numeric_data = rhs(numeric_data, num_sub_seq, len_sub_seq)
    
    list_data = numeric_data
    if len(list_data.shape) == 2:
        list_data = list_data[np.newaxis, :, :]

    return np.asarray(list_data)



######  FUNZIONE RHS        
##
##      L'algoritmo RHS genera sottosequenze casuali dai punti minimi e massimi e successivamente visualizza tali sottosequenze in un grafico.
##      INPUT:   - numeric_data: sequenza di dati rappresentata come un array NumPy dove otteniamo gli rhs dei dati;
##               - numero_sotto_sequenze: il numero di sottosequenze da estrarre;
##               - ampiezza_sotto_sequenze: ampiezza delle sottosequenze da estrarre. 
##      OUTPUT:  - list_sottosequenze: array NumPy contenente le sottosequenze estratte dalla sequenza di dati iniziale, dati sottoforma di rhs.

#def rhs(numeric_data, numero_sotto_sequenze, ampiezza_sotto_sequenze):

    lunghezza_sequenza = len(numeric_data)          # Calcola la lunghezza totale della sequenza di dati  
    list_sottosequenze = list()                     # Inizializza una lista vuota per memorizzare le sottosequenze estratte
    
    for i in range(numero_sotto_sequenze):          # Esegue un ciclo per estrarre il numero desiderato di sottosequenze
  
        starting_point = np.random.randint(0, lunghezza_sequenza - ampiezza_sotto_sequenze)   # Genera un punto di partenza casuale all'interno 
                                                                                              # della sequenza             
           
        list_sottosequenze.append(numeric_data[starting_point:starting_point + ampiezza_sotto_sequenze, :]) # Estrae la sottosequenza dalla sequenza
                                                                                                            # di dati ed aggiungi la sottosequenza 
                                                                                                            # alla lista delle sottosequenze            
    return np.array(list_sottosequenze)             # Restituisce le sottosequenze come un array NumPy