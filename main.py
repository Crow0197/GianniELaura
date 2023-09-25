from funzioni import *
from stringhe import *

##
## 1. Lettura di un file di dati .dta
##

if __name__ == "__main__":                          # Verifica se il file Python è eseguito direttamente come script principale o importato come 
                                                    # modulo in un altro script. In questo caso, il codice seguente verrà eseguito solo se il file 
                                                    # è eseguito come script principale.

    df=path_to_paser("dataframe/baixuerui.dta")     # Viene richiamata la funzione path_to_paser con il  percorso del file "baixuerui.dta" come 
                                                    # argomento e assegna il DataFrame risultante alla variabile df. La funzione legge il file e
                                                    # converte i dati in un dataframe
colorama.init(autoreset='true')                     # Inizializzazione libreria front-end
print(f"{Fore.MAGENTA}{Style.BRIGHT}"+s0)
print(f"{Fore.BLUE}{Style.BRIGHT}"+p1)
print(f"{Fore.CYAN}{Style.BRIGHT}"+s1)     

##
## 2. Applico la funzione normalize per la preparazione dei dati  La normalizzazione è utile quando le scale dei dati in 
##    diverse colonne sono molto diverse o quando i dati devono essere portati in un intervallo specifico per l'elaborazione. 
##    Nel nostro caso, porteremo tutti i valori delle colonne "X" e "Y" nel range [0, 1] per uniformarli.
## 

# Estrai le colonne "X", "Y" e "TIME" da df e convertele in un array NumPy
numeric_data = df[["X", "Y", "TIME"]].values

# Applica la normalizzazione alle colonne "X" e "Y"
normalized_x = normalize(numeric_data[:, 0])
normalized_y = normalize(numeric_data[:, 1])

numeric_data[:, 0] = normalized_x
numeric_data[:, 1] = normalized_y


##
## 3. Procedendo nella preparazione dei dati, vengono eliminati i valori duplicati in base al valore del timestamp.
## 

num_rows_before = len(df)               # Calcola il numero di righe nel DataFrame 'df' prima di effettuare alcuna operazione.                           
print(m1, num_rows_before)

columns_to_check = ["TIME"]             # Specifica le colonne da considerare per la ricerca di duplicati, in questo caso solo "TIME".                           
duplicates = df.duplicated(subset=columns_to_check, keep=False)     # Trova le righe duplicate basandosi sulle colonne specificate
                                                                    # e le tiene tutte (keep=False).
duplicate_rows = df[duplicates]         # Seleziona le righe duplicate in base alla condizione precedente.
num_duplicates = len(duplicate_rows)    # Calcola il numero di righe duplicate trovate.
df = df.drop_duplicates(subset=columns_to_check, keep='first')      # Elimina le righe duplicate mantenendo solo la prima occorrenza.

num_rows_after = len(df)                # Calcola il numero di righe nel DataFrame 'df' dopo aver rimosso i duplicati.
print(m2, num_rows_after)               
print(f"{Fore.GREEN}{Style.BRIGHT}"+f +s2)


##
## 4. Trova i picchi nella velocità dei dati normalizzati
##

print(f"{Fore.BLUE}{Style.BRIGHT}"+p2)
print(f"{Fore.CYAN}{Style.BRIGHT}"+s3)     

peaks_max, _ = find_peaks(df["VELOCITY"])               # Trova i picchi massimi nella velocità
peaks_min, _ = find_peaks(-df["VELOCITY"])              # Trova i picchi minimi nella velocità
sampled_points_max = df.iloc[peaks_max]                 # Estrai i punti corrispondenti ai picchi massimi dalla colonna "VELOCITY" del dataframe df
sampled_points_min = df.iloc[peaks_min]                 # Estrai i punti corrispondenti ai picchi minimi dalla colonna "VELOCITY" del dataframe df
sampled_points = pd.concat([sampled_points_max, sampled_points_min])     # Combina sia i picchi massimi che i picchi minimi in un unico DataFrame
sampled_points = sampled_points.sort_values(by="TIME")                   # ordina il dataframe 'sampled_points' in base alla colonna "TIME".


##
## 5. Creazione di un grafico che visualizza la velocità in funzione del tempo, evidenziando i picchi massimi e minimi nella velocità.
##    È un modo efficace per visualizzare i dati e individuare i punti salienti all'interno di essi.
##

plt.figure(figsize=(12, 6))                                           # Crea una nuova figura per il grafico con le dimensioni specificate
plt.plot(df["TIME"], df["VELOCITY"], label="Velocità", color='blue')  # Grafico a linea della velocità in funzione del tempo. I dati vengono
                                                                      # prelevati dal DataFrame df e la colonna "TIME" è sull'asse x
                                                                      # mentre la colonna "VELOCITY" è sull'asse delle ordinate (y). Il parametro
                                                                      # label specifica una etichetta per questa linea nel grafico, il parametro
                                                                      # color imposta il colore della linea

# Queste righe creano dei punti sul grafico per rappresentare i picchi massimi e minimi nella velocità. La funzione scatter viene utilizzata per i
# punti, e vengono specificate le coordinate x e y per ciascun punto. Nel caso dei picchi massimi, sampled_points_max["TIME"] rappresenta il tempo e
# sampled_points_max["VELOCITY"] rappresenta la velocità dei picchi massimi. Il parametro c specifica il colore dei punti
# e il parametro label fornisce etichette per i punti nel grafico.

plt.scatter(sampled_points_max["TIME"], sampled_points_max["VELOCITY"], c='red', label="Picchi Massimi")
plt.scatter(sampled_points_min["TIME"], sampled_points_min["VELOCITY"], c='yellow', label="Picchi Minimi")
plt.xlabel("Tempo")
plt.ylabel("Velocità")
plt.legend()
plt.title("Rilevamento di Picchi nella Velocità")
plt.show()



##
## 6. Preparazione delle sottosequenze. In questa parte, vengono definiti i parametri per la preparazione delle sottosequenze dei dati.
##    Viene utilizzata la funzione preprocessing per creare le sottosequenze dei dati in base ai parametri specificati, e poi vengono stampati
##    a schermo il numero di sottosequenze e l'ampiezza delle stesse.
##

print(f"{Fore.BLUE}{Style.BRIGHT}"+p3)
print(f"{Fore.CYAN}{Style.BRIGHT}"+s4)    
numero_sotto_sequenze = 10
ampiezza_sotto_sequenze = 50
# Esegue la funzione 'preprocessing' per preparare le sottosequenze dei dati.
sottosequenze = preprocessing(sampled_points[["X", "Y", "TIME"]], numero_sotto_sequenze, ampiezza_sotto_sequenze)
print(m3, numero_sotto_sequenze)
print(m4, ampiezza_sotto_sequenze)

##
## 7. Addestramento del modello utilizzando le sottosequenze precedentemente preparate. Le sottosequenze vengono suddivise in set di addestramento
##    e test, poi viene creato un modello LSTM modificato per un output binario. Il modello viene addestrato e valutato, e alla fine vengono stampate
##    alcune metriche e informazioni sui dati di input.
##

print(f"{Fore.BLUE}{Style.BRIGHT}"+p4)
print(f"{Fore.CYAN}{Style.BRIGHT}"+s5)    
X = sottosequenze
y = np.zeros(len(sottosequenze))                                                                # Etichetta tutte le sottosequenze come classe 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)       # Suddivide i dati in set di addestramento e test

units = 64           # Definizione dei parametri del modello
dropout = 0.2
input_shape = (X_train.shape[1], X_train.shape[2])

model = bilstm(units, input_shape=(X_train.shape[1], X_train.shape[2]), dropout=dropout)        # Modifica della definizione del modello per un
                                                                                                # output binario
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)  

model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))       # Addestra il modello sui dati di addestramento

y_pred = model.predict(X_test)                        # Valuta il modello sui dati di test
y_pred = (y_pred > 0.5).astype(int)                   # Converte le predizioni in valori binari (0 o 1)
print(f"{Fore.BLUE}{Style.BRIGHT}"+p5)
print(f"{Fore.CYAN}{Style.BRIGHT}"+s6)   
print(classification_report(y_test, y_pred, zero_division=1))

y_pred = model.predict(X_test)                        # Valuta il modello sui dati di test
y_pred = (y_pred > 0.5).astype(int)                   # Converte le predizioni in valori binari (0 o 1)

loss, accuracy = model.evaluate(X_test, y_test)       # Calcola la perdita e l'accuratezza finali


print(f"{Fore.BLUE}{Style.BRIGHT}"+p6)
print(f"{Fore.CYAN}{Style.BRIGHT}"+s7)    
print("Perdita finale:", loss)
print("Accuratezza finale:", accuracy)

print(f"{Fore.GREEN}{Style.BRIGHT}"+z1+f+m5+f2+z2)

##########################################################################################################################

##                                   S V I L U P P I                 F U T U R I                                        ##

##########################################################################################################################

# Campionamento alternativo dei punti: non basato sui picchi di min e max ma sulla sigma-log normale.

"""
##
## 1. Calcolo delle Statistiche Iniziali per l'applicazione della Sigma-Log Normale: calcolo della media e della 
##    deviazione standard necessarie per inizializzare la variabile initial_params.
##


df['VELOCITY'] = df['VELOCITY'].replace([np.inf, -np.inf], np.nan)      #Pulizia dei dati da valori INF e NULL
df = df.dropna(subset=['VELOCITY', 'TIME'])

x = df["TIME"]      
y = df["VELOCITY"]   

mean_x = np.mean(x)  # Calcola la media dei valori nella colonna "n_time" e la assegna a mean_x.
mean_y = np.mean(y)  # Calcola la media dei valori nella colonna "n_velocity" e la assegna a mean_y.

initial_mu = (mean_x + mean_y) / 2  # Calcola la media composta delle medie di "TIME" e "VELOCITY" e la assegna a initial_mu.
print("Media iniziale:", initial_mu)

std_x = np.std(x)  # Calcola la deviazione standard dei valori nella colonna "n_time" e la assegna a std_x.
std_y = np.std(y)  # Calcola la deviazione standard dei valori nella colonna "n_velocity" e la assegna a std_y.

initial_sigma = np.sqrt((std_x ** 2 + std_y ** 2) / 2)  # Calcola la deviazione standard composta delle deviazioni standard di "TIME"  
                                                        # e "VELOCITY" e ne calcola la radice quadrata. Il risultato è assegnato 
                                                        # a initial_sigma.
print("Deviazione Standard iniziale:", initial_sigma)



##
## 2. Decomposizione del segnale originale in una serie di funzioni gaussiane ottimizzando i parametri di queste gaussiane. 
##

best_mse = float('inf')                                 # Inizializza il miglior errore quadratico medio (MSE) a infinito
best_num_gaussian = 0                                   # Inizializza il numero migliore di gaussiane a 0
initial_params = [1.0, initial_mu, initial_sigma]       # Inizializza i parametri del modello gaussiano con una singola gaussiana

for num_gaussians in range(2, len(peaks_max) + 1, 5):   # Itera su un range di numero di gaussiane, incrementando di 5
    print("Numero di gaussiane adatte al segnale", num_gaussians)
    
    initial_params = [1.0, initial_mu, initial_sigma]   # Reinizializza i parametri per ogni numero di gaussiane
    
    for i in range(num_gaussians):
        initial_params.extend([1.0, i * 3, 1.0])        # Estendi la lista dei parametri con A, mu e sigma per ciascuna gaussiana

    if len(initial_params) >= len(x):                   # Verifica se il numero di parametri è maggiore o uguale al numero di punti dati
        break

    try:
        params, covariance = curve_fit(gaussian, x, y, p0=initial_params, maxfev=500)  # Esegui l'ottimizzazione dei parametri del modello gaussiano
    except RuntimeError as e:
        continue                                        # Ignora eccezioni di runtime e continua con la prossima iterazione

    reconstructed_signal = gaussian(x, *params)         # Ricostruisci il segnale utilizzando i parametri ottimizzati

    mse = np.mean((y - reconstructed_signal)**2)        # Calcola l'errore quadratico medio (MSE) tra il segnale originale e quello ricostruito
    if mse < best_mse:                                  # Se il MSE ottenuto è migliore del miglior MSE precedente
        best_mse = mse                                  # Aggiorna il miglior MSE
        best_num_gaussian = num_gaussians               # Aggiorna il numero migliore di gaussiane

initial_params = [1.0, initial_mu, initial_sigma]       # Reinizializza i parametri con il miglior numero di gaussiane trovato

for i in range(best_num_gaussian):
    initial_params.extend([1.0, i * 3, 1.0])            # Estendi la lista dei parametri con A, mu e sigma per ciascuna gaussiana

params, covariance = curve_fit(gaussian, x, y, p0=initial_params)   # Esegui l'ottimizzazione dei parametri finali

reconstructed_signal = gaussian(x, *params)                         # Ricostruisci il segnale utilizzando i parametri ottimizzati

mse = np.mean((y - reconstructed_signal)**2)                        # Calcola l'errore quadratico medio (MSE) tra il segnale originale e quello
                                                                    # ricostruito

plt.figure(figsize=(10, 6))                                         # Plot dei risultati
plt.plot(x, y, label='Segnale Originale')
plt.plot(x, gaussian(x, *params), label='Gaussiane Fittate', linestyle='--')
plt.plot(x, reconstructed_signal, label='Segnale Ricostruito', linestyle=':')

for i in range(best_num_gaussian):
    plt.plot(x, gaussian(x, params[i * 3], params[i * 3 + 1], params[i * 3 + 2]), label=f'Gaussiana {i + 1}')

plt.legend()
plt.xlabel('Tempo')
plt.ylabel('Ampiezza')
plt.title('Decomposizione e Ricostruzione del Segnale')
plt.show()
"""