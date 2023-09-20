## MAIN ###############################################################################################################################################

from funzioni import *


##
## 1. Lettura di un file di dati .dta
##

if __name__ == "__main__":                          # Verifica se il file Python è eseguito direttamente come script principale o importato come 
                                                    # modulo in un altro script. In questo caso, il codice seguente verrà eseguito solo se il file 
                                                    # è eseguito come script principale.

    df=path_to_paser("dataframe/baixuerui.dta")     # Viene richiamata la funzione path_to_paser con il  percorso del file "baixuerui.dta" come 
                                                    # argomento e assegna il DataFrame risultante alla variabile df. La funzione legge il file e
                                                    # converte i dati in un dataframe.


##
## 2. Applico la funzione normalize per la preparazione dei dati  La normalizzazione è utile quando le scale dei dati in 
##    diverse colonne sono molto diverse o quando i dati devono essere portati in un intervallo specifico per l'elaborazione. 
##    Nel nostro caso, porteremo tutti i valori delle colonne "X" e "Y" nel range [0, 1] per uniformarli.
## 

x_data = df["X"]                            # Estrae la colonna di dati da normalizzare
y_data = df["Y"]

normalized_x = normalize(x_data)            # Normalizza i dati
normalized_y = normalize(y_data)

df["X"] = normalized_x                      # Sostituisci i dati nel DataFrame con quelli normalizzati
df["Y"] = normalized_y


##
## 3. Procedendo nella preparazione dei dati, vengono eliminati i valori duplicati in base al valore del timestamp.
## 

num_rows_before = len(df)                                      
print("Numero di righe prima della rimozione dei duplicati:",
       num_rows_before)

columns_to_check = ["TIME"]                                     
duplicates = df.duplicated(subset=columns_to_check, keep=False)
duplicate_rows = df[duplicates]
num_duplicates = len(duplicate_rows)
df = df.drop_duplicates(subset=columns_to_check, keep='first')

num_rows_after = len(df)
print("Numero di righe dopo la rimozione dei duplicati:", num_rows_after)
print("Dataframe ripulito dai valori duplicati in base alla presenza dello stesso timestamp")



##
## 4. Trova i picchi nella velocità dei dati normalizzati
##

peaks_max, peaks_min = find_peaks(df["VELOCITY"])   # Questa riga chiama la funzione find_peaks con la colonna "VELOCITY" del DataFrame df come
                                                    # argomento. La funzione find_peaks trova i massimi e i minimi locali nella colonna della 
                                                    # velocità e restituisce gli indici di questi picchi come peaks_max (massimi) e peaks_min (minimi).

sampled_points_max = df.iloc[peaks_max]             # Questa riga seleziona i punti corrispondenti ai picchi massimi dalla colonna "VELOCITY" del
                                                    # dataframe df utilizzando gli indici contenuti in peaks_max. Il risultato è memorizzato nel 
                                                    # dataframe sampled_points_max.

sampled_points_min = df.iloc[peaks_min]             # Stesso procedimento ma applicato ai picchi minimi dalla colonna "VELOCITY"


##
## 5. Creazione di un grafico che visualizza la velocità in funzione del tempo, evidenziando i picchi massimi e minimi nella velocità, e di un secondo
##    grafico che vada a zoommare su un intervallo di interesse.
##    È un modo efficace per visualizzare i dati e individuare i punti salienti all'interno di essi.
##

plt.figure(figsize=(12, 12))                                              # Crea una nuova figura per il grafico con le dimensioni specificate

# Primo grafico: tutti i valori
plt.subplot(2, 1, 1)                                                      # Creazione del primo subplot                                                                          
plt.plot(df["TIME"], df["VELOCITY"], label="Velocità", color='blue')      # Grafico a linea della velocità in funzione del tempo. I dati vengono 
                                                                          # prelevati dal DataFrame df e la colonna "TIME" è sull'asse x
                                                                          # mentre la colonna "VELOCITY" è sull'asse delle ordinate (y). Il parametro 
                                                                          # label specifica una etichetta per questa linea nel grafico, il parametro 
                                                                          # color imposta il colore della linea

# Queste due righe creano dei punti sul grafico per rappresentare i picchi massimi e minimi nella velocità. La funzione scatter viene utilizzata per i
# punti, e vengono specificate le coordinate x e y per ciascun punto. Nel caso dei picchi massimi, sampled_points_max["TIME"] rappresenta il tempo e 
# sampled_points_max["VELOCITY"] rappresenta la velocità dei picchi massimi. Il parametro c specifica il colore dei punti 
# e il parametro label fornisce etichette per i punti nel grafico.

plt.scatter(sampled_points_max["TIME"], sampled_points_max["VELOCITY"], c='red', label="Massimi")
plt.scatter(sampled_points_min["TIME"], sampled_points_min["VELOCITY"], c='yellow', label="Minimi")

plt.xlabel("Tempo")
plt.ylabel("Velocità")
plt.legend()
plt.title("Rilevamento di Picchi nella Velocità")

# Secondo grafico: intervallo delimitato da xlim e ylim
plt.subplot(2, 1, 2)                                                      # Creazione del secondo subplot
plt.plot(df["TIME"], df["VELOCITY"], label="Velocità", color='blue')
plt.scatter(sampled_points_max["TIME"], sampled_points_max["VELOCITY"], c='red', label="Massimi")
plt.scatter(sampled_points_min["TIME"], sampled_points_min["VELOCITY"], c='yellow', label="Minimi")
plt.xlabel("Tempo")
plt.ylabel("Velocità")
plt.legend()
plt.title("Focus sui valori compresi in un intervallo")

x_min = 2.20539e+07                                                        # Valori min e max dell'intervallo scelto
x_max = 2.206413e+07  
y_min = -17.9  
y_max = 32.8  

plt.xlim(x_min, x_max)                                                     # Imposta i limiti dell'asse x e y per il secondo grafico
plt.ylim(y_min, y_max)

plt.subplots_adjust(hspace=0.3)                                            # spaziamento tra i due subplot
plt.show()

##########################################################################################################################
##########################################################################################################################
##########################################################################################################################

# Calcolo delle Statistiche Iniziali: ovvero della media e della deviazione standard necessarie per il calcolo dei 
# parametri iniziali.

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



best_mse = float('inf')
best_num_gaussian = 0
initial_params = [1.0, initial_mu, initial_sigma]

for num_gaussians in range(2, len(peaks_max) + 1, 5):  # Incremento di 5
    print("Numero di gaussiane adatte al segnale", num_gaussians)
    
    initial_params = [1.0, initial_mu, initial_sigma]  # Reinizializza i parametri
    
    for i in range(num_gaussians):
        initial_params.extend([1.0, i * 3, 1.0])  # A, mu, sigma

    if len(initial_params) >= len(x):
        break

    try:
        params, covariance = curve_fit(gaussian, x, y, p0=initial_params, maxfev=500)  # Limite massimo di iterazioni
    except RuntimeError as e:
        continue

    reconstructed_signal = gaussian(x, *params)

    mse = np.mean((y - reconstructed_signal)**2)
    if mse < best_mse:
        best_mse = mse
        best_num_gaussian = num_gaussians

initial_params = [1.0, initial_mu, initial_sigma]

for i in range(best_num_gaussian):
    initial_params.extend([1.0, i * 3, 1.0])  # A, mu, sigma


# Esegui l'ottimizzazione dei parametri
params, covariance = curve_fit(gaussian, x, y, p0=initial_params)

reconstructed_signal = gaussian(x, *params)

mse = np.mean((y - reconstructed_signal)**2)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='Original Signal')
plt.plot(x, gaussian(x, *params), label='Fitted Gaussians', linestyle='--')
plt.plot(x, reconstructed_signal, label='Reconstructed Signal', linestyle=':')

for i in range(best_num_gaussian):
    plt.plot(x, gaussian(x, params[i * 3], params[i * 3 + 1], params[i * 3 + 2]), label=f'Gaussian {i + 1}')

plt.legend()
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Signal Decomposition and Reconstruction')
plt.show()