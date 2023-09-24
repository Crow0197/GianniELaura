from librerie import *

s= '\n' + u'\u2727'
t= ' \u221A'
f= '\n       \u25BA\u25BA\u25BA\u25BA\u25BA '
f2= '      \u25C4\u25C4\u25C4\u25C4\u25C4 '
z1 = '\n\n'
z2 = '\n\n\n\n\n\n\n\n\n\n'

s0='\n\n \t\t\t\t\t  PROGETTO DI SISTEMI BIOMETRICI - Studenti: Amendola Laura, Lentini Giovanni.\n'
s1= s+' Applico la normalizzazione dei dati col fine di portare tutti i valori delle colonne "X" e "Y" nel range [0, 1] per uniformarli. Procedendo nella preparazione dei dati vengono eliminati i valori duplicati in base al valore del timestamp.: \n'
s2= 'Dataframe ripulito dai valori duplicati in base alla presenza dello stesso timestamp ' + t
s3= s+' Trovati i picchi nella velocità dei dati normalizzati, combina sia i picchi massimi che i picchi minimi in un unico DataFrame ed infine ordina il dataframe sampled_points in base alla colonna "TIME". \nEcco un grafico che permette la visualizzazione della velocità in funzione del tempo, evidenziando i picchi massimi e minimi nella velocità.'
s4= s+' In questa parte, vengono definiti i parametri per la preparazione delle sottosequenze dei dati, e stampati a video i risultati:\n'
s5= s+' Addestramento del modello utilizzando le sottosequenze precedentemente preparate. Le sottosequenze vengono suddivise in set di addestramento e test, poi viene creato un modello LSTM modificato per un output binario. Il modello viene addestrato e valutato, e alla fine vengono stampate alcune metriche e informazioni sui dati di input.\n'
s6= s+' Questo report fornisce una panoramica delle prestazioni del modello per ciascuna classe di classificazione, nonché una valutazione complessiva delle prestazioni del modello. \n'
s7= s+' Valutazione finale del modello di machine learning. Questi valori sono sono utili per valutare le prestazioni finali di un modello di machine learning dopo che è stato addestrato e per determinare quanto bene il modello generalizza su nuovi dati (i dati di test) che non sono stati utilizzati durante l''addestramento. \n'

p= u'\u2728 ' 
p1= '\n\n' + p + ' 1. PREPARAZIONE DEI DATI ' + p
p2= '\n\n\n' + p + ' 2. PICCHI MINIMI E MASSIMI ' + p
p3= '\n\n\n' + p + ' 3. PREPARAZIONE DELLE SOTTOSEQUENZE ' + p
p4= '\n\n\n' + p + ' 4. ADDESTRAMENTO DEL MODELLO ' + p
p5= '\n\n\n' + p + ' 5. REPORT DI CLASSIFICAZIONE ' + p
p6= '\n\n\n' + p + ' 6. PERDITA E ACCURATEZZA ' + p

m1= 'Numero di righe prima della rimozione dei duplicati:'
m2= 'Numero di righe dopo la rimozione dei duplicati:'
m3= 'Numero di sottoquenze:'
m4= 'Ampiezza delle sottoquenze:'
m5= '       Il programma è stato eseguito con successo. Puoi utilizzare il modello addestrato per fare previsioni su nuovi dati  '