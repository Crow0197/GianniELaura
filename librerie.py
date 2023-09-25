## LIBRERIE #########################################################################################################################################

import re                                               # Questa libreria è utilizzata per lavorare con espressioni regolari. 
import pandas as pd                                     # Libreria per l'analisi dei dati. Fornisce strutture dati flessibili, come i DataFrame.
import numpy as np                                      # Calcolo scientifico. Fornisce strutture dati come gli array multidimensionali (ndarray).
import random
import scipy.stats as stats                             # SciPy è una libreria che si basa su NumPy. scipy.stats fornisce funzioni per la statistica 
                                                        # e le distribuzioni di probabilità.
from scipy.stats import sigmaclip, lognorm              # 'sigmaclip' esegue il clipping dei dati basandosi sulla deviazione standard, e 'lognorm' 
                                                        # rappresenta una distribuzione log-normale, utile per il fitting statistico di dati.
from scipy.special import erf                           # Funzione di errore, utilizzata in matematica e fisica, e spesso anche per modellare i dati.
from scipy.optimize import curve_fit                    # Questa funzione è utilizzata per adattare una curva a un set di dati sperimentali.
                                                        # È comunemente utilizzata per il fitting di curve non lineari ai dati.
import scipy.signal as signal                           # Fornisce strumenti per il trattamento dei segnali (filtri, trasformate).

from sklearn.model_selection import train_test_split    # Scikit-Learn è una libreria di apprendimento. Fornisce una vasta gamma di algoritmi di 
                                                        # apprendimento automatico e strumenti per la selezione delle feature.
from sklearn.preprocessing import StandardScaler        # Parte di Scikit-Learn.
from sklearn.metrics import classification_report       # Fornisce metriche di valutazione delle prestazioni per i modelli di classificazione.
import matplotlib.pyplot as plt                         # Libreria è utilizzata per la creazione di grafici e visualizzazioni dei dati.
import colorama
from colorama import Fore, Style

# TensorFlow è una libreria di apprendimento automatico e deep learning. Queste importazioni includono layer, modelli e altri componenti utili per costruire reti neurali.
# 'Input' è utilizzato per definire l'input del modello, 'BatchNormalization' applica la normalizzazione ai dati in batch,
# 'Bidirectional' implementa un layer LSTM bidirezionale, 'LSTM' è un tipo di layer LSTM, 'Attention' rappresenta un layer di attenzione,
# 'Dropout' implementa il dropout per prevenire l'overfitting, e 'Dense' è utilizzato per creare un layer completamente connesso.

import tensorflow as tf                          
from tensorflow.keras.layers import Input, BatchNormalization, Bidirectional, LSTM, Attention, Dropout, Dense
from tensorflow.keras.models import Model, Sequential
from keras.utils import to_categorical
#####################################################################################################################################################   