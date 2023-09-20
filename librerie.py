## LIBRERIE #########################################################################################################################################

import re                                               # Questa libreria è utilizzata per lavorare con espressioni regolari. 
import pandas as pd                                     # Libreria per l'analisi dei dati. Fornisce strutture dati flessibili, come i DataFrame.
import numpy as np                                      # Calcolo scientifico. Fornisce strutture dati come gli array multidimensionali (ndarray).
import scipy.stats as stats                             # SciPy è una libreria che si basa su NumPy. scipy.stats fornisce funzioni per la statistica 
                                                        # e le distribuzioni di probabilità.
from scipy.stats import sigmaclip, lognorm
from scipy.special import erf                           # Funzione di errore, utilizzata in matematica e fisica, e spesso anche per modellare i dati.
from scipy.optimize import curve_fit                    # Questa funzione è utilizzata per adattare una curva a un set di dati sperimentali.
                                                        # È comunemente utilizzata per il fitting di curve non lineari ai dati.
import scipy.signal as signal                           # Fornisce strumenti per il trattamento dei segnali (filtri, trasformate).
import tensorflow as tf                                 # Libreria di apprendimento automatico e deep learning sviluppata da Google. È utilizzata per
                                                        # creare, addestrare e valutare modelli di machine learning (modelli neurali profondi).
from sklearn.model_selection import train_test_split    # Scikit-Learn è una libreria di apprendimento. Fornisce una vasta gamma di algoritmi di 
                                                        # apprendimento automatico e strumenti per la selezione delle feature.
from sklearn.preprocessing import StandardScaler        # Parte di Scikit-Learn.
import matplotlib.pyplot as plt                         # Libreria è utilizzata per la creazione di grafici e visualizzazioni dei dati.

#####################################################################################################################################################