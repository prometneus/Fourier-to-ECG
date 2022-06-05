import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import scipy.signal
import sklearn
from numba import prange
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def data_importing():
    """
    Function importing data. A full link to file is to be provided.

    Parameters
    ----------

    Returns
    -------
    datafile: pd.DataFrame
    A dataframe obtained from reading csv. Is already checked for presence of necessary columns.
    """
    print('Enter path to a file:')
    datafile = pd.read_csv(str(input()))
    try:
        if 'data' in datafile.columns:
            print('ECG line is presented')
            datafile['data'] = datafile['data'].str.strip('{}')
        if 'duration' in datafile.columns or 'durarion' in datafile.columns:
          try:
            dur = datafile['duration'][0]
            print(f'Duration is {dur} seconds')
          except: 
            dur = datafile['durarion'][0]
            print(f'Duration is {dur} seconds')
    except:
        raise ValueError('Not all requested fields are presented. Please, reload the file')
    return datafile


def FourierTransform(signal, time, sample_rate=1000):
    """
    A function that performs Fourier Transform. Creates DataFrame for saving obtained results.

    Parameters
    ----------
    signal: array_like.
    ECG data.

    time: int.
    ECG signal recording time. Usually obtained from initial data.

    sample_rate: int. Default=1000.
    Sample rate of used digital ECG system. Usually obtained from initial data.

    Returns
    -------
    df: pandas DataFrame.
    A table with all necessary information and obtained from Fourier Transformation results.
    """
    N = sample_rate * time
    fourier = np.fft.rfft(a=signal, n=N)
    frequency = np.fft.rfftfreq(n=N, d=1 / sample_rate)

    return fourier, frequency

def get_amplitudes(fourier):
  for index, element in enumerate(fourier):
    fourier[index] = np.abs(element)
  return np.array([np.real(elem) for elem in fourier])

def fourierParse(row, use_complex = False): #функция проходит по столбцу со значениями Фурье, чтобы убрать из него все запятые и скобочки, а также преобразовать все комплексные числа  в их модули 
    # line = row.strip('[]') 
    values = []
    for elem in row.strip('[]') .split(','):
        elem = elem.strip('() ')
        m = abs(complex(elem))
        values.append(m) 
        
    return values #возвращаем лист 

 #создаем столбец с флагами наличия заболеваний
def create_dimask_column(y):
      i = 0
      for item in y:
        if y[i] > 0:
          y[i] = 1
          i += 1
        else:
          y[i] = 0
          i += 1
      return np.array(y)

def add_to_data_for_clustering(M, str_to_add):
  new_M_ = np.vstack([M, str_to_add])
  return new_M_

def clustering(M, KLASTER_COUNT, y):  # Функция кластеризации без добавления элемента
    reduced_data = PCA(n_components=2).fit_transform(M)
    kmean_model = KMeans(n_clusters=KLASTER_COUNT, init="k-means++", random_state=0)
    kmean_model.fit(reduced_data)

    print('Точность определения кластеров равна', accuracy_score(kmean_model.labels_, y) * 100, 'процентов')
    # plt.scatter(M[:,0], M[:,1], c=kmean_model.labels_, cmap='rainbow')
    centroids = kmean_model.cluster_centers_
    # cluster_0 = centroids[0] #кластер больных
    # cluster_1 = centroids[1] #кластер здоровых


def clust(M, KLASTER_COUNT):  # Функция повторной кластеризации с учетом добавления в систему новой дорожки
    reduced_data = PCA(n_components=2).fit_transform(M)
    kmean_model = KMeans(n_clusters=KLASTER_COUNT, init="k-means++", random_state=0)
    kmean_model.fit_predict(reduced_data)

    if kmean_model.labels_[-1] > 0:
        print(
            'В загруженной ЭКГ обнаружены некоторые отклонения. Рекомендуем обратиться к специалисту для более детального обследования.')
    else:
        print('Отклонений не обнаружено.')

def main():
    """
    Some kind of main function which calls everything else.

    Parameters
    ----------
    data: array_like.
    Data to process.

    Returns
    -------
    None.
    """
    data = data_importing()
    ecg = list()
    for string in data['data'][0].split(','):
        ecg.extend(float(s) for s in re.findall(r'[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?', string))

    fourier, frequency = FourierTransform(ecg, data['durarion'][0])
    frequency_cutted = frequency[np.where(frequency <= 40)]

    DI_MASK = 'DI_MASK'
    NONE_DISEASE = 'NONE_DISEASE'
    FOURIER_TRANSFORM = 'fourier_transform'

    dataset = pd.read_csv('/content/drive/MyDrive/Проект ЭКГ/Преобразование Фурье до 40Гц/Combined.csv')
    pd.set_option('display.max_columns', None)
    # вырезаю маркеры болезней, чтобы объединить все в одну группу - больные
    y1 = dataset.iloc[:, 3].values
    y2 = dataset.iloc[:, 4].values
    y3 = dataset.iloc[:, 5].values
    y = y1 + y2 + y3
    dataset = dataset.drop(['cid','pid','duration','disease_diabetes','disease_ibs',
                            'respiratory_disease','frequencies','filename'], axis=1)
    dimask_col = pd.DataFrame(create_dimask_column(y) , columns =['DI_MASK'])
    dataset.join(dimask_col)

    #cоздаем массив, в котором каждый элемент - это массив со значениями из столбика Фурье (их модулем, т.к. числа комплексные)
    mas = []
    for index, row in dataset.iterrows():
      F = fourierParse(row[FOURIER_TRANSFORM]) 
      mas.append([])
      k = 0
      # b = 
      for j in range(2401):
            mas[index].append(F[j])
            k += 1
    M = np.array(mas)
    clustering(M, 2, y)

    ECG = get_amplitudes(fourier[:len(frequency_cutted)])
    array_for_clustering = add_to_data_for_clustering(M, ECG[:2401])
    clust(array_for_clustering, 2)
