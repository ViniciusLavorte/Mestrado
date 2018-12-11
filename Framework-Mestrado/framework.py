from scipy.io import arff
from sklearn.cluster import KMeans
from Tkinter import *
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import time
import tkFileDialog as filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import os
import warnings

warnings.filterwarnings("ignore")


class Application:

    # Caso seja necessario um contador global ele ja esta aqui
    counter = 1
    global set_z1
    set_z1 = []
    global set_C
    set_C = []

    def __init__(self, master=None):

        # Colocando nome a janela principal
        master.wm_title("Menu Principal")
        # Criando titulo do programa
        label_title = Label(
            master, text='Framework Mestrado', font='Helvetica -25 bold')
        label_title.grid(row=0, column=1)
        # Escolhendo a tecnica de normalizacao
        label_normalization = Label(master, text='Escolha a tecnica de Normalizacao:',
                                    font='Helvetica -18 bold')
        label_normalization.grid(row=1, column=1)
        optionMenu_normalization = OptionMenu(
            master, selectedNormalization, *normalizationList)
        optionMenu_normalization.grid(row=2, column=1)
        # Botao load Arff
        button_arff = Button(
            master, text='Carregar arquivo .Arff', command=self.readArff, font='Helvetica -18 bold', foreground='white', background='black')
        button_arff.grid(row=3, column=1)
        # Escolha das tecnicas
        label_title = Label(master, text='Escolha a Tecnica de Clusterizacao:',
                            font='Helvetica -18 bold')
        label_title.grid(row=4, column=1)
        button_tecnica1 = Button(
            master, text='K-Means', command=self.num_cluster_kMeans, font='Helvetica -18 bold', foreground='white', background='black')
        button_tecnica1.grid(row=5, column=0)
        button_tecnica2 = Button(
            master, text='OPF', command=self.OPF, font='Helvetica -18 bold', foreground='white', background='black')
        button_tecnica2.grid(row=5, column=2)
        # Gerenciamento das Iteracoes
        label_title = Label(master, text='Pre Processamento',
                            font='Helvetica -18 bold')
        label_title.grid(row=6, column=1)
        button_iteracao = Button(
            master, text='Buscar Raizes', command=self.getRoots, font='Helvetica -18 bold', foreground='white', background='black')
        button_iteracao.grid(row=7, column=1)
        button_iteracao2 = Button(
            master, text='Selecionar Amostras de Fronteira', command=self.getBorder, font='Helvetica -18 bold', foreground='white', background='black')
        button_iteracao2.grid(row=8, column=1)
        # Escolha do classificador
        label_title2 = Label(master, text='Classificador',
                             font='Helvetica -18 bold')
        label_title2.grid(row=9, column=1)
        button_iteracao3 = Button(
            master, text='Treinar o Classificador com Z1', command=self.trainClassifier, font='Helvetica -18 bold', foreground='white', background='black')
        button_iteracao3.grid(row=10, column=1)
        button_iteracao4 = Button(
            master, text='Classificar Amostras de C', command=self.testClassifier, font='Helvetica -18 bold', foreground='white', background='black')
        button_iteracao4.grid(row=11, column=1)
        # Botao de sair
        button_quit = Button(master, text='Sair',
                             command=master.quit, font='Helvetica -15 bold', foreground='white', background='red')
        button_quit.grid(row=15, column=2)

    # Funcao para nova janela escolher o num de clusters
    def num_cluster_kMeans(self):
        global window1
        # Cria uma nova janela
        window1 = Toplevel()
        # Coloca titulo nessa Janela
        window1.wm_title("k-Means")
        label_cluster = Label(window1, text='Digite o numero de clusters',
                              font='Helvetica -18 bold')
        label_cluster.grid(row=0, column=0)
        global entry_cluster
        entry_cluster = Entry(window1)
        entry_cluster.grid(row=1, column=0)
        button_tecnica = Button(
            window1, text='OK', command=self.kMeans, font='Helvetica -18 bold', foreground='white', background='black')
        button_tecnica.grid(row=4, column=0)

    # Funcao de leitura do Arff
    def readArff(self):
        print('===') * 30
        print('')
        print('Abrindo arquivo')
        print('')
        # Obter o path do arquivo
        file = filedialog.askopenfile()
        t = time.time()
        # Carrega o .arff
        dados = arff.loadarff(file.name)
        # Transforma somente os dados em Data Frame
        global df
        df = pd.DataFrame(dados[0])
        # Obter o numero de features
        length = df.iloc[0, :].values
        print('Numero de Features: {}'.format(len(length) - 1))
        print('')
        # Separa a coluna de rotulos
        labels = df.iloc[:, len(length) - 1].values
        # Criar os index para gravar o numero das imagens
        index = pd.Series(np.arange(1, len(df.index) + 1))
        # Adicionar coluna de index ao dataframe
        df = df.assign(index=index.values)
        # Armazenar as caracteristicas sem a Classe na variavel X
        data_aux = df.iloc[:, 0:(len(length) - 1)].values
        # Escolha da tecnica de Normalizacao
        if selectedNormalization.get() == 'MinMaxScaler':
            scaler = MinMaxScaler()
        if selectedNormalization.get() == 'StandardScaler':
            scaler = StandardScaler()
        if selectedNormalization.get() == 'MaxAbsScaler':
            scaler = MaxAbsScaler()
        if selectedNormalization.get() == 'RobustScaler':
            scaler = RobustScaler()
        print('Tecnica de Normalizacao Escolhida: {}'.format(
            selectedNormalization.get()))
        print('')
        data_normalized = scaler.fit_transform(data_aux)
        # Adicionar o index e as clases aos dados normalizados
        aux1 = df.iloc[:, len(length)].values
        aux1 = np.vstack(aux1)
        aux2 = df.iloc[:, len(length) - 1].values
        aux2 = np.vstack(aux2)
        data_normalized = np.concatenate((data_normalized, aux2, aux1), axis=1)
        # Dividindo em conjunto e teste 80/20 mantendo sempre a mesma divisao
        # por meio do random state = 42 e dividindo as classes
        print('Numero total de amostras: {}'.format(len(labels)))
        # Definir a pasta que ira pegar o path das imagens
        global define_path
        if len(labels) < 5000:
            define_path = 1
        if len(labels) > 5000 & len(labels) < 25000:
            define_path = 2
        if len(labels) > 25000:
            define_path = 3
        print('')
        print('Dividindo em 80% treinamento e 20% teste')
        print('')
        global train
        train, test = train_test_split(
            data_normalized, test_size=0.2, random_state=0, stratify=data_normalized[:, len(length) - 1])
        # Contar o numero de amostras por classe
        print('Numero de amostras de Treinamento por Classe:')
        print('')
        aux = aux2[0]
        regula = True
        for i in aux2:
            if i == aux:
                if regula:
                    print('{}: {} '.format(i, np.count_nonzero(
                        train[:, len(length) - 1] == i)))
                    regula = False
            else:
                aux = i
                regula = True
        print('')
        print('Numero de amostras de Teste por Classe:')
        print('')
        aux = aux2[0]
        regula = True
        for i in aux2:
            if i == aux:
                if regula:
                    print('{}: {} '.format(i, np.count_nonzero(
                        test[:, len(length) - 1] == i)))
                    regula = False
            else:
                aux = i
                regula = True
        # Fazer backup das classes e index
        global train_bkp_class_index
        train_bkp_class_index = train[:, -2:]
        train = train[:, :-2]
        global test_bkp_class_index
        test_bkp_class_index = test[:, -2:]
        test = test[:, :-2]
        print('')
        print('Arquivo processado com sucesso! (Tempo de execucao: {})'.format(
            time.time() - t))
        print('')

    # Funcao do K-Means
    def kMeans(self):
        print('===') * 30
        print('')
        print('Inicializando K-Means')
        print('')
        t = time.time()
        global num_c
        num_c = entry_cluster.get()
        # Inicializar o KMeans com N centroides
        kmeans = KMeans(n_clusters=int(num_c), init='random')
        print('Numero de clusters: {}'.format(int(num_c)))
        print('')
        # Executar passando como parametro os dados
        kmeans.fit(train)
        # Verificar os centroides gerados pelo metodo
        centers = kmeans.cluster_centers_
        # Variavel distance recebe uma tabela de distancia de todas amostras para
        # os clusters
        distance = kmeans.fit_transform(train)
        # Transforma o NDARRAY em Array
        closests = distance.flatten()
        # Separa os clusters num vetor
        global clusters_distances
        clusters_distances = []
        for i in range(0, int(num_c)):
            clusters_distances.append(closests[i:][::int(num_c)])
        # Variavel labels recebe os clusters atribuidos a cada amostra
        rotulos = kmeans.labels_
        window1.destroy()
        print('K-Means aplicado com sucesso! (Tempo de execucao: {})'.format(
            time.time() - t))
        # print('')
        #print('Plotando grafico..')
        print('')
        # Funcao de Geracao dos Graficos
        #plt.scatter(train[:, 0], train[:, 1], s=10, c=rotulos)
        # plt.scatter(centers[:, 0], centers[:, 1],
        #            s=300, c='red', label='Centroids')
        # plt.title('Grafico')
        # plt.show()
        print('Pronto..')
        print('')

    # Funcao do OPF Cluster
    def OPF(self):
        global train
        global train_bkp_class_index
        # Loop para deixar todas classes = zero
        label = train_bkp_class_index[:, 0]
        x = 0
        for i in label:
            label[x] = 1
            x += 1
        # Recuperar o numero de amostras
        num_samples = len(train)
        # Recuperar o numero de classes e colocar todas como numero 1 para ser
        # aprendizado nao supervisionado
        aux2 = []
        for i in label:
            if i not in aux2:
                aux2.append(i)
        num_classes = len(aux2)
        # Recuperar o numero de features
        num_features = len(train[0])
        # Escrever num txt o cabecalho OPF
        cabecalho = str(num_samples) + ' ' + \
            str(num_classes) + ' ' + str(num_features)
        # Criar um array de 0 ate o num de amostras
        array = np.arange(0, num_samples, 1)
        array = np.vstack(array)
        # Concatenar o train com a ordenacao de amostras
        label = np.vstack(label)
        array = np.concatenate((array, label, train), axis=1)
        # Escrever num txt o dataframe em formato OPF
        print('===') * 30
        print('')
        print('Salvar TXT, converter em OPF e clusterizar')
        print('')
        np.savetxt('saida.txt', array, delimiter=' ',
                   fmt='%.10s', header=cabecalho, comments='')
        t = time.time()
        # Transformar o arquivo txt em OPF e clusterizar
        os.system('bash automation.sh')
        print('')
        print('OPF Cluster aplicado com sucesso! (Tempo de execucao: {})'.format(
            time.time() - t))
        print('')

    def getPath(self, c, op):
        if op == 1:
            if int(c) < 10:
                path = '/home/messias/Documentos/larvae/dataset/ALL/0000' + \
                    str(c) + '.png'
            else:
                if int(c) < 100:
                    path = '/home/messias/Documentos/larvae/dataset/ALL/000' + \
                        str(c) + '.png'
                else:
                    if int(c) < 1000:
                        path = '/home/messias/Documentos/larvae/dataset/ALL/00' + \
                            str(c) + '.png'
                    else:
                        path = '/home/messias/Documentos/larvae/dataset/ALL/0' + \
                            str(c) + '.png'
        if op == 2:
            if int(c) < 10:
                path = '/home/messias/Documentos/eggs/dataset/ALL/0000' + \
                    str(c) + '.png'
            else:
                if int(c) < 100:
                    path = '/home/messias/Documentos/eggs/dataset/ALL/000' + \
                        str(c) + '.png'
                else:
                    if int(c) < 1000:
                        path = '/home/messias/Documentos/eggs/dataset/ALL/00' + \
                            str(c) + '.png'
                    else:
                        if int(c) < 10000:
                            path = '/home/messias/Documentos/eggs/dataset/ALL/0' + \
                                str(c) + '.png'
                        else:
                            path = '/home/messias/Documentos/eggs/dataset/ALL/' + \
                                str(c) + '.png'
        if op == 3:
            if int(c) < 10:
                path = '/home/messias/Documentos/protozoan/dataset/ALL/0000' + \
                    str(c) + '.png'
            else:
                if int(c) < 100:
                    path = '/home/messias/Documentos/protozoan/dataset/ALL/000' + \
                        str(c) + '.png'
                else:
                    if int(c) < 1000:
                        path = '/home/messias/Documentos/protozoan/dataset/ALL/00' + \
                            str(c) + '.png'
                    else:
                        if int(c) < 10000:
                            path = '/home/messias/Documentos/protozoan/dataset/ALL/0' + \
                                str(c) + '.png'
                        else:
                            path = '/home/messias/Documentos/protozoan/dataset/ALL/' + \
                                str(c) + '.png'
        return path

    def plotImg(self, imagem):
        x = 1
        # Plotar as imagens juntas
        plt.figure(1)
        for i in imagem:
                # Percorre o vetor de path's
            plt.subplot(1, len(imagem), x)
            # Le a imagem
            image = img.imread(i)
            # Plota
            plt.imshow(image)
            # Incrementa o subplot lateral
            x += 1
        plt.show()

    def getRoots(self):
        print('===') * 30
        print('')
        print('Recuperando as Amostras Raizes')
        print('')
        global clusters_distances
        global train_bkp_class_index
        global num_c
        # Pegando apenas os index das imagens
        index = train_bkp_class_index[:, 1]
        # Cria uma lista de index para quantos cluster tiver
        list_index = []
        for i in range(0, int(num_c)):
            list_index.append(index)
        c = []
        i = 0
        path = []
        for cluster in clusters_distances[0:int(num_c)]:
            # Pega as menores distancias de cada amostra a cada cluster
            # (raizes)
            c.append(cluster.argmin())
            # Imprime as primeiras amostras selecionadas como raizes e deleta
            # do index as amostras ja utilizadas
            aux = list_index[i]
            print('Amostra selecionada do Cluster {}: {}'.format(
                i + 1, aux[c[i]]))
            aux = np.delete(aux, c[i])
            list_index[i] = aux
            # Remove ela dos clusters
            cluster = np.delete(cluster, c[i])
            clusters_distances[i] = cluster
            # Chama funcao para obter os caminhos das imagens (+1 pois o index
            # comeca em 0 mas as imagens nao)
            path.append(self.getPath(aux[c[i]], define_path))
            # Auxiliar para percorrer os vetores
            i += 1
        print('')
        # Criando o conjunto de Raizes
        global train
        label = train_bkp_class_index[:, 0]
        label = np.vstack(label)
        global set_z1
        # Aqui juntamos o conjunto de treino com a classe e imprime o true
        # label da amostra selecionada
        for index2 in c:
            set_z1.append(np.concatenate(
                (train[index2], label[index2]), axis=0))
            print('Classe da Amostra {}: {}'.format(
                index[index2], label[index2]))
        print('')
        print('Numero de amostras no conjunto Z1: {}'.format(len(set_z1)))
        print('')
        # Le as imagens
        img = []
        for i in path:
            img.append(i)
        self.plotImg(img)

    def getBorder(self):
        print('===') * 30
        print('')
        print('Recuperando as Amostras de Fronteira')
        print('')
        global clusters_distances
        global train_bkp_class_index
        global num_c
        # Pegando apenas os index das imagens
        index = train_bkp_class_index[:, 1]
        # Cria uma lista de index para quantos cluster tiver
        list_index = []
        for i in range(0, int(num_c)):
            list_index.append(index)
        c = []
        i = 0
        path = []
        for cluster in clusters_distances:
            # Pega as maiores distancias de cada amostra a cada cluster
            c.append(cluster.argmax())
            # Imprime as primeiras amostras selecionadas como raizes e deleta
            # do index as amostras ja utilizadas
            aux = list_index[i]
            print('Amostra selecionada do Cluster {}: {}'.format(
                i + 1, aux[c[i]]))
            aux = np.delete(aux, c[i])
            list_index[i] = aux
            # Remove ela dos clusters
            cluster = np.delete(cluster, c[i])
            clusters_distances[i] = cluster
            # Chama funcao para obter os caminhos das imagens (+1 pois o index
            # comeca em 0 mas as imagens nao)
            path.append(self.getPath(aux[c[i]], define_path))
            # Auxiliar para percorrer os vetores
            i += 1
        print('')
        # Criando o conjunto C
        global train
        label = train_bkp_class_index[:, 0]
        label = np.vstack(label)
        global set_C
        # Aqui juntamos o conjunto de treino com a classe e imprime o true
        # label da amostra selecionada
        for index2 in c:
            set_C.append(np.concatenate(
                (train[index2], label[index2]), axis=0))
            print('Classe da Amostra {}: {}'.format(
                index[index2], label[index2]))
        print('')
        print('Numero de amostras no conjunto C: {}'.format(len(set_C)))
        print('')
        # Le as imagens
        img = []
        for i in path:
            img.append(i)
        self.plotImg(img)

    def trainClassifier(self):
        print('===') * 30
        print('')
        print('Treinando os Classificadores com Z1')
        print('')
        global set_z1
        # Cria um numpy array para usar de cabecalho do dataframe
        index = np.arange(1, len(set_z1[0]) + 1)
        # Cria o Dataframe
        df = pd.DataFrame(data=set_z1, columns=index)
        # Renomeia a ultima coluna para CLASS
        df.rename(columns={int(len(set_z1[0])): 'class'}, inplace=True)
        # Obter o numero de features
        length = df.iloc[0, :].values
        # Separa a coluna de rotulos
        X_labels = df.iloc[:, len(length) - 1].values
        # Armazenar as caracteristicas sem a Classe na variavel X
        X_train = df.iloc[:, 0:(len(length) - 1)].values
        # Inicializar os classificadores
        # Gaussian Naive Bayes
        global gnb
        gnb = GaussianNB()
        # Logistic Regression
        global logreg
        logreg = LogisticRegression()
        # Decision Tree
        global dectree
        dectree = DecisionTreeClassifier()
        # K-Nearest Neighbors
        global knn
        knn = KNeighborsClassifier(n_neighbors=int(num_c))
        # Linear Discriminant Analysis
        global lda
        lda = LinearDiscriminantAnalysis()
        # Support Vector Machine
        global svm
        svm = SVC()
        # RandomForest
        global rf
        rf = RandomForestClassifier()
        # Neural Net
        global nnet
        nnet = MLPClassifier(alpha=1)
        # Treinar os classificadores
        t = time.time()
        model = gnb.fit(X_train, X_labels)
        print('Treino do Gaussian Naive Bayes Terminado. (Tempo de execucao: {})'.format(
            time.time() - t))
        t = time.time()
        model = logreg.fit(X_train, X_labels)
        print('Treino do Logistic Regression Terminado. (Tempo de execucao: {})'.format(
            time.time() - t))
        t = time.time()
        model = dectree.fit(X_train, X_labels)
        print('Treino do Decision Tree Terminado. (Tempo de execucao: {})'.format(
            time.time() - t))
        t = time.time()
        model = knn.fit(X_train, X_labels)
        print('Treino do K-Nearest Neighbors Terminado. (Tempo de execucao: {})'.format(time.time() - t))
        t = time.time()
        model = lda.fit(X_train, X_labels)
        print('Treino do Linear Discriminant Analysis Terminado. (Tempo de execucao: {})'.format(
            time.time() - t))
        t = time.time()
        model = svm.fit(X_train, X_labels)
        print('Treino do Support Vector Machine Terminado. (Tempo de execucao: {})'.format(
            time.time() - t))
        t = time.time()
        model = rf.fit(X_train, X_labels)
        print('Treino do RandomForest Terminado. (Tempo de execucao: {})'.format(
            time.time() - t))
        t = time.time()
        model = nnet.fit(X_train, X_labels)
        print('Treino do Neural Net Terminado. (Tempo de execucao: {})'.format(
            time.time() - t))
        print('')
        print('Pronto..')
        print('')

    def testClassifier(self):
        print('===') * 30
        print('')
        print('Testando os Classificadores com C')
        print('')
        global set_C
        # Cria um numpy array para usar de cabecalho do dataframe
        index = np.arange(1, len(set_C[0]) + 1)
        # Cria o Dataframe
        df = pd.DataFrame(data=set_C, columns=index)
        # Renomeia a ultima coluna para CLASS
        df.rename(columns={int(len(set_C[0])): 'class'}, inplace=True)
        # Obter o numero de features
        length = df.iloc[0, :].values
        # Separa a coluna de rotulos
        Y_labels = df.iloc[:, len(length) - 1].values
        # Armazenar as caracteristicas sem a Classe na variavel Y
        Y_train = df.iloc[:, 0:(len(length) - 1)].values
        # Fazer predicoes e printar a acuracia obtida
        t = time.time()
        preds = gnb.predict(Y_train)
        print('Acuracia obtida com o Gaussian Naive Bayes: {}. (Tempo de execucao: {})'.format(
            accuracy_score(Y_labels, preds), time.time() - t))
        t = time.time()
        preds = logreg.predict(Y_train)
        print('Acuracia obtida com o Logistic Regression: {}. (Tempo de execucao: {})'.format(
            accuracy_score(Y_labels, preds), time.time() - t))
        t = time.time()
        preds = dectree.predict(Y_train)
        print('Acuracia obtida com o Decision Tree: {}. (Tempo de execucao: {})'.format(
            accuracy_score(Y_labels, preds), time.time() - t))
        t = time.time()
        preds = knn.predict(Y_train)
        print('Acuracia obtida com o K-Nearest Neighbors: {}. (Tempo de execucao: {})'.format(
            accuracy_score(Y_labels, preds), time.time() - t))
        t = time.time()
        preds = lda.predict(Y_train)
        print('Acuracia obtida com o Linear Discriminant Analysis: {}. (Tempo de execucao: {})'.format(
            accuracy_score(Y_labels, preds), time.time() - t))
        t = time.time()
        preds = lda.predict(Y_train)
        print('Acuracia obtida com o Support Vector Machine: {}. (Tempo de execucao: {})'.format(
            accuracy_score(Y_labels, preds), time.time() - t))
        t = time.time()
        preds = rf.predict(Y_train)
        print('Acuracia obtida com o RandomForest: {}. (Tempo de execucao: {})'.format(
            accuracy_score(Y_labels, preds), time.time() - t))
        t = time.time()
        preds = nnet.predict(Y_train)
        print('Acuracia obtida com o Neural Net: {}. (Tempo de execucao: {})'.format(
            accuracy_score(Y_labels, preds), time.time() - t))
        print('')
        print('Pronto..')
        print('')

#---------------------MAIN---------------------
# Instancia a classe TK() permitindo os widgets possam ser utilizados
root = Tk()
normalizationList = ['MinMaxScaler',
                     'StandardScaler', 'MaxAbsScaler', 'RobustScaler']
selectedNormalization = StringVar()
selectedNormalization.set(normalizationList[0])
Application(root)
root.mainloop()
