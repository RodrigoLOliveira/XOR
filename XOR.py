import numpy as np
import csv

def sigmoid(soma):
    return 1/ (1+np.exp(-soma))

def sigmoidDerivada(sig):
    return sig * (1 - sig)

entradas = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [1,1]])
saidas = np.array([[0],[1],[1],[0]])

pesos0 = 2 * np.random.random((2,3)) - 1
pesos1 = 2 * np.random.random((3,1)) - 1

epocas = 200_000
taxaAprendizagem = 0.6
momento = 1

#Roda o aprendizado a quantidade de epocas configuradas
for e in range(epocas):
    #Recupera os valores da camada de entrada para realizar as operações
    camadaEntrada = entradas

    #Multiplicação e soma da camada entrada com os pesos da camada de entrada para a camada oculta
    #pegando o dotproduct ou Produto escalar dos dois vetores
    somaSinapse0 = np.dot(camadaEntrada, pesos0)

    #Executa uma função para retornar o valor sigmoid da soma das sinapses da camada de entrada
    camadaOculta = sigmoid(somaSinapse0)

    #Recupera o produto escalar dos vetores da camada oculta e seus pesos
    somaSinapse1 = np.dot(camadaOculta, pesos1)

    #Recupera o valor da saida
    camadaSaida = sigmoid(somaSinapse1)

    #Recupera os valores de erro da rede neural
    erroSaida = saidas - camadaSaida

    #Obtem a média de erro da rede
    mediaAbsoluta = np.mean(np.abs(erroSaida))
    if (e % 1000 == 0):
        print("Epoca: " + str(e) + " | Erro: " + str(mediaAbsoluta))

    #Encontrar Delta da camada oculta para calculo do erro 
    derivadaSaida = sigmoidDerivada(camadaSaida)
    deltaSaida = erroSaida * derivadaSaida

    pesos1Transposta = pesos1.T
    deltaSaidaXPeso = deltaSaida.dot(pesos1Transposta)
    deltaCamadaOculta = deltaSaidaXPeso * sigmoidDerivada(camadaOculta)

    #Utilizando o método Backpropagation
    #é feito o reajuste dos pesos da camada oculta para a camada de saida
    camadaOcultaTransposta = camadaOculta.T
    pesosNovo1 = camadaOcultaTransposta.dot(deltaSaida)
    pesos1 = (pesos1 * momento) + (pesosNovo1 * taxaAprendizagem)

    #Realiza o reajuste dos pesos da camada de entrada para a camada oculta
    camadaEntradaTransposta = camadaEntrada.T
    pesosNovos0 = camadaEntradaTransposta.dot(deltaCamadaOculta)
    pesos0 = (pesos0 * momento) + (pesosNovos0 * taxaAprendizagem)


print("Taxa de acerto é de aproximadamente " + str(int((1 - mediaAbsoluta)*100)) + "%")

