from rede_mlp import RedeMlp
from validacoes import validacao_cruzada_erro_minimo, validacao_cruzada_parada_antecipada
import numpy as np
import time

# carrega os arquivos X.npy e Y_classe.npy

arquivo_X = np.load(r"C:\Users\rafat\Documents\GitHub\EP1_MLP\X.npy")
matriz_target = np.load(r"C:\Users\rafat\Documents\GitHub\EP1_MLP\Y_classe.npy")

matriz_X = []

for dados in arquivo_X:
    # Converte o arquivo X.npy que contém um array de 4 dimensões com a seguinte forma: (1326, 10, 12, 1) em uma matriz no formato (1326, 120)
    linha = dados.flatten()
    linha[linha == -1] = 0 # Substituímos os valores -1 por 0 para que coincida com a codificação one-hot do arquivo Y_classe
    matriz_X.append(linha)

matriz_X = np.array(matriz_X)
matriz_target = np.array(matriz_target)

# Inclusão de parâmetros da rede [nº neuronios camada de entrada, nº neuronios camada escondida, nº neuronios camada de saída, taxa de aprendizado]

mlp1 = RedeMlp([120, 63, 26, 0.4], "mlp1", r"C:\Users\rafat\Documents\GitHub\EP1_MLP")
mlp1.resumo_funcionamento()

print("Selecione a opção desejada")
print("Digite 1 para utilizar o treinamento com parada antecipada")
print("Digite 2 para utilizar o treinamento com parada através do erro mínimo ou número máximo de épocas")
print("Digite 3 para para pular para as funções de validação cruzada")
comando = int(input("Digite sua escolha: "))


if comando == 1:
    inicio = time.time()
    mlp1.treinamento_parada_antecipada(epocas=10000,
                                       matriz_X=matriz_X[:858],
                                       matriz_target=matriz_target[:858],
                                       matriz_X_validacao=matriz_X[858:1196],
                                       matriz_target_validacao=matriz_target[858:1196],
                                       paciencia=5, fold=None
                                       )
    fim = time.time()

    print(f"Tempo de execução do treinamento: {fim - inicio:.4f} segundos")

    inicio = time.time()
    mlp1.teste(matriz_X_teste=matriz_X[1196:1326],
               matriz_target_teste=matriz_target[1196:1326],
               exibir_matriz_confusao=True, nome_arquivo_matriz_confusao="PA")

    fim = time.time()

    print(f"Tempo de execução do treinamento: {fim - inicio:.4f} segundos")


if comando == 2:
    inicio = time.time()
    mlp1.treinamento_erro_minimo(
        epocas=1000,
        matriz_X=matriz_X[:1196],
        matriz_target=matriz_target[:1196],
        erro_minimo=0.012, fold=None
    )
    fim = time.time()

    print(f"Tempo de execução do treinamento: {fim - inicio:.4f} segundos")

elif comando == 3:

    print("Selecione o tipo de função de avaliação")
    print("Digite 1 para efetuar a validação cruzada com parada antecipada")
    print("Digite 2 para efetuar a validação cruzada com parada através do erro mínimo ou número máximo de épocas")
    print("Digite 3 para sair")
    comand = int(input("Digite sua escolha: "))

    if comand == 1:
        inicio = time.time()
        validacao_cruzada_parada_antecipada(rede=mlp1,
                                            matriz_X=matriz_X[:1196],
                                            matriz_target=matriz_target[:1196],
                                            k=10, epocas=1000, paciencia=6,
                                            matriz_X_teste=matriz_X[1197:1326],
                                            matriz_target_teste=matriz_target[1197:1326]
                                            )
        fim = time.time()
        print(f"Tempo de execução do treinamento: {fim - inicio:.4f} segundos")
        
    elif comand == 2:
        inicio = time.time()
        validacao_cruzada_erro_minimo(rede=mlp1,
                                      matriz_X=matriz_X[:1196],
                                      matriz_target=matriz_target[:1196],
                                      k=10, epocas=1000, erro_minimo=0.019,
                                      matriz_X_teste=matriz_X[1197:1326],
                                      matriz_target_teste=matriz_target[1197:1326]
                                      )
        fim = time.time()
        print(f"Tempo de execução do treinamento: {fim - inicio:.4f} segundos")

    elif comand == 3:
        print("Saindo do programa...")
        exit()
