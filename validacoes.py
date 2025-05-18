import numpy as np
import os
from rede_mlp import RedeMlp
import time


def validacao_cruzada_parada_antecipada(rede, matriz_X, matriz_target, k, epocas, paciencia, matriz_X_teste, matriz_target_teste):
        # Divide os dados em k folds
        matriz_fold_X = np.array_split(matriz_X, k)
        matriz_fold_target = np.array_split(matriz_target, k)

        vetor_acuracia = []

        for i in range(k - 1):
            a = k - 1  # Último fold reservado como teste em cada rodada

            # Constrói os conjuntos de treino, validação e teste
            X_val = matriz_fold_X[i]
            target_val = matriz_fold_target[i]

            X_teste = matriz_fold_X[a]
            target_teste = matriz_fold_target[a]

            # Remove os folds de validação e teste para formar o treino

            X_treinamento = [fold for j, fold in enumerate(matriz_fold_X) if j not in [i, a]]
            target_treinamento = [fold for j, fold in enumerate(matriz_fold_target) if j not in [i, a]]

            # Junta os folds restantes
            X_treinamento = np.concatenate(X_treinamento, axis=0)
            target_treinamento = np.concatenate(target_treinamento, axis=0)

            # Treinamento com validação
            rede.treinamento_parada_antecipada(epocas, X_treinamento, target_treinamento, X_val, target_val, paciencia, fold=i+1)

            # Avaliação
            acuracia, _ = rede.teste(X_teste, target_teste, exibir_matriz_confusao=False, nome_arquivo_matriz_confusao=None)
            vetor_acuracia.append(acuracia)

            # --- Reinicializa os pesos ---
            rede.matriz_v = np.zeros((rede.n + 1, rede.p))
            rede.matriz_v[0, :] = 1
            rede.matriz_v[1:, :] = np.random.uniform(-0.5, 0.5, size=(rede.n, rede.p))

            rede.matriz_w = np.zeros((rede.p + 1, rede.m))
            rede.matriz_w[0, :] = 1
            rede.matriz_w[1:, :] = np.random.uniform(-0.5, 0.5, size=(rede.p, rede.m))


        # Estatísticas dos folds
        media_acuracia = np.mean(vetor_acuracia)
        desvio_padrao = np.std(vetor_acuracia)

        # --- Salvar resultados em arquivo TXT ---
        with open("resultado_avaliacao.txt", "w") as arquivo:
            arquivo.write("Resultado da Validação Cruzada\n")
            arquivo.write("==============================\n")
            arquivo.write(f"Vetor de Acurácias: {vetor_acuracia}\n")
            arquivo.write(f"Média de Acurácia: {media_acuracia:.4f}\n")
            arquivo.write(f"Desvio Padrão: {desvio_padrao:.4f}\n")

        # --- Reinicializa os pesos ---
        rede.matriz_v = np.zeros((rede.n + 1, rede.p))
        rede.matriz_v[0, :] = 1
        rede.matriz_v[1:, :] = np.random.uniform(-0.5, 0.5, size=(rede.n, rede.p))

        rede.matriz_w = np.zeros((rede.p + 1, rede.m))
        rede.matriz_w[0, :] = 1
        rede.matriz_w[1:, :] = np.random.uniform(-0.5, 0.5, size=(rede.p, rede.m))

        # Treinamento para gerar um novo conjunto de pesos
        rede.treinamento_parada_antecipada(epocas, X_treinamento, target_treinamento, X_val, target_val, paciencia, fold = "CVPA")

        # Avaliação final no conjunto de teste externo
        acuracia_final, matriz_confusao = rede.teste(matriz_X_teste, matriz_target_teste, exibir_matriz_confusao=True, nome_arquivo_matriz_confusao="CVPA")
        print(f"Acurácia final no conjunto de teste externo: {acuracia_final:.4f}")
        print("Matriz de Confusão:")
        print(matriz_confusao)

        with open("resultado_avaliacao.txt", "a") as arquivo:
            arquivo.write("\nAvaliação Final no Conjunto de Teste Externo\n")
            arquivo.write("===========================================\n")
            arquivo.write(f"Acurácia Final: {acuracia_final:.4f}\n")


def validacao_cruzada_erro_minimo(rede, matriz_X, matriz_target, k, epocas, erro_minimo, matriz_X_teste, matriz_target_teste):
    # Divide os dados em k folds
    matriz_fold_X = np.array_split(matriz_X, k)
    matriz_fold_target = np.array_split(matriz_target, k)

    vetor_acuracia = []

    for i in range(k - 1):
        a = k - 1  # Último fold reservado como teste em cada rodada

        # Constrói o conjunto de teste

        X_teste = matriz_fold_X[a]
        target_teste = matriz_fold_target[a]

        # Remove os folds de teste para formar o treino

        X_treinamento = [fold for j, fold in enumerate(matriz_fold_X) if j not in [a]]
        target_treinamento = [fold for j, fold in enumerate(matriz_fold_target) if j not in [a]]

        # Junta os folds restantes
        X_treinamento = np.concatenate(X_treinamento, axis=0)
        target_treinamento = np.concatenate(target_treinamento, axis=0)

        # Treinamento com erro minimo
        rede.treinamento_erro_minimo(epocas, X_treinamento, target_treinamento, erro_minimo, fold = i+1)
        
        # Avaliação
        acuracia, _ = rede.teste(X_teste, target_teste, exibir_matriz_confusao=False, nome_arquivo_matriz_confusao=None)
        vetor_acuracia.append(acuracia)

         # --- Reinicializa os pesos ---
        rede.matriz_v = np.zeros((rede.n + 1, rede.p))
        rede.matriz_v[0, :] = 1
        rede.matriz_v[1:, :] = np.random.uniform(-0.5, 0.5, size=(rede.n, rede.p))

        rede.matriz_w = np.zeros((rede.p + 1, rede.m))
        rede.matriz_w[0, :] = 1
        rede.matriz_w[1:, :] = np.random.uniform(-0.5, 0.5, size=(rede.p, rede.m))

    # Estatísticas dos folds
    media_acuracia = np.mean(vetor_acuracia)
    desvio_padrao = np.std(vetor_acuracia)

    print(f"Média de acurácia nos folds: {media_acuracia:.4f}")
    print(f"Desvio padrão das acurácias: {desvio_padrao:.4f}")

     # --- Reinicializa os pesos ---
    rede.matriz_v = np.zeros((rede.n + 1, rede.p))
    rede.matriz_v[0, :] = 1
    rede.matriz_v[1:, :] = np.random.uniform(-0.5, 0.5, size=(rede.n, rede.p))

    rede.matriz_w = np.zeros((rede.p + 1, rede.m))
    rede.matriz_w[0, :] = 1
    rede.matriz_w[1:, :] = np.random.uniform(-0.5, 0.5, size=(rede.p, rede.m))

    # Treinamento para gerar um novo conjunto de pesos

    rede.treinamento_erro_minimo(epocas, X_treinamento, target_treinamento, erro_minimo, fold = "CVEM")

    # Avaliação final no conjunto de teste externo
    acuracia_final, matriz_confusao = rede.teste(matriz_X_teste, matriz_target_teste, exibir_matriz_confusao=True, nome_arquivo_matriz_confusao="CVEM")
    print(f"Acurácia final no conjunto de teste externo: {acuracia_final:.4f}")
    print("Matriz de Confusão:")
    print(matriz_confusao)
