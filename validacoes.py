import numpy as np
import os
from rede_mlp import RedeMlp


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
            rede.treinamento_parada_antecipada(epocas, X_treinamento, target_treinamento, X_val, target_val, paciencia)

            # Avaliação
            acuracia, _ = rede.teste(X_teste, target_teste, exibir_matriz_confusao=False)
            vetor_acuracia.append(acuracia)

            # --- Reinicializa os pesos ---
            rede.matriz_v = np.zeros((rede.n + 1, rede.p))
            rede.matriz_v[0, :] = 1
            rede.matriz_v[1:, :] = np.random.uniform(-0.5, 0.5, size=(rede.n, rede.p))

            rede.matriz_w = np.zeros((rede.p + 1, rede.m))
            rede.matriz_w[0, :] = 1
            rede.matriz_w[1:, :] = np.random.uniform(-0.5, 0.5, size=(rede.p, rede.m))

        
        # --- Recupera os pesos do último treinamento ---
        pasta = "C:/Users/gaaaa/OneDrive/Desktop/Faculdade/IA/MLP"
        rede.arquivo_v = os.path.join(pasta, f"pesos_camada_entrada_para_escondida_{rede.nome_rede}.npy")
        rede.arquivo_w = os.path.join(pasta, f"pesos_camada_escondida_saida_{rede.nome_rede}.npy")
        rede.matriz_v = np.load(rede.arquivo_v)
        rede.matriz_w = np.load(rede.arquivo_w)

        # Estatísticas dos folds
        media_acuracia = np.mean(vetor_acuracia)
        desvio_padrao = np.std(vetor_acuracia)

        print(f"Média de acurácia nos folds: {media_acuracia:.4f}")
        print(f"Desvio padrão das acurácias: {desvio_padrao:.4f}")

        # Avaliação final no conjunto de teste externo
        acuracia_final, matriz_confusao = rede.teste(matriz_X_teste, matriz_target_teste, exibir_matriz_confusao=True)
        print(f"Acurácia final no conjunto de teste externo: {acuracia_final:.4f}")
        print("Matriz de Confusão:")
        print(matriz_confusao)


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
        rede.treinamento_erro_minimo(epocas, X_treinamento, target_treinamento, erro_minimo)
        
        # Avaliação
        acuracia, _ = rede.teste(X_teste, target_teste, exibir_matriz_confusao=False)
        vetor_acuracia.append(acuracia)

         # --- Reinicializa os pesos ---
        rede.matriz_v = np.zeros((rede.n + 1, rede.p))
        rede.matriz_v[0, :] = 1
        rede.matriz_v[1:, :] = np.random.uniform(-0.5, 0.5, size=(rede.n, rede.p))

        rede.matriz_w = np.zeros((rede.p + 1, rede.m))
        rede.matriz_w[0, :] = 1
        rede.matriz_w[1:, :] = np.random.uniform(-0.5, 0.5, size=(rede.p, rede.m))

        
    # --- Recupera os pesos do último treinamento ---
    pasta = "C:/Users/gaaaa/OneDrive/Desktop/Faculdade/IA/MLP"
    rede.arquivo_v = os.path.join(pasta, f"pesos_camada_entrada_para_escondida_{rede.nome_rede}.npy")
    rede.arquivo_w = os.path.join(pasta, f"pesos_camada_escondida_saida_{rede.nome_rede}.npy")
    rede.matriz_v = np.load(rede.arquivo_v)
    rede.matriz_w = np.load(rede.arquivo_w)

    # Estatísticas dos folds
    media_acuracia = np.mean(vetor_acuracia)
    desvio_padrao = np.std(vetor_acuracia)

    print(f"Média de acurácia nos folds: {media_acuracia:.4f}")
    print(f"Desvio padrão das acurácias: {desvio_padrao:.4f}")

    # Avaliação final no conjunto de teste externo
    acuracia_final, matriz_confusao = rede.teste(matriz_X_teste, matriz_target_teste, exibir_matriz_confusao=True)
    print(f"Acurácia final no conjunto de teste externo: {acuracia_final:.4f}")
    print("Matriz de Confusão:")
    print(matriz_confusao)