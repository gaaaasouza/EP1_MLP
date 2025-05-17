import os
import numpy as np
import matplotlib.pyplot as plt
import string
import random
import numpy as np


class RedeMlp:
    def __init__(self, modelo_mlp, nome_rede):          # construtor da rede
        self.n = modelo_mlp[0]      # neurônios entrada
        self.p = modelo_mlp[1]      # neurônios camada escondida
        self.m = modelo_mlp[2]      # neurônios saída
        self.alfa = modelo_mlp[3]   # taxa de aprendizado
        self.nome_rede = nome_rede
        pasta = "C:/Users/gaaaa/OneDrive/Desktop/Faculdade/IA/MLP"
        self.arquivo_v = os.path.join(pasta, f"pesos_camada_entrada_para_escondida_{nome_rede}.npy")
        self.arquivo_w = os.path.join(pasta, f"pesos_camada_escondida_saida_{nome_rede}.npy")

        # Criando matriz de pesos com valores aleatórios entre -0.5, 0.5 e bias = 1 da camada de entrada -> camada escondida
        self.matriz_v = np.zeros((self.n + 1, self.p))  
        self.matriz_v[0, :] = 1
        self.matriz_v[1:, :] = np.random.uniform(-0.5, 0.5, size=(self.n, self.p))
        np.save(self.arquivo_v, self.matriz_v)

        # Criando matriz de pesos com valores aleatórios entre -0.5, 0.5 e bias = 1 da camada escondida -> camada de saída
        self.matriz_w = np.zeros((self.p + 1, self.m))
        self.matriz_w[0, :] = 1
        self.matriz_w[1:, :] = np.random.uniform(-0.5, 0.5, size=(self.p, self.m))
        np.save(self.arquivo_w, self.matriz_w)


    def sigmoid(self, x):               
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2
    
    def vetor_para_letra(self, vetor):      #função que converte vetor one-hot para letra
        indice = np.argmax(vetor)
        return chr(ord('A') + indice)


    def func_matriz_confusao(self, matriz_confusao):      # função que gera a matriz confusão após o teste da rede
        plt.figure(figsize=(12, 10))
        plt.imshow(matriz_confusao, interpolation='nearest', cmap='Blues')
        plt.title("Matriz de Confusão")
        plt.colorbar()

        # Define os rótulos dos eixos (A até Z)
        classes = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        # Adiciona os números dentro de cada célula
        for i in range(len(classes)):
            for j in range(len(classes)):
                valor = matriz_confusao[i, j]
                cor = "white" if valor > matriz_confusao.max() / 2 else "black"
                plt.text(j, i, str(valor), horizontalalignment="center", color=cor)

        plt.xlabel("Letra Predita")
        plt.ylabel("Letra Real")
        plt.tight_layout()
        plt.show()


    def feedforward(self, vetor_X):                 # função que executa a fase feedforward
        bias_v = self.matriz_v[0]
        pesos_v = self.matriz_v[1:]
        vetor_Z_in = bias_v + vetor_X @ pesos_v
        vetor_Z = self.sigmoid(vetor_Z_in)

        bias_w = self.matriz_w[0]
        pesos_w = self.matriz_w[1:]
        vetor_Y_in = bias_w + vetor_Z @ pesos_w
        vetor_Y = self.sigmoid(vetor_Y_in)

        return vetor_Y, vetor_Y_in, vetor_Z, vetor_Z_in

    def backpropagation(self, vetor_target, vetor_Y, vetor_Y_in, vetor_Z, vetor_Z_in, vetor_X): # função que executa a fase de retropropagação do erro
        vetor_erro = vetor_target - vetor_Y
        vetor_deltinha_y = vetor_erro * self.derivada_sigmoid(vetor_Y_in)

        matriz_delta_w = np.zeros_like(self.matriz_w)
        matriz_delta_w[0] = self.alfa * vetor_deltinha_y
        matriz_delta_w[1:] = self.alfa * np.outer(vetor_Z, vetor_deltinha_y)

        vetor_deltinha_z = self.derivada_sigmoid(vetor_Z_in) * (self.matriz_w[1:] @ vetor_deltinha_y)
        matriz_delta_v = np.zeros_like(self.matriz_v)
        matriz_delta_v[0] = self.alfa * vetor_deltinha_z
        matriz_delta_v[1:] = self.alfa * np.outer(vetor_X, vetor_deltinha_z)

        return vetor_erro, matriz_delta_v, matriz_delta_w

    def atualiza_pesos(self, matriz_delta_v, matriz_delta_w):           # função que atualiza os pesos após o cálculo dos deltas
        self.matriz_v += matriz_delta_v
        self.matriz_w += matriz_delta_w

    def erro_quadratico_medio(self, matriz_de_erro):                    # função que calcula o erro quadrático médio
        erro_total = np.sum(np.square(matriz_de_erro))
        return erro_total / matriz_de_erro.shape[0]

    def treinamento_parada_antecipada(self, epocas, matriz_X, matriz_target, matriz_X_validacao, matriz_target_validacao, paciencia):
        vetor_erro_medio = []
        vetor_erro_medio_validacao = []

        melhor_erro_validacao = float('inf')
        epocas_sem_melhora = 0  # contador de épocas sem melhora

        for epoca in range(epocas):

            # ---------- TREINAMENTO ----------
            matriz_de_erro = np.zeros_like(matriz_target)
            for i in range(matriz_X.shape[0]):
                Y, Y_in, Z, Z_in = self.feedforward(matriz_X[i])
                erro, delta_v, delta_w = self.backpropagation(matriz_target[i], Y, Y_in, Z, Z_in, matriz_X[i])
                self.atualiza_pesos(delta_v, delta_w)
                matriz_de_erro[i] = erro
            erro_medio = self.erro_quadratico_medio(matriz_de_erro)
            vetor_erro_medio.append(erro_medio)
            print(f"Época {epoca}, Erro Médio treinamento = {erro_medio:.6f}")

            # ---------- VALIDAÇÃO ----------
            matriz_de_erro_validacao = np.zeros_like(matriz_target_validacao)
            for j in range(matriz_X_validacao.shape[0]):
                Y_val, _, _, _ = self.feedforward(matriz_X_validacao[j])
                matriz_de_erro_validacao[j] = matriz_target_validacao[j] - Y_val
            erro_medio_validacao = self.erro_quadratico_medio(matriz_de_erro_validacao)
            vetor_erro_medio_validacao.append(erro_medio_validacao)
            print(f"Época {epoca}, Erro Médio validacao = {erro_medio_validacao:.6f}")

            # ---------- PARADA ANTECIPADA ----------
            if erro_medio_validacao < melhor_erro_validacao:
                melhor_erro_validacao = erro_medio_validacao
                epocas_sem_melhora = 0  # zera contador
            else:
                epocas_sem_melhora += 1
                print(f"Sem melhora há {epocas_sem_melhora} épocas.")

                if epocas_sem_melhora >= paciencia:
                    print(f"Parada antecipada: erro de validação não melhora há {paciencia} épocas.")
                    break

        # print("Vetor Y final de treino:", Y)
        # print("Vetor Y final de validacao:", Y_val)

        # ---------- Salvando os pesos ----------

        np.save(f"pesos_camada_entrada_para_escondida_{self.nome_rede}_final.npy", self.matriz_v)
        np.save(f"pesos_camada_escondida_saida_{self.nome_rede}_final.npy", self.matriz_w)

        # ---------- Gráfico ----------
        plt.plot(vetor_erro_medio, label="Treinamento")
        plt.plot(vetor_erro_medio_validacao, label="Validação")
        plt.xlabel("Épocas")
        plt.ylabel("Erro Quadrático Médio")
        plt.title("Erro por Época com Parada Antecipada")
        plt.legend()
        plt.show()

    def treinamento_erro_minimo(self, epocas, matriz_X, matriz_target, erro_minimo):           # treinamento com parada utilizando limite do valor do erro quadrático médio por época
        vetor_erro_medio = []
        vetor_erro_medio.append(1e10)
        for epoca in range(epocas):
            matriz_de_erro = np.zeros_like(matriz_target)
            for i in range(matriz_X.shape[0]):
                matriz_X[i]
                Y, Y_in, Z, Z_in = self.feedforward(matriz_X[i])
                erro, delta_v, delta_w = self.backpropagation(matriz_target[i], Y, Y_in, Z, Z_in, matriz_X[i])
                self.atualiza_pesos(delta_v, delta_w)
                matriz_de_erro[i] = erro
            erro_medio = self.erro_quadratico_medio(matriz_de_erro)
            vetor_erro_medio.append(erro_medio)
            print(f"Época {epoca}, Erro Médio = {erro_medio:.6f}")
            if erro_medio < erro_minimo:
                break

        # print("Vetor Y final de treino:", Y)

        plt.plot(vetor_erro_medio)
        plt.xlabel("Épocas")
        plt.ylabel("Erro Quadrático Médio")
        plt.title("Erro por Época")
        min_erro = min(vetor_erro_medio)
        max_erro = max(vetor_erro_medio)
        plt.yticks(np.linspace(min_erro, max_erro, 20))  # 20 divisões entre mínimo e máximo

        plt.grid(True)
        plt.show()

        # ---------- Salvando os pesos ----------

        np.save(f"pesos_camada_entrada_para_escondida_{self.nome_rede}_final.npy", self.matriz_v)
        np.save(f"pesos_camada_escondida_saida_{self.nome_rede}_final.npy", self.matriz_w)


    def teste(self, matriz_X_teste, matriz_target_teste, exibir_matriz_confusao=False):        # função que testa a rede
        acerto = 0
        matriz_confusao = np.zeros((26, 26), dtype=int)

        for i in range(matriz_X_teste.shape[0]):
            vetor_Y_teste, _, _, _ = self.feedforward(matriz_X_teste[i])
            letra_Y = self.vetor_para_letra(vetor_Y_teste)
            letra_target = self.vetor_para_letra(matriz_target_teste[i])

            if letra_Y == letra_target:
                acerto += 1

            # Posição real (linha) e predita (coluna) na matriz de confusão
            linha = ord(letra_target) - ord('A')
            coluna = ord(letra_Y) - ord('A')
            matriz_confusao[linha, coluna] += 1

        acuracia = acerto / matriz_X_teste.shape[0]
        print(f"Acurácia: {acuracia:.4f}")

        if exibir_matriz_confusao:
            self.func_matriz_confusao(matriz_confusao)

        return acuracia, matriz_confusao