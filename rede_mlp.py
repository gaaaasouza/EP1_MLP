import os
import numpy as np
import matplotlib.pyplot as plt


class RedeMlp:
    def __init__(self, modelo_mlp, nome_rede, pasta):          # construtor da rede
        self.n = modelo_mlp[0]      # neurônios entrada
        self.p = modelo_mlp[1]      # neurônios camada escondida
        self.m = modelo_mlp[2]      # neurônios saída
        self.alfa = modelo_mlp[3]   # taxa de aprendizado
        self.nome_rede = nome_rede
        self.pasta = pasta
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

    def resumo_funcionamento(self):
        print("=" * 60)
        print(f"Rede MLP Inicializada: {self.nome_rede}")
        print("-" * 60)
        print(f"• Neurônios de Entrada      : {self.n}")
        print(f"• Neurônios na Camada Oculta: {self.p}")
        print(f"• Neurônios de Saída        : {self.m}")
        print(f"• Taxa de Aprendizado (α)   : {self.alfa}")
        print("-" * 60)
        print("• Função de Ativação        : Sigmoid")
        print("• Algoritmo de Treinamento  : Backpropagation")
        print("• Critérios de Parada       :")
        print("   - Parada antecipada por paciência (sem melhora)")
        print("   - Erro quadrático médio mínimo")
        print("-" * 60)
        print(f"• Pesos salvos em: {self.arquivo_v}")
        print(f"               e: {self.arquivo_w}")
        print("=" * 60)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def derivada_sigmoid(self, x):
        return np.exp(-x) / (1 + np.exp(-x))**2

    def vetor_para_letra(self, vetor):  # função que converte vetor one-hot para letra
        indice = np.argmax(vetor)
        return chr(ord('A') + indice)

    def func_matriz_confusao(self, matriz_confusao, nome_arquivo=None):      # função que gera a matriz confusão após o teste da rede
        """
        Exibe uma matriz de confusão como um heatmap,
        com rotulagem de letras de A a Z.
        """
        letras = [chr(ord('A') + i) for i in range(26)]  # ['A', 'B', ..., 'Z']

        # Cria o gráfico
        fig, ax = plt.subplots(figsize=(10, 8))

        # Desenha a matriz como uma imagem
        cax = ax.matshow(matriz_confusao, cmap='Blues')
        plt.colorbar(cax)

        # Adiciona rótulos
        ax.set_xticks(np.arange(26))
        ax.set_yticks(np.arange(26))
        ax.set_xticklabels(letras)
        ax.set_yticklabels(letras)

        # Rótulos dos eixos
        plt.xlabel('Letra Predita')
        plt.ylabel('Letra Real')
        plt.title('Matriz de Confusão', pad=20)

        # Anota os valores em cada célula
        for i in range(26):
            for j in range(26):
                valor = matriz_confusao[i][j]
                if valor > 0:
                    ax.text(j, i, str(valor), va='center', ha='center', color='black')

        plt.tight_layout()
        if nome_arquivo:
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            plt.close()
        else:
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

    def backpropagation(self, vetor_target, vetor_Y, vetor_Y_in, vetor_Z, vetor_Z_in, vetor_X):  # função que executa a fase de retropropagação do erro
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

    def treinamento_parada_antecipada(self, epocas, matriz_X, matriz_target, matriz_X_validacao, matriz_target_validacao, paciencia, fold=None):
        vetor_erro_medio = []
        vetor_erro_medio_validacao = []

        # Vetores para acompanhar a convergência dos pesos
        normas_l2_v = []
        normas_l2_w = []

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

            # ---------- Norma L2 dos pesos ----------
            norma_v = np.linalg.norm(self.matriz_v)
            norma_w = np.linalg.norm(self.matriz_w)
            normas_l2_v.append(norma_v)
            normas_l2_w.append(norma_w)

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

        # ---------- Salvando os pesos ----------

        np.save(f"pesos_camada_entrada_para_escondida_{self.nome_rede}_final.npy", self.matriz_v)
        np.save(f"pesos_camada_escondida_saida_{self.nome_rede}_final.npy", self.matriz_w)

        # ---------- Salvando normas L2 ----------
        with open(f"normas_l2_pesos_{self.nome_rede}.txt", "w") as f:
            f.write("Norma L2 dos pesos V (entrada -> escondida):\n")
            f.write(",".join([f"{v:.6f}" for v in normas_l2_v]) + "\n")
            f.write("Norma L2 dos pesos W (escondida -> saída):\n")
            f.write(",".join([f"{w:.6f}" for w in normas_l2_w]) + "\n")

        # ---------- Gráfico dos erros----------
        plt.plot(vetor_erro_medio, label="Treinamento")
        plt.plot(vetor_erro_medio_validacao, label="Validação")
        plt.xlabel("Épocas")
        plt.ylabel("Erro Quadrático Médio")
        plt.title("Erro por Época com Parada Antecipada")
        plt.legend()
        plt.grid(True)
        # Nome do arquivo com número do fold (se fornecido)
        if fold is not None:
            nome_arquivo = f"grafico_PA_erro_fold_{fold}.png"
        else:
            nome_arquivo = f"grafico_PA_erro_{self.nome_rede}.png"
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            plt.close()

        # ---------- Gráfico das normas L2 ----------
        plt.figure(figsize=(10, 5))
        plt.plot(normas_l2_v, label="Norma L2 - Pesos V")
        plt.plot(normas_l2_w, label="Norma L2 - Pesos W")
        plt.xlabel("Épocas")
        plt.ylabel("Norma L2")
        plt.title("Convergência dos Pesos da MLP (Norma L2)")
        plt.legend()
        plt.grid(True)
        if fold is not None:
            nome_arquivo = f"grafico_PA_Norma_Pesos_fold_{fold}.png"
        else:
            nome_arquivo = f"grafico_PA_Norma_Pesos_{self.nome_rede}.png"
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            plt.close()

    def treinamento_erro_minimo(self, epocas, matriz_X, matriz_target, erro_minimo, fold=None):           # treinamento com parada utilizando limite do valor do erro quadrático médio por época
        vetor_erro_medio = []
        vetor_erro_medio.append(1e10)

        # Vetores para acompanhar a convergência dos pesos
        normas_l2_v = []
        normas_l2_w = []

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

            # ---------- Norma L2 dos pesos ----------
            norma_v = np.linalg.norm(self.matriz_v)
            norma_w = np.linalg.norm(self.matriz_w)
            normas_l2_v.append(norma_v)
            normas_l2_w.append(norma_w)

        # print("Vetor Y final de treino:", Y)

        plt.plot(vetor_erro_medio)
        plt.xlabel("Épocas")
        plt.ylabel("Erro Quadrático Médio")
        plt.title("Erro por Época")
        min_erro = min(vetor_erro_medio)
        max_erro = max(vetor_erro_medio)
        plt.yticks(np.linspace(min_erro, max_erro, 20))  # 20 divisões entre mínimo e máximo
        plt.grid(True)
        if fold is not None:
            nome_arquivo = f"grafico_erro_minimo_fold_{fold}.png"
        else:
            nome_arquivo = f"grafico_erro_mínimo{self.nome_rede}.png"
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            plt.close()

        # ---------- Salvando os pesos ----------

        np.save(f"pesos_camada_entrada_para_escondida_{self.nome_rede}_final.npy", self.matriz_v)
        np.save(f"pesos_camada_escondida_saida_{self.nome_rede}_final.npy", self.matriz_w)

        # ---------- Salvando normas L2 ----------
        with open(f"normas_l2_pesos_{self.nome_rede}.txt", "w") as f:
            f.write("Norma L2 dos pesos V (entrada -> escondida):\n")
            f.write(",".join([f"{v:.6f}" for v in normas_l2_v]) + "\n")
            f.write("Norma L2 dos pesos W (escondida -> saída):\n")
            f.write(",".join([f"{w:.6f}" for w in normas_l2_w]) + "\n")

        # ---------- Gráfico das normas L2 ----------
        plt.figure(figsize=(10, 5))
        plt.plot(normas_l2_v, label="Norma L2 - Pesos V")
        plt.plot(normas_l2_w, label="Norma L2 - Pesos W")
        plt.xlabel("Épocas")
        plt.ylabel("Norma L2")
        plt.title("Convergência dos Pesos da MLP (Norma L2)")
        plt.legend()
        plt.grid(True)
        if fold is not None:
            nome_arquivo = f"grafico_erro_minimo_Norma_Pesos_fold_{fold}.png"
        else:
            nome_arquivo = f"grafico_erro_minimo_Norma_Pesos_{self.nome_rede}.png"
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            plt.close()

    def teste(self, matriz_X_teste, matriz_target_teste, exibir_matriz_confusao=False, nome_arquivo_matriz_confusao=None):        # função que testa a rede
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
            self.func_matriz_confusao(matriz_confusao, nome_arquivo=nome_arquivo_matriz_confusao)

        return acuracia, matriz_confusao
