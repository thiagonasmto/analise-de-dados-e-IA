import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport
from sklearn.preprocessing import MinMaxScaler
sns.set()

# Função para plotar histogramas de uma ou mais colunas
def plotar_histograma(dados, colunas, unico_grafico=False):
    """
    Plota histogramas para uma ou mais colunas de um DataFrame.

    :param dados: DataFrame contendo os dados.
    :param colunas: Lista de nomes de colunas a serem plotadas.
    :param unico_grafico: Se True, plota todas as colunas no mesmo gráfico; caso contrário, plota separadamente.
    """
    if unico_grafico:
        plt.figure(figsize=(10, 6))
        for coluna in colunas:
            sns.histplot(dados[coluna], kde=False, label=coluna, element="step")
        plt.title('Histograma de múltiplas colunas')
        plt.xlabel('Valor')
        plt.ylabel('Frequência')
        plt.legend()
        plt.show()
    else:
        for coluna in colunas:
            plt.figure(figsize=(10, 6))
            sns.histplot(dados[coluna], kde=False)
            plt.title(f'Histograma da coluna {coluna}')
            plt.xlabel(coluna)
            plt.ylabel('Frequência')
            plt.show()

# Função para plotar gráficos KDE de uma ou mais colunas
def plotar_kde(dados, colunas, unico_grafico=False):
    """
    Plota gráficos de Densidade Kernel (KDE) para uma ou mais colunas.

    :param dados: DataFrame contendo os dados.
    :param colunas: Lista de colunas a serem plotadas.
    :param unico_grafico: Se True, plota todas as colunas em um único gráfico; caso contrário, plota separadamente.
    """
    if unico_grafico:
        plt.figure(figsize=(10, 6))
        for coluna in colunas:
            sns.kdeplot(dados[coluna], shade=True, label=coluna)
        plt.title('Gráfico KDE de múltiplas colunas')
        plt.xlabel('Valor')
        plt.ylabel('Densidade')
        plt.legend()
        plt.show()
    else:
        for coluna in colunas:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(dados[coluna], fill=True)
            plt.title(f'Gráfico KDE da coluna {coluna}')
            plt.xlabel(coluna)
            plt.ylabel('Densidade')
            plt.show()

def plotar_fdp_kde(dados, pares_colunas, unico_grafico=False):
    """
    Plota a Densidade de Probabilidade (FDP) via KDE para pares de colunas.

    :param dados: DataFrame contendo os dados.
    :param pares_colunas: Lista de pares de colunas para o eixo X e Y.
    :param unico_grafico: Se True, plota todos os pares no mesmo gráfico; caso contrário, plota separadamente.
    """
    if unico_grafico:
        plt.figure(figsize=(10, 6))
        for coluna_x, coluna_y in pares_colunas:
            sns.kdeplot(x=dados[coluna_x], y=dados[coluna_y], fill=True, label=f'{coluna_x} vs {coluna_y}')
        plt.title('Densidade de Probabilidade (FDP) via KDE de múltiplos pares de colunas')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.show()
    else:
        for coluna_x, coluna_y in pares_colunas:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(x=dados[coluna_x], y=dados[coluna_y], fill=True)
            plt.title(f'Densidade de Probabilidade (FDP) via KDE das colunas {coluna_x} e {coluna_y}')
            plt.xlabel(coluna_x)
            plt.ylabel(coluna_y)
            plt.show()

# Função para plotar jointplot com estilo 'white' e tipo 'hex'
def plotar_jointplot_hex(dados, coluna_x, coluna_y, titulo=None, referencia=None):
    """
    Plota um gráfico jointplot do tipo 'hex' com as opções de título e referência.

    :param dados: DataFrame contendo os dados.
    :param coluna_x: Nome da coluna para o eixo x.
    :param coluna_y: Nome da coluna para o eixo y.
    :param titulo: Título opcional do gráfico.
    :param referencia: Texto de referência opcional para ser exibido abaixo do gráfico.
    """
    with sns.axes_style('white'):
        sns.jointplot(x=coluna_x, y=coluna_y, data=dados, kind='hist', cmap='viridis')
        
        # Adiciona o título, se fornecido
        if titulo:
            plt.title(titulo)
        
        # Exibe o gráfico
        plt.show()
        
        # Adiciona a referência, se fornecida
        if referencia:
            plt.figtext(0.5, 0.05, referencia, ha='center', va="center", fontsize=10, color='black')

# Função para plotar pairplot com título e referência opcionais
def plotar_pairplot(dados, colunas=None, hue_coluna=None, tamanho=2.5, titulo=None, referencia=None): 
    """
    Plota um gráfico pairplot com as opções de título e referência.

    :param dados: DataFrame contendo os dados.
    :param colunas: Lista de colunas a serem consideradas para o pairplot. Se None, todas as colunas são usadas.
    :param hue_coluna: Nome da coluna para colorir as variáveis (opcional).
    :param tamanho: Tamanho dos gráficos (altura).
    :param titulo: Título opcional do gráfico.
    :param referencia: Texto de referência opcional para ser exibido abaixo do gráfico.
    """
    if colunas is not None:
        dados = dados[colunas]
    
    sns.pairplot(dados, hue=hue_coluna, height=tamanho)
    
    # Adiciona o título, se fornecido
    if titulo:
        plt.title(titulo)
    
    # Exibe o gráfico
    plt.show()
    
    # Adiciona a referência, se fornecida
    if referencia:
        plt.figtext(0.5, 0.05, referencia, ha='center', va="center", fontsize=10, color='black')

def plotar_boxplot(dados, coluna_x, coluna_y, hue_coluna=None, estilo='ticks'):
    """
    Plota um gráfico de BoxPlot para as colunas especificadas de um DataFrame.

    :param dados: DataFrame contendo os dados.
    :param coluna_x: Nome da coluna a ser usada no eixo x.
    :param coluna_y: Nome da coluna a ser usada no eixo y.
    :param hue_coluna: Nome da coluna para ser usada como base para diferenciar as cores dos boxplots. Pode ser None.
    :param estilo: Estilo dos eixos a ser aplicado ao gráfico.
    :param tamanho: Tupla representando o tamanho da figura (largura, altura).
    """
    with sns.axes_style(style=estilo):
        g = sns.catplot(x=coluna_x, y=coluna_y, hue=hue_coluna, data=dados, kind="box")
        g.set_axis_labels(coluna_x, coluna_y)
        plt.show()

def plotar_pairgrid(dados, colunas, hue_coluna=None, palette='RdBu_r', alpha=0.8):
    """
    Plota uma grade de gráficos de dispersão (PairGrid) para as colunas especificadas de um DataFrame,
    colorindo os pontos de acordo com a coluna de categorias (hue_coluna).

    :param dados: DataFrame contendo os dados.
    :param colunas: Lista de nomes de colunas a serem incluídas no PairGrid.
    :param hue_coluna: (Opcional) Nome da coluna para diferenciar as cores dos pontos no gráfico.
    :param palette: Paleta de cores a ser usada para as categorias especificadas em hue_coluna.
    :param alpha: Transparência dos pontos no gráfico.
    """
    # Criação da Grade de Pares
    g = sns.PairGrid(dados, vars=colunas, hue=hue_coluna, palette=palette)
    g.map(plt.scatter, alpha=alpha)
    g.add_legend()
    plt.show()

def plotar_heatmap_correlacao(dados, colunas, annot=True, cmap='coolwarm'):
    """
    Calcula a matriz de correlação para as colunas selecionadas e plota um heatmap dessa matriz.

    :param dados: DataFrame contendo os dados.
    :param colunas: Lista de nomes de colunas para calcular a matriz de correlação.
    :param annot: Se True, exibe os valores de correlação no heatmap.
    :param cmap: Paleta de cores a ser usada no heatmap.
    """
    # Cálculo da matriz de correlação
    matriz_correlacao = dados[colunas].corr()
    
    # Plot do heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_correlacao, annot=annot, cmap=cmap, fmt=".2f")
    plt.title('Heatmap da Matriz de Correlação')
    plt.show()

def plotar_violinplot(dados, coluna_x, coluna_y, hue_coluna=None, split=False, inner="quartile", palette=["lightblue", "lightpink"], estilo=None, titulo=None, fonte_dados=None):
    """
    Plota um gráfico de violino para visualizar distribuições de uma variável contínua (coluna_y)
    em relação a uma variável categórica (coluna_x), com a opção de dividir por uma terceira variável categórica (hue_coluna).

    :param dados: DataFrame contendo os dados.
    :param coluna_x: Nome da coluna categórica para o eixo x.
    :param coluna_y: Nome da coluna contínua para o eixo y.
    :param hue_coluna: Nome da coluna para dividir os violinos (se aplicável). Pode ser None.
    :param split: Se True, divide os violinos ao longo da categoria em hue_coluna.
    :param inner: Especifica o tipo de representação interna nos violinos. Valores comuns são 'box', 'quartile', 'point', 'stick', ou None.
    :param palette: Paleta de cores a ser usada para os violinos.
    :param estilo: Estilo dos eixos a ser aplicado ao gráfico (exemplo: 'white', 'dark', 'ticks', etc.).
    :param titulo: Título opcional para o gráfico.
    :param fonte_dados: Texto opcional para a fonte dos dados, a ser exibido abaixo do gráfico.
    """
    with sns.axes_style(style=estilo):
        plt.figure(figsize=(10, 6))
        sns.violinplot(x=coluna_x, y=coluna_y, hue=hue_coluna, data=dados, split=split, inner=inner, palette=palette)
        
        if titulo:
            plt.title(titulo, fontsize=16)
        
        if fonte_dados:
            plt.figtext(0.5, 0.01, fonte_dados, ha='center', fontsize=10, color='black')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

def gerar_relatorio_profiling(dados, titulo="Pandas Profiling Report", arquivo_saida=None):
    """
    Gera um relatório exploratório de um DataFrame utilizando pandas profiling.

    :param dados: DataFrame contendo os dados a serem analisados.
    :param titulo: Título do relatório.
    :param arquivo_saida: Nome do arquivo de saída para salvar o relatório em formato HTML. Se None, não salva.
    """
    # Gerando o relatório
    profile = ProfileReport(dados, title=titulo, explorative=True)

    # Exibindo o relatório no notebook (não funcional fora de um ambiente interativo)
    profile.to_notebook_iframe()

    # Salvando o relatório em um arquivo HTML, se especificado
    if arquivo_saida:
        profile.to_file(output_file=arquivo_saida)

def normalizar_dados(dataframe, colunas=None):
    """
    Normaliza os dados de um DataFrame usando o MinMaxScaler.

    :param dataframe: DataFrame contendo os dados.
    :param colunas: Lista de colunas a serem normalizadas. Se None, normaliza todas as colunas numéricas.
    :return: DataFrame com as colunas normalizadas.
    """
    # Se colunas não forem especificadas, selecionar todas as colunas numéricas
    if colunas is None:
        colunas = dataframe.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Inicializar o MinMaxScaler
    scaler = MinMaxScaler()
    
    # Aplicar o scaler nas colunas especificadas
    dataframe[colunas + '_normalizada'] = scaler.fit_transform(dataframe[colunas])
    
    return dataframe

# Exemplo de uso das funções
if __name__ == "__main__":
    caminho_arquivo = './caminho'
    dados = pd.read_csv(caminho_arquivo)

    # Exemplo de chamadas de função
    # colunas = ['col1','col2','col3']
    # pares_colunas = [('col1', 'col2')]
    
    # Plota gráficos usando as funções definidas
    # plotar_histograma(dados, colunas)
    # plotar_kde(dados, colunas)
    # plotar_histograma_kde(dados, colunas)
    # plotar_rug(dados, colunas)
    # plotar_fdp_kde(dados, pares_colunas)
    # plotar_jointplot_hex(dados, 'col1', 'col2', titulo='title1', referencia='Fonte dos dados')
    # plotar_pairplot(dados, colunas=colunas, hue_coluna='col1', tamanho=2.5)
    # plotar_pairgrid(dados, colunas=colunas, hue_coluna='col1', palette='RdBu_r', alpha=0.7)
    # plotar_violinplot(dados, coluna_x='col1', coluna_y='col2', hue_coluna='col3', split=True, inner="quartile", palette=['lightblue', 'lightyellow'], estilo='dark', fonte_dados='Fonte dos dados')

    #----------------------------------------------------------------------------------------------------------------------------------------------------

    # Exemplo de uso da função plotar_heatmap_correlacao
    # plotar_heatmap_correlacao(dados, colunas=colunas)

    #----------------------------------------------------------------------------------------------------------------------------------------------------

    # Gerar relatório de profiling e salvar como HTML
    # gerar_relatorio_profiling(dados, titulo="Relatório de Profiling", arquivo_saida="Relatório do dataset.html")

    #----------------------------------------------------------------------------------------------------------------------------------------------------

    #Normalização de dados
    # dados_normalizados = normalizar_dados(dados)
    # print(dados_normalizados.head())
    # dados_normalizados.to_csv('dados_normalizados.csv', index=False)

    #----------------------------------------------------------------------------------------------------------------------------------------------------

    # Dev. Responsável: Thiago Lopes - thiagonasmento20@gmail.com