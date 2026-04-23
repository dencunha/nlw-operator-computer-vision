# 🧠 LeNet-5 — Classificação de Dígitos MNIST

![Versão do Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

Este projeto implementa a arquitetura clássica **LeNet-5** (Yann LeCun, 1998) utilizando **PyTorch**. O objetivo principal é a classificação de dígitos manuscritos do dataset **MNIST**, aplicando melhorias modernas à arquitetura original para otimizar o desempenho.

[⬅️ Voltar para a Raiz do Projeto](../README.md)

---

## 🚀 Sobre o Projeto

Este projeto faz parte da trilha de **Visão Computacional** da **NLW (Next Level Week)** da **Rocketseat**.
A implementação no notebook cobre o ciclo completo de Deep Learning:
- Processamento e visualização de dados.
- Definição da arquitetura da Rede Neural Convolucional (CNN).
- Treinamento e ajuste de hiperparâmetros.
- Avaliação e análise visual de erros.

---

## 🛠️ Arquitetura LeNet-5 (Modificada)

A rede segue a essência da LeNet-5, mas com ajustes contemporâneos:
- **Camadas Convolucionais**: 3 camadas (C1, C3, C5).
- **Ativação**: Substituição de Sigmoid por **ReLU** para mitigar o problema do gradiente sumindo.
- **Pooling**: Substituição de Subsampling por **Max Pooling** (2x2).
- **Camadas Densas**: 2 camadas totalmente conectadas (F6 e Output).

---

## 📊 Visualizações & Análises

O projeto oferece uma visão clara do que acontece "sob o capô" da rede:
- **Filtros Iniciais**: Visualização dos pesos da primeira camada convolucional logo após a inicialização.
- **Feature Maps**: Análise das ativações da primeira camada para entender quais características (bordas, formas) a rede está extraindo.
- **Imagens Classificadas Incorretamente**: Uma galeria automática de imagens onde o modelo falhou, ajudando a identificar padrões de confusão (ex: distinguir 4 de 9).

---

## 📁 Estrutura do Projeto

```text
lenet/
├── data/               # Dataset MNIST (baixado automaticamente)
├── weights/            # Pesos do modelo treinado (.pth)
├── lenet5.ipynb        # Notebook principal com o código
├── pyproject.toml      # Gerenciamento de dependências (uv)
└── README.md           # Você está aqui!
```

---

## ⚙️ Como Executar

### 1. Pré-requisitos
Este projeto utiliza o gerenciador de pacotes **uv**. Se você ainda não o tem instalado, siga as instruções em [astral.sh/uv](https://astral.sh/uv).

### 2. Instalação
Navegue até o diretório e sincronize as dependências:
```bash
cd lenet
uv sync
```

### 3. Executando o Notebook
Inicie seu ambiente Jupyter favorito (VS Code, Jupyter Lab, etc.) e execute as células em `lenet5.ipynb`.

O notebook guiará você por:
1. Instanciação do modelo.
2. Download e inspeção do MNIST.
3. Ciclo de treinamento (5 épocas por padrão).
4. Teste de acurácia.
5. Exportação dos pesos para a pasta `weights`.

---

## 🧪 Resultados Esperados
O modelo treinado deve atingir uma acurácia superior a **98%** no conjunto de teste em poucas épocas, demonstrando a eficácia da arquitetura clássica para este problema.

---

*Feito com ❤️ durante a NLW @ Rocketseat*

