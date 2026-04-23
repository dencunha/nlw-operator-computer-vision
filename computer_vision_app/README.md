# 🚀 Rockit Vision — Reconhecimento de Gestos com IA em Tempo Real

[![Rockit Vision Logo](https://img.shields.io/badge/Rockit_Vision-Hand%20Gesture%20Recognition-blue?style=for-the-badge&logo=mediamarkt)](https://github.com/athk/nlw-2026-computer-vision)
![Versão do Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastHTML](https://img.shields.io/badge/FastHTML-Web%20App-000000?style=for-the-badge&logo=fastapi)

**Rockit Vision** é um sistema de reconhecimento de gestos em alta performance e tempo real, construído utilizando **FastHTML**, **OpenCV** e **MediaPipe**. Ele utiliza modelos de ML customizados para identificar e visualizar gestos das mãos capturados diretamente pela webcam.

[⬅️ Voltar para a Raiz do Projeto](../README.md)

---

## ✨ Funcionalidades

- 🖐️ **Reconhecimento de Gestos em Tempo Real**: Detecção e rotulagem instantânea de diversos gestos manuais.
- ⚡ **Integração com WebSockets**: Comunicação de baixa latência entre o frontend e o backend alimentado por IA.
- 📊 **Monitoramento Dinâmico de Performance**: Contador de FPS em tempo real para monitorar o desempenho do sistema.
- 🎚️ **Controles Interativos**:
    - **Slider de Qualidade de Imagem**: Ajuste a resolução dos frames enviados para o servidor para otimizar a performance.
    - **Alternar Marcos (Landmarks)**: Mostre ou esconda os marcos do esqueleto da mão dinamicamente.
- 🖼️ **Feedback Visual**: Renderização em tempo real de frames processados e imagens específicas de cada gesto.

---

## 🛠️ Construído Com

- **[FastHTML](https://fasthtml.com/)**: O framework para construção de aplicações web modernas inteiramente em Python.
- **[MediaPipe](https://mediapipe.dev/)**: Para rastreamento de mãos e detecção de marcos com alta fidelidade.
- **[OpenCV](https://opencv.org/)**: Para processamento e manipulação de imagens.
- **[Scikit-Learn](https://scikit-learn.org/)**: Alimentando os modelos de classificação de gestos.
- **[uv](https://github.com/astral-sh/uv)**: Gerenciamento de pacotes Python extremamente rápido.

---

## 🚀 Como Começar

### Pré-requisitos

- **Python 3.11+**
- **uv** (Recomendado para gerenciamento de pacotes)

### Instalação

1. **Navegue até a pasta**:
   ```bash
   cd computer_vision_app
   ```

2. **Sincronize as dependências usando uv**:
   ```bash
   uv sync
   ```

### Executando a Aplicação

Inicie o servidor de desenvolvimento:

```bash
uv run app.py
```

Abra seu navegador e vá para `http://localhost:5001`.

---

## 🛠️ Download dos Modelos

Para que a aplicação funcione corretamente, é necessário baixar o modelo base do MediaPipe e colocá-lo na pasta `models/` do projeto:

- **[gesture_recognizer.task](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task)**: Modelo essencial para a extração de marcos e reconhecimento de gestos base.

Certifique-se de que o arquivo esteja em: `computer_vision_app/models/gesture_recognizer.task`


---

## 📂 Estrutura do Projeto

```text
computer_vision_app/
├── app.py              # Ponto de entrada principal do FastHTML & handlers de WebSocket
├── core/
│   ├── processor.py    # Lógica de processamento de gestos & integração com MediaPipe
│   ├── models.py       # Carregamento de modelos de ML e classes de predição
│   └── utils.py        # Utilitários de codificação/decodificação de imagem
├── models/
│   ├── gesture_model.joblib # Modelo de classificação de gestos pré-treinado
│   └── label_encoder.joblib # Encoder para os nomes dos gestos
├── assets/
│   ├── script.js       # Lógica do frontend para WebSocket & Webcam
│   ├── style.css       # Estilos da UI de alta qualidade & animações
│   └── images/         # Imagens e ícones dos gestos
├── pyproject.toml      # Dependências e metadados do projeto
└── README.md           # Você está aqui!
```

---

## ⚙️ Como Funciona

1. **Captura**: O navegador captura os frames de vídeo da webcam.
2. **Compressão**: Os frames são redimensionados com base na qualidade selecionada e enviados via **WebSockets**.
3. **Processamento**: O backend (`core/processor.py`) usa o **MediaPipe** para detectar os marcos das mãos.
4. **Predição**: Os marcos detectados são passados para um modelo **scikit-learn** para classificar o gesto.
5. **Resposta**: O frame processado (com marcos opcionais) e os nomes dos gestos são enviados de volta para o frontend.
6. **Visualização**: O frontend renderiza os dados em um Canvas HTML5 e atualiza a interface dinamicamente.

---

## 📄 Licença

Este projeto faz parte da jornada **NLW Operator**. Sinta-se à vontade para usar e modificar para fins de aprendizado.

---

*Feito com ❤️ durante a NLW @ Rocketseat*