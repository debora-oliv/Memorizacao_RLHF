# Memorização e Exposição em LLMs Alinhados

Este repositório contém a infraestrutura completa de treinamento e avaliação empírica focada na privacidade de Modelos de Linguagem de Grande Escala (LLMs). O pipeline reproduz o experimento de Carlini (2019) no regime de fine-tuning, injetando *canaries* durante o fine-tuning e avaliando a extração baseada na métrica de **Exposure**. O objetivo foi mensurar como diferentes estágios de alinhamento (SFT, DPO e PPO) impactam a retenção e o vazamento de dados sensíveis memorizados.

## Pré-requisitos

- **Python 3.10+**

- **GPU Mínima:** 1x NVIDIA Tesla T4 (15GB VRAM) ou equivalente.

- `torch`: framework base utilizado para as operações de deep learning e cálculos de tensores.

- `transformers`: utilizada para carregar, configurar e tokenizar o modelo base de linguagem (Qwen/Qwen1.5-1.8B).

- `trl`: essencial para instanciar os treinadores específicos de alinhamento (SFTTrainer, DPOTrainer, PPOTrainer e AutoModelForCausalLMWithValueHead).

- `peft`: usada para aplicar as matrizes do método LoRA (LoraConfig, PeftModel), reduzindo os parâmetros treináveis de forma eficiente.

- `bitsandbytes`: responsável por injetar a quantização em 4-bits no modelo em tempo real, evitando o estouro da placa de vídeo.

- `datasets`: utilizada para carregar, fatiar e aplicar funções de higienização nos conjuntos de dados (HuggingFaceH4/ultrachat_200k e Anthropic/hh-rlhf).

- `matplotlib` e `numpy`: empregadas na última célula do pipeline para o cálculo matemático das médias e a plotagem comparativa do gráfico de Exposure.

## Passo a Passo para Reprodução

Todo o fluxo do experimento foi consolidado e ordenado em um único Jupyter Notebook para facilitar a reprodução sequencial.

**1. Instalação dos Pacotes:** execute as primeiras células do notebook para instalar as dependências e carregar os imports. 

**2. Geração do Ground Truth (Passo 1):** rode a célula da função `generate_and_save_canaries` para criar as senhas aleatórias e anotar o ground truth automaticamente no arquivo local `canaries_ground_truth.json`.

**3. Preparação e Injeção de Dados (Passo 2):** execute a célula responsável por baixar o dataset `ultrachat_200k`. O código fatiará automaticamente os primeiros 5.000 registros para comportar os limites de RAM e injetará as strings de canary nas respostas do assistente. 

**4. Pipeline de Treinamento e Alinhamento (Passo 3):**
- **SFT**: rode a célula Configuração (a) para treinar o modelo a reconhecer as senhas no formato chat e exportar os pesos para a pasta `./sft_final_adapter.`

- **DPO**: rode a célula Configuração (b) para alinhar com dados de preferências, purificar os tensores invisíveis e salvar os ajustes em `./dpo_final_adapter.`

- **PPO**: rode a célula Configuração (c) e sua célula complementar logo abaixo para executar o treinamento de recompensas e criar o diretório para salvar o adaptador final `./ppo_final_adapter` de forma segura.

**5. Avaliação e Plotagem Final (Passo 4 e 5):** execute a célula com a arquitetura customizada da função de Log-Likelihood.  Em seguida, execute a célula de Avaliação em Lote. O script automatizado testará cada um dos três adaptadores na GPU de forma isolada, limpando a memória interna após a sondagem de Exposure.  O arquivo comparativo grafico_exposure_rlhf.png será renderizado na tela e salvo na raiz do seu projeto ao final da operação. 
