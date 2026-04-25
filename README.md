# Processo Seletivo – Intensivo Maker | AI

## Identificação do Candidato

Nome: Samuel Wagner Tiburi Silveira
Instituição: Universidade Federal do Cariri - UFCA

## Pre-requisitos

- Python 3.10 ou 3.11
- pip
- TensorFlow/Keras 2.12 ou superior
- NumPy


## Instalação

```bash
pip install -r requirements.txt
```

## Execução

```bash
# Etapa 1 – Treinamento
python train_model.py

# Etapa 2 – Otimização
python optimize_model.py
```

Após a execução, os arquivos `model.h5` e `model.tflite` serão gerados no diretório raiz do projeto.

---

## Relatório do Candidato

### Resumo da Arquitetura do Modelo

CNN composta por dois blocos convolucionais e uma camada de classificação direta:

- Conv2D(16 filtros, 3×3, ReLU) → MaxPooling2D(2×2)
- Conv2D(16 filtros, 3×3, ReLU) → MaxPooling2D(2×2)
- Flatten → Dense(10, Softmax)

O modelo totaliza 6.490 parâmetros treináveis. A ausência de camadas densas intermediárias é deliberada: a camada `Flatten → Dense(10)` possui apenas 4.010 pesos, mantendo o modelo compacto sem sacrificar a capacidade de extração de features, que fica inteiramente a cargo das convoluções.

### Técnica de Otimização do Modelo

**Dynamic Range Quantization**, aplicada via `tf.lite.Optimize.DEFAULT`. Os pesos são convertidos de float32 para int8 em tempo de conversão, reduzindo o tamanho do modelo sem exigir dataset de calibração e mantendo compatibilidade ampla com hardware embarcado.

### Resultados Obtidos

| Metrica | Valor |
|---|---|
| Acuracia (conjunto de teste) | 98.43% |
| Tamanho do modelo `.tflite` | 11.14 KB |

### Comentários Adicionais

A principal decisão técnica foi remover a camada `Dense` intermediária. Em testes, a arquitetura com `Dense(16)` entre o Flatten e a saída atingiu 98.98% de acurácia, porém com 24 KB — mais do dobro do tamanho final. A camada intermediária sozinha adicionava ~12.800 pesos (400 × 16), inviabilizando o modelo para Edge AI. A versão final delega toda a representação às convoluções e obtém um resultado competitivo com menos de 12 KB.