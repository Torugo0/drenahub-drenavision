# Arquitetura do DrenaVision

## Objetivo e fronteira

O DrenaVision transforma imagens e vídeos de pontos de drenagem em evidências
estruturadas para triagem humana. O sistema detecta somente:

| ID | Classe | Interpretação permitida |
|---:|---|---|
| 0 | `obstrucao_visivel` | material visível que pode justificar inspeção |
| 1 | `agua_acumulada` | região de água visível, sem estimar profundidade |
| 2 | `animal_visivel` | animal presente, sem afirmar obstrução ou risco sanitário |

Previsão de alagamento, cálculo hidráulico, identificação de espécie, estimativa
de profundidade e decisão automática de manutenção estão fora do escopo.

## Visão de componentes

```text
Mendeley / Zenodo
        |
        v
app.data.downloader ----> datasets/raw/                 (não versionado)
        |
        v
app.data.converter -----> dataset YOLO + manifest CSV  (dados não versionados)
        |
        v
app.data.validator -----> relatório de integridade
        |
        v
app.inference.training -> artifacts/models/best.pt      (não versionado)
        |
        +-----------------------> app.inference.evaluation
        |                                      |
        |                                      v
        |                           métricas + matriz + erros
        v
app.inference.predictor
        |
        +------------> app.api.main ---------> SQLite + evidências
        |                    ^
        |                    |
        +------------> app.ui.streamlit_app
        |
        +------------> processamento de vídeo + tracking
```

### Dados

- `app/data/downloader.py` baixa somente fontes previamente catalogadas,
  verifica hash/tamanho quando disponível, retoma downloads e impede extração
  fora do diretório de destino;
- `app/data/converter.py` mapeia anotações de origem para as três classes,
  aplica splits determinísticos com seed 42 e gera um manifesto por imagem;
- `app/data/validator.py` verifica YAML, classes, imagens, labels YOLO,
  coordenadas, pares imagem-label, duplicatas exatas, duplicatas perceptuais e
  grupos que atravessam splits.

Arquivos brutos e processados ficam fora do Git. Manifestos, metadados de fonte
e código de aquisição são a base da reprodutibilidade.

### Treino e avaliação

O treino parte de um detector nano pré-treinado oficialmente suportado e faz
fine-tuning sobre as três classes. `configs/train.yaml` é a configuração
versionada; a execução preserva parâmetros efetivos, versões, dispositivo,
hashes, horários e caminho dos checkpoints.

A avaliação lê exclusivamente o split `test`, escreve métricas globais e por
classe, matriz de confusão e amostras de falso positivo/falso negativo. Nenhum
valor padrão do código é apresentado como resultado experimental.

### Inferência e API

`YoloPredictor` é a única implementação de produção do contrato de inferência.
Na inicialização ele confere a existência do checkpoint e seu mapa de classes.
Sem checkpoint customizado, a aplicação entra em estado degradado; ela não cai
silenciosamente para pesos COCO.

A API expõe:

- `GET /health`;
- `GET /vision/classes`;
- `POST /vision/predict`;
- `GET /vision/events`;
- `GET /vision/events/{event_id}`;
- `GET /vision/evidence/{event_id}`.

Aliases legados sob `/v1` permanecem ocultos do OpenAPI somente para
compatibilidade; o contrato canônico e documentado usa `/vision`.

### Interface e vídeo

O Streamlit chama a API para imagens e histórico. Para vídeo, invoca o mesmo
pacote de inferência por processo local, usando a mesma configuração de modelo,
limiares, dispositivo, banco e diretório de evidências. O tracking usa o maior
valor entre IDs únicos e pico simultâneo por classe. Ao concluir, persiste um
evento-resumo `input_type=video` com ponto, UTC, versão, detecções
representativas, frame/timestamp e evidência JSON; não soma todos os quadros.

## Sequência de uma imagem

```text
usuário
  | POST multipart (JPEG/PNG + point_id)
  v
FastAPI -- tipo/tamanho/pixels --> decodificação segura
  |                                    |
  | erro 400/413/415                   v
  +------------------------------ YoloPredictor
                                         |
                               detections xyxy + confiança
                                         |
                                         v
                              PredictionService
                               /             \
                      evidência anotada    evento SQLite
                               \             /
                                resposta JSON 200
```

Uma imagem válida sem objetos-alvo gera evento com lista vazia e HTTP 200.
`inspection_recommended` é uma regra determinística (`detections` não vazia), não
uma probabilidade de enchente.

## Persistência

O SQLite armazena identificador da ocorrência, `point_id`, horário UTC, versão
do modelo, detecções e URI lógica da evidência. Caminhos absolutos não fazem
parte do contrato externo. A imagem original enviada à API não é persistida.
Para imagens, a evidência derivada é JPEG; para vídeos, a evidência do evento é
um JSON resumido com contagens e detecções representativas. Ambos são locais e
ignorados pelo Git.

Para implantação multi-instância, SQLite e arquivos locais precisam ser
substituídos por armazenamento compartilhado/transacional. Essa migração não é
necessária para a demonstração acadêmica e não deve ser confundida com estado
de produção.

## Configuração

Variáveis usam o prefixo `DRENAVISION_`. As principais são:

| Variável | Padrão | Finalidade |
|---|---|---|
| `DRENAVISION_MODEL_PATH` | `artifacts/models/best.pt` | checkpoint customizado |
| `DRENAVISION_DATABASE_PATH` | `artifacts/events/drenavision.db` | histórico local |
| `DRENAVISION_EVIDENCE_DIR` | `artifacts/events/evidence` | JPEGs anotados e resumos JSON |
| `DRENAVISION_CONFIDENCE_THRESHOLD` | `0.35` | limiar de serviço |
| `DRENAVISION_IOU_THRESHOLD` | `0.45` | NMS/inferência |
| `DRENAVISION_DEVICE` | `auto` | CUDA, MPS ou CPU |
| `DRENAVISION_MAX_IMAGE_BYTES` | 12 MiB | proteção de upload |
| `DRENAVISION_MAX_IMAGE_PIXELS` | 40 milhões | proteção de decodificação |

O arquivo `.env.example` contém valores de desenvolvimento sem segredo.

## Segurança e falhas esperadas

| Condição | Comportamento esperado |
|---|---|
| checkpoint ausente/incompatível | `/health` degradado; predição HTTP 503 |
| MIME não permitido | HTTP 415 |
| imagem acima do limite | HTTP 413 |
| bytes inválidos ou decompression bomb | HTTP 400 |
| zero detecções | HTTP 200 e lista vazia |
| dataset ausente/inválido | treino/avaliação encerram antes de criar evidência enganosa |
| codec de vídeo indisponível | CLI falha com mensagem; não produz arquivo fictício |

## Reprodutibilidade

O protocolo fixa seed 42, mantém grupos de origem em um único split e registra
hashes de dados/configuração/pesos. O determinismo absoluto em GPU não é
garantido por todas as operações CUDA; por isso o ambiente e o dispositivo
fazem parte da evidência. Consulte [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Decisões que exigem revisão antes de produção

- política de retenção e descarte de evidências;
- autenticação, autorização, TLS, rate limiting e auditoria;
- banco e object storage compartilhados;
- monitoramento de drift e revalidação periódica;
- revisão de vieses e desempenho por local/câmera/classe;
- licenciamento do detector e dos datasets para uso comercial;
- procedimento humano de resposta e contestação.

