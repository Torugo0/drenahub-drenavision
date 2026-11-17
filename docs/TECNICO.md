# Guia técnico

## Componentes

O DrenaVision faz triagem visual de três classes: resíduos/material visível
(`obstrucao_visivel`), água visível (`agua_acumulada`, parada ou em movimento) e animal visível
(`animal_visivel`, principalmente roedores nesta versão).

```text
Foto ──→ FastAPI ──→ YOLO customizado + SegFormer água ──→ caixas/máscara + evento
              ↑                                  ↓
          interface ←──────────────────── resultado visual
              │
Vídeo ────────┴──→ análise por quadro + rastreamento ──→ MP4 + resumo

Fontes → conversão → validação → treino → avaliação no teste separado
```

| Código | Responsabilidade |
|---|---|
| `drenavision/api/` | valida uploads e expõe o contrato HTTP |
| `drenavision/core/` | configuração, esquemas, serviço e histórico SQLite |
| `drenavision/data/` | download, conversão YOLO e validação de integridade |
| `drenavision/inference/` | treino, avaliação, detecção, vídeo e rastreamento |
| `drenavision/ui/` | experiência do operador e apresentação dos resultados |

O Streamlit usa a API para imagens e histórico. O vídeo executa o mesmo pacote
de inferência em processo local, com o mesmo modelo, limiares e banco. Não há
respostas simuladas. Sem checkpoint customizado compatível, a análise fica
indisponível e o diagnóstico da API informa o problema. Um segmentador de água
configurado, mas ausente/incompatível, também torna a análise indisponível;
não existe troca silenciosa para um resultado incompleto.

A imagem e o vídeo compartilham a política `inference/runtime.py`. O YOLO usa
640, confiança 0,35 e limiar de animal 0,60 escolhido na validação. O SegFormer
B1 usa 512, probabilidade de pixel 0,95 e área mínima de componente 5%, sem
dilatação ou preenchimento de espaços. As coordenadas são devolvidas no tamanho
original. Somente propostas existentes são mantidas na fusão de água; não se
cria um retângulo artificial pela união das caixas. Os pesos/configurações e
esses parâmetros entram no hash do perfil.

`water.py` carrega safetensors localmente, verifica hashes e nomes das classes
ADE20K e não executa código remoto. A preparação separada
`prepare_water_model.py` usa revisão oficial fixa e conversão `weights_only=True`.
Os pesos NVIDIA são restritos à pesquisa/avaliação não comercial.

## API

Inicie pelo `iniciar.ps1`. O contrato completo e testável fica no
[Swagger local](http://127.0.0.1:8000/docs) e no
[OpenAPI](http://127.0.0.1:8000/openapi.json).

| Método e endpoint | O que faz | Entrada principal |
|---|---|---|
| `GET /health` | verifica serviço, banco e modelo | nenhuma |
| `GET /vision/classes` | lista as três classes e nomes em português | nenhuma |
| `POST /vision/predict` | analisa uma imagem e registra o resultado | `file` JPEG/PNG; `point_id` opcional |
| `GET /vision/events` | consulta o histórico | `limit`, `offset`, `point_id` |
| `GET /vision/events/{event_id}` | consulta uma análise específica | identificador retornado pela análise |
| `GET /vision/evidence/{event_id}` | obtém a evidência derivada | identificador da análise |

`/v1/classes`, `/v1/predict/image` e `/v1/events` são aliases legados ocultos
do Swagger. Novas integrações devem usar `/vision`.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health

curl.exe -X POST http://127.0.0.1:8000/vision/predict `
  -F "point_id=garagem-bloco-a-01" `
  -F "file=@C:\dados\teste.jpg;type=image/jpeg"

Invoke-RestMethod "http://127.0.0.1:8000/vision/events?limit=20&offset=0"
```

### Contrato de resposta

Uma predição retorna `event_id`, `point_id`, horário UTC, tipo de entrada,
versão do modelo, `detections`, recomendação de inspeção, status, URI lógica da
evidência e tempo de inferência. Cada detecção inclui:

| Campo | Significado |
|---|---|
| `class_id` / `class_name` | ID e nome estáveis da classe |
| `confidence` | escore do detector ou média da probabilidade de água nos pixels aceitos; não é probabilidade de risco |
| `bounding_box` | `[x_min, y_min, x_max, y_max]` em pixels |
| `track_id` | ID de rastreamento, quando disponível |
| `frame_index` / `timestamp_seconds` | quadro e instante da detecção no vídeo |

Uma imagem válida sem detecções retorna HTTP 200 e `detections: []`. O valor
`inspection_recommended` é verdadeiro quando existe ao menos uma detecção
aceita pelo limiar. Isso é uma regra de encaminhamento; não mede risco de
enchente ou gravidade hidráulica.

`point_id` aceita até 80 caracteres. Na listagem, `limit` aceita de 1 a 500
(padrão 50) e `offset` começa em zero. Caminhos absolutos do servidor não são
expostos nas respostas.

### Erros esperados

| HTTP | Motivo | Providência |
|---:|---|---|
| 400 | imagem vazia, corrompida ou com dimensões inseguras | reexportar JPEG/PNG |
| 404 | evento ou evidência inexistente | conferir o identificador |
| 413 | arquivo acima do limite | reduzir o arquivo |
| 415 | tipo de imagem não permitido | enviar JPEG/PNG |
| 422 | parâmetros inválidos | conferir o formulário no Swagger |
| 503 | modelo ausente ou incompatível | obter/configurar o checkpoint |

Erros explícitos usam `{"detail": "mensagem"}`. Erros HTTP 422 seguem o
formato de validação do FastAPI. `/health` retorna `status: "degraded"` e
`model_loaded: false` quando o modelo não está disponível.

## Vídeo, marcações e histórico

O processador analisa os quadros com o checkpoint real e desenha as caixas
correspondentes às detecções aceitas. A contagem usa o maior valor entre IDs
únicos rastreados e pico simultâneo por classe, evitando somar a mesma
detecção em todos os quadros. Oclusões podem provocar troca de IDs; a contagem
não comprova quantidade de objetos físicos. A interface apresenta presença do
sinal por classe nos vídeos; o relatório técnico preserva as contagens.

A etapa temporal de vídeo exige uma sequência de pelo menos 0,5 s com água
marcada para confirmar água no vídeo. Quando esse critério falha, a exportação
é redesenhada dos quadros originais com as demais classes. As propostas brutas
de água permanecem no relatório. O perfil de vídeo registra esse parâmetro e
o perfil-base de imagem. As 179 imagens de teste não avaliam essa regra temporal.

Cada execução produz um vídeo anotado e um JSON com hash da entrada/modelo,
resolução, FPS, quadros processados, contagens e detecções. Quando o banco é
informado, registra também um evento-resumo com detecções representativas.
O vídeo completo permanece em `resultados/local/videos/`.

O SQLite guarda identificador, local técnico, UTC, versão e detecções. Para
imagem, a evidência derivada é JPEG anotado; para vídeo, JSON resumido. O
upload original da API não é persistido e a cópia temporária de vídeo é
removida após a análise. Resultados e histórico operacional ficam fora do Git.

## Configuração

Os caminhos relativos partem da raiz do repositório. Variáveis de ambiente
usam o prefixo `DRENAVISION_`; o modelo de configuração é `.env.example`.

| Variável | Padrão |
|---|---|
| `DRENAVISION_MODEL_PATH` | `resultados/local/models/best.pt` |
| `DRENAVISION_DATABASE_PATH` | `resultados/local/events/drenavision.db` |
| `DRENAVISION_EVIDENCE_DIR` | `resultados/local/events/evidence` |
| `DRENAVISION_CONFIDENCE_THRESHOLD` | `0.35` |
| `DRENAVISION_IOU_THRESHOLD` | `0.45` |
| `DRENAVISION_DEVICE` | `auto`: CUDA quando disponível; senão CPU |
| `DRENAVISION_IMAGE_SIZE` | `640` |
| `DRENAVISION_ANIMAL_CONFIDENCE_THRESHOLD` | `0.60` |
| `DRENAVISION_WATER_MODEL_PATH` | `resultados/local/models/water_segformer_b1_ade` |
| `DRENAVISION_WATER_PIXEL_CONFIDENCE` | `0.95` |
| `DRENAVISION_WATER_MIN_AREA_FRACTION` | `0.05` |
| `DRENAVISION_WATER_MODE` | `augment` |
| `DRENAVISION_VIDEO_WATER_CONFIRMATION_SECONDS` | `0.5` |
| `DRENAVISION_MAX_IMAGE_BYTES` | 12 MiB |
| `DRENAVISION_MAX_VIDEO_BYTES` | 500 MiB |
| `DRENAVISION_MAX_IMAGE_PIXELS` | 40 milhões |

## Limites do produto

O modelo não estima profundidade, vazão, causa do problema ou probabilidade
de alagamento. Resíduo próximo ao ralo não comprova entupimento. A classe de
animal não identifica espécie com confiabilidade validada. A ausência de
detecção não significa local seguro.

Reflexos, piso molhado, sombras, pouca luz, câmera distante e objetos ocultos
reduzem a qualidade. Nas 119 imagens adicionais, o produto encontrou água em
33/40 positivas e gerou dois alertas em dez negativos revisados. Animal foi
encontrado em 46/69 positivas; localização por caixa tem referência parcial.
Esses resultados não são validação de campo. As fontes ainda têm pouca cobertura
da drenagem pretendida. Resultados completos e erros conhecidos
estão em [MODEL_CARD.md](MODEL_CARD.md).

Uma pessoa deve confirmar o que foi marcado e decidir o encaminhamento. O
limiar modifica o equilíbrio entre objetos perdidos e alertas indevidos;
qualquer ajuste deve usar validação independente e ser registrado.

## Dados pessoais e licenças

Use nome técnico do local, sem nome de morador, telefone ou endereço. Imagens
podem conter pessoas e placas; uma operação real exige minimização, controle
de acesso, prazo de retenção e definição da base legal aplicável.

O serviço local é para demonstração controlada. Uma publicação operacional
precisa de autenticação, TLS, limites de requisição/concorrência e política de
retenção. SQLite e arquivos locais devem ser revistos para múltiplas instâncias.

O código é AGPL-3.0-only. Licenças do código, pesos e imagens são independentes.
O iRodent é tratado como CC BY-NC por divergência entre o registro e os metadados
de origem. Consulte [os créditos](../THIRD_PARTY_NOTICES.md) antes de reutilizar
material comercialmente.
