# Guia técnico

Referência da versão **0.5.0**, perfil de imagem congelado
`pipeline-4c5c993223219929`. API 0.5.0 ativa, 239 testes automatizados aprovados
e benchmark de 219 imagens concluído em coortes separadas (40 reservadas e
179 regressões conhecidas). A verificação funcional de fotos/vídeos foi concluída.

## Componentes

O DrenaVision faz triagem visual de três classes: resíduos/material visível
(`obstrucao_visivel`), água visível (`agua_acumulada`, parada ou em movimento) e animal visível
(`animal_visivel`, principalmente roedores nesta versão).

```text
Foto → FastAPI → runtime comum → caixas/máscara + evento → interface
Vídeo → FastAPI → runtime compartilhado por quadro + rastreamento → MP4 + resumo

runtime: YOLO customizado v3 + Grounding DINO → SigLIP 2 nos recortes → fusão 0/2
         + SegFormer B1 de água preservado → resultado visual

Fontes → conversão → validação → treino → avaliação no teste separado
```

| Código | Responsabilidade |
|---|---|
| `drenavision/api/` | valida uploads e expõe o contrato HTTP |
| `drenavision/core/` | configuração, esquemas, serviço e histórico SQLite |
| `drenavision/data/` | download, conversão YOLO e validação de integridade |
| `drenavision/inference/` | treino, avaliação, detecção, vídeo e rastreamento |
| `drenavision/ui/` | experiência do operador e apresentação dos resultados |

O Streamlit usa HTTP para imagens, vídeos e histórico. O endpoint de vídeo
reutiliza o preditor da API, compartilhando Grounding DINO, SigLIP 2 e SegFormer
entre as análises. Apenas o rastreador YOLO pequeno tem estado próprio por
vídeo, para evitar misturar IDs. A UI não inicia subprocesso com os modelos
pesados. Foto e vídeo passam pelo mesmo controle de concorrência de inferência.
Não há respostas simuladas. Um checkpoint, especialista, verificador ou segmentador
configurado que esteja ausente/incompatível torna a análise indisponível;
o diagnóstico informa o problema. Não existe troca silenciosa para um
resultado incompleto nem download de pesos durante a requisição.

A imagem e o vídeo compartilham a política `inference/runtime.py`. O YOLO v3
usa 640, confiança base/resíduos 0,35 e corte de animal 0,60, preservados.
Grounding DINO Tiny gera propostas com vocabulário fixo, arestas 640/1024,
texto 0,20 e cortes 0,30 para resíduos/0,25 para animal. O SigLIP 2 compara os
recortes com frases positivas/de fundo e exige margens de cosseno 0,04/0,08.
O contexto de 10% do recorte não altera as coordenadas de saída.

No modo `animal_replace`, o especialista complementa os resíduos do mestre
e substitui suas caixas de animal. Na supressão de duplicatas da mesma classe
com IoU 0,5, suas caixas têm prioridade. Água não participa dessa supressão de
objetos. Nenhum desses componentes externos foi retreinado nesta revisão.

O SegFormer B1 mantém 512, probabilidade de pixel 0,95 e componente mínimo de
5%, sem dilatação ou preenchimento. As coordenadas são devolvidas no tamanho
original. A fusão de água mantém propostas existentes, sem criar um retângulo
artificial de união. Pesos, processadores, frases, parâmetros e regras entram
no hash do perfil. O registro fixado está em
[frozen_candidate.json](../resultados/metrics/v4/frozen_candidate.json).

`water.py` carrega safetensors localmente, verifica hashes e nomes das classes
ADE20K e não executa código remoto. A preparação separada
`prepare_water_model.py` usa revisão oficial fixa e conversão `weights_only=True`.
Os pesos NVIDIA são restritos à pesquisa/avaliação não comercial.
`objects.py` e `object_verifier.py` usam os componentes locais Grounding DINO
e SigLIP 2, com revisões e arquivos permitidos verificados. Os preparadores
separados `prepare_object_model.py` e `prepare_verifier_model.py` obtêm
aproximadamente 2,2 GB adicionais de pesos/configuração das fontes oficiais;
eles já estão preparados neste computador. A preparação não é acionada pela API.

## API

Inicie pelo `iniciar.ps1`. O contrato completo e testável fica no
[Swagger local](http://127.0.0.1:8000/docs) e no
[OpenAPI](http://127.0.0.1:8000/openapi.json).

| Método e endpoint | O que faz | Entrada principal |
|---|---|---|
| `GET /health` | verifica serviço, banco e modelo | nenhuma |
| `GET /vision/classes` | lista as três classes e nomes em português | nenhuma |
| `POST /vision/predict` | analisa uma imagem e registra o resultado | `file` JPEG/PNG; `point_id` opcional |
| `POST /vision/predict/video` | aguarda a análise do vídeo e registra resultado/MP4 | `file` MP4/MOV/AVI/MKV com MIME correspondente; `point_id` opcional |
| `GET /vision/videos/{event_id}` | entrega o MP4 anotado de uma análise concluída | identificador retornado pela análise de vídeo |
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

curl.exe -X POST http://127.0.0.1:8000/vision/predict/video `
  -F "point_id=garagem-bloco-a-01" `
  -F "file=@C:\dados\video.mp4;type=video/mp4"

Invoke-RestMethod "http://127.0.0.1:8000/vision/events?limit=20&offset=0"
```

### Contrato de resposta

Uma predição de imagem retorna `event_id`, `point_id`, horário UTC, tipo de entrada,
versão do modelo, `detections`, recomendação de inspeção, status, URI lógica da
evidência e tempo de inferência. Cada detecção inclui:

| Campo | Significado |
|---|---|
| `class_id` / `class_name` | ID e nome estáveis da classe |
| `confidence` | escore do detector ou média da probabilidade de água nos pixels aceitos; não é probabilidade de risco |
| `bounding_box` | `[x_min, y_min, x_max, y_max]` em pixels |
| `track_id` | ID de rastreamento, quando disponível |
| `frame_index` / `timestamp_seconds` | quadro e instante da detecção no vídeo |

O POST de vídeo é síncrono: responde após o processamento, sem job em segundo
plano. Retorna o manifesto com `status: "completed"`, `event_id`, `video_url`,
quadros/contagens/perfis e o evento-resumo em `event`. A UI aguarda esse retorno
e baixa o MP4 por `GET /vision/videos/{event_id}`. O vídeo anotado não é enviado
como base64 no JSON. O evento-resumo pode ser consultado no histórico; a
evidência JSON e o arquivo MP4 têm endpoints próprios.

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
| 400 | mídia vazia, corrompida, com dimensões inseguras ou vídeo não processável | reexportar a foto/vídeo |
| 404 | evento, evidência ou vídeo inexistente | conferir o identificador |
| 413 | arquivo acima do limite | reduzir o arquivo |
| 415 | formato/MIME não permitido ou incompatível | enviar JPEG/PNG ou vídeo MP4/MOV/AVI/MKV com MIME correto |
| 422 | parâmetros inválidos | conferir o formulário no Swagger |
| 503 | um modelo configurado está ausente ou incompatível | preparar/verificar os modelos locais configurados |

Erros explícitos usam `{"detail": "mensagem"}`. Erros HTTP 422 seguem o
formato de validação do FastAPI. `/health` retorna `status: "degraded"` e
`model_loaded: false` quando o modelo não está disponível.

## Vídeo, marcações e histórico

O processador analisa os quadros com os componentes reais do perfil e desenha as caixas
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
Na API, o MP4 e o manifesto ficam sob `resultados/local/events/evidence/videos/`;
a interface mantém sua cópia para apresentação em `resultados/local/videos/`.
No uso por CLI, o destino é o caminho solicitado em `--output`.

O SQLite guarda identificador, local técnico, UTC, versão e detecções. Para
imagem, a evidência derivada é JPEG anotado; para vídeo, JSON resumido. O
upload original da API não é persistido e a cópia temporária de vídeo é
removida após a análise. Resultados e histórico operacional ficam fora do Git.

### Verificação funcional da versão 0.5

[product_validation.json](../resultados/metrics/v4/product_validation.json)
registra `status: "completed"` nos oito itens do catálogo: seis vídeos e duas
fotos, com eventos persistidos e serviço saudável após cada caso. Os 1.388
quadros de saída foram decodificados integralmente; todos os MP4 têm H.264,
`moov` antes de `mdat` para início rápido e entrega HTTP Range 206 conferida.
O perfil-base de imagem é `pipeline-4c5c993223219929` e o perfil de vídeo,
com confirmação de água por 0,5 s, é `pipeline-380203274bf9c28c`.
O [registro ativo](../resultados/metrics/v4/active_profile.json) reúne hashes,
provas e limites com status `verified_running_product`.

[ui_verification.json](../resultados/metrics/v4/ui_verification.json) registra
envios reais das duas fotos do usuário e reprodução nativa do vídeo Hanoi até
12/12 segundos, sem erro do player. Esse teste de tela não significa que os
seis vídeos tenham sido reproduzidos manualmente na interface; a verificação
integral dos seis arquivos, seus endpoints e decodificação está no relatório
do catálogo.

O êxito funcional não é acurácia de detecção: Rua teve uma marcação ambígua de
animal em 240 quadros, sem acerto confirmado. Hanoi teve animal em 144/240,
contra 187/240 no perfil 0.4, sem evidência de ganho temporal. Há possíveis
falsos resíduos em vegetação. A inspeção confirmou alertas indevidos na
lixeira fixa em Chuva e no cotovelo de PVC instalado em Sarjeta. `ralo_lixo`
ainda perde outros resíduos evidentes e água rasa; em Canal, a região de água
inclui concreto. Os exemplos são conhecidos e alguns forneceram quadros de
adaptação; não há referência completa por quadro para estimar recall ou
contagem física de animais.

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
| `DRENAVISION_OBJECT_MODEL_PATH` | `resultados/local/models/grounding_dino_tiny` |
| `DRENAVISION_OBJECT_VERIFIER_PATH` | `resultados/local/models/siglip2_base_patch16_224` |
| `DRENAVISION_OBJECT_RESIDUE_CONFIDENCE` | `0.30` |
| `DRENAVISION_OBJECT_ANIMAL_CONFIDENCE` | `0.25` |
| `DRENAVISION_OBJECT_RESIDUE_MIN_MARGIN` | `0.04` |
| `DRENAVISION_OBJECT_ANIMAL_MIN_MARGIN` | `0.08` |
| `DRENAVISION_OBJECT_TEXT_CONFIDENCE` | `0.20` |
| `DRENAVISION_OBJECT_MODE` | `animal_replace` |
| `DRENAVISION_OBJECT_SHORTEST_EDGE` | `640` |
| `DRENAVISION_OBJECT_LONGEST_EDGE` | `1024` |
| `DRENAVISION_WATER_MODEL_PATH` | `resultados/local/models/water_segformer_b1_ade` |
| `DRENAVISION_WATER_PIXEL_CONFIDENCE` | `0.95` |
| `DRENAVISION_WATER_MIN_AREA_FRACTION` | `0.05` |
| `DRENAVISION_WATER_MODE` | `augment` |
| `DRENAVISION_VIDEO_WATER_CONFIRMATION_SECONDS` | `0.5` |
| `DRENAVISION_MAX_IMAGE_BYTES` | 12 MiB |
| `DRENAVISION_MAX_VIDEO_BYTES` | 500 MiB |
| `DRENAVISION_MAX_IMAGE_PIXELS` | 40 milhões |

Os cortes do mestre não são os cortes do especialista. Com `animal_replace`,
a caixa final de animal vem do especialista verificado. Uma execução que mude
componentes ou parâmetros recebe outra identidade e não deve ser apresentada
como o perfil congelado. Depois de preparar/atualizar os arquivos, reinicie
com `iniciar.ps1 -Reiniciar`; uma instância antiga pode continuar na versão 0.4.

## Limites do produto

O modelo não estima profundidade, vazão, causa do problema ou probabilidade
de alagamento. Resíduo próximo ao ralo não comprova entupimento. A classe de
animal não identifica espécie com confiabilidade validada. A ausência de
detecção não significa local seguro.

Reflexos, piso molhado, sombras, pouca luz, câmera distante e objetos ocultos
reduzem a qualidade. No novo teste CLICK40, congelado antes das previsões,
as imagens com resíduos sinalizados passaram de 38 para 39; nas 60 caixas
nativas, as perdas caíram de 17 para 12, mas as previsões sem par aumentaram
de 16 para 20. Com IoU 0,50, recall foi de 0,7167 para 0,8000 e precisão de
0,7288 para 0,7059. Isso mede acordo com uma referência de fonte potencialmente
parcial. Não houve falso alerta de animal em 38 negativas claras nos dois
perfis; duas incertas foram excluídas e não há positivos de animal nessa coorte.

A latência mediana por foto na RTX 4060 aumentou de 0,615 s para 0,965 s,
com possível custo inicial adicional de carregamento. Não é uma promessa de
execução em tempo real, nem uma medida de CPU/vídeo. A água permaneceu igual
nas 40 imagens, sem retunar o componente; essa paridade não comprova acurácia.
A API respondeu `/health` com `status: "ok"`, `model_loaded: true` e versão
`0.5.0` após reiniciar os serviços e permaneceu saudável nas oito análises do
catálogo. A verificação funcional foi concluída, preservando os erros de
detecção no relatório.

Nas regressões conhecidas, animal passou de 14/20 para 19/20 positivas no
test60 e de 46/69 para 63/69 no extra119. Entretanto, o F1 de caixas IoU 0,50
caiu de 0,4706 para 0,3913 e de 0,6207 para 0,6014. Há caixas duplicadas e
amplas; envelopes parciais de pontos de pose limitam as referências, sem
explicar automaticamente todos os falsos positivos. Água ficou igual em
179/179 saídas. Os resultados por presença nos demais rótulos usam ausência
de anotação como proxy negativo, com possível incompletude entre classes.
As fontes ainda têm pouca cobertura da drenagem pretendida. Consulte
[MODEL_CARD.md](MODEL_CARD.md) e [VALIDACAO_V4.md](VALIDACAO_V4.md).

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
