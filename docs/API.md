# API DrenaVision

## Execução local

```powershell
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

- OpenAPI: `http://127.0.0.1:8000/openapi.json`
- Swagger UI: `http://127.0.0.1:8000/docs`
- Redoc: `http://127.0.0.1:8000/redoc`

A API inicia mesmo sem checkpoint para permitir diagnóstico. Nesse caso o
estado é `degraded` e endpoints de predição retornam HTTP 503.

## Convenções

- horários em UTC e ISO 8601;
- caixas no formato `xyxy_pixels`: `[x_min, y_min, x_max, y_max]`;
- confiança no intervalo `[0, 1]`;
- classes com IDs estáveis 0, 1 e 2;
- uploads via `multipart/form-data`;
- a lista `detections` existe mesmo quando vazia;
- `inspection_recommended` é uma regra operacional, não uma estimativa de risco;
- caminhos absolutos do servidor não devem fazer parte de respostas públicas.

Os exemplos abaixo documentam o formato e **não são resultados medidos**.

## `GET /health`

Informa se o serviço, o banco e o checkpoint estão disponíveis.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Exemplo degradado:

```json
{
  "status": "degraded",
  "model_loaded": false,
  "model_path": "best.pt",
  "database_ready": true,
  "version": "0.1.0"
}
```

Em hospedagem pública, `model_path` deve permanecer sanitizado e relativo ou
ser substituído por um booleano; nunca exponha a estrutura absoluta do host.

## `GET /vision/classes`

```powershell
Invoke-RestMethod http://127.0.0.1:8000/vision/classes
```

```json
{
  "classes": [
    {
      "class_id": 0,
      "name": "obstrucao_visivel",
      "label_pt_br": "Obstrução visível"
    },
    {
      "class_id": 1,
      "name": "agua_acumulada",
      "label_pt_br": "Água acumulada"
    },
    {
      "class_id": 2,
      "name": "animal_visivel",
      "label_pt_br": "Animal visível"
    }
  ]
}
```

## `POST /vision/predict`

Campos:

| Campo | Tipo | Regra |
|---|---|---|
| `file` | arquivo | JPEG ou PNG válido, dentro dos limites configurados |
| `point_id` | texto opcional | até 80 caracteres; use identificador técnico sem dado pessoal |

```powershell
curl.exe -X POST http://127.0.0.1:8000/vision/predict `
  -H "accept: application/json" `
  -F "point_id=garagem-bloco-a-01" `
  -F "file=@C:\dados\teste.jpg;type=image/jpeg"
```

Exemplo de **formato** com detecção:

```json
{
  "event_id": "00000000-0000-0000-0000-000000000000",
  "point_id": "garagem-bloco-a-01",
  "timestamp": "2026-11-16T18:42:00Z",
  "input_type": "image",
  "model_version": "best-<12-primeiros-caracteres-do-sha256>",
  "detections": [
    {
      "class_id": 0,
      "class_name": "obstrucao_visivel",
      "confidence": 0.87,
      "bounding_box": [108.0, 96.0, 411.0, 356.0],
      "bounding_box_format": "xyxy_pixels",
      "track_id": null,
      "frame_index": null,
      "timestamp_seconds": null
    }
  ],
  "inspection_recommended": true,
  "status": "revisao_recomendada",
  "evidence_uri": "00000000-0000-0000-0000-000000000000.jpg",
  "inference_ms": 24.5,
  "disclaimer": "Triagem assistida por IA. A detecção não confirma risco sanitário ou hidráulico e não substitui inspeção humana qualificada."
}
```

Os números acima são ilustrativos. Não os copie para relatório, vídeo ou model
card como se fossem resultado experimental.

Exemplo de formato sem detecção:

```json
{
  "event_id": "00000000-0000-0000-0000-000000000001",
  "point_id": "garagem-bloco-a-02",
  "timestamp": "2026-11-16T18:43:00Z",
  "input_type": "image",
  "model_version": "best-<12-primeiros-caracteres-do-sha256>",
  "detections": [],
  "inspection_recommended": false,
  "status": "sem_deteccao",
  "evidence_uri": "00000000-0000-0000-0000-000000000001.jpg",
  "inference_ms": 23.1,
  "disclaimer": "Triagem assistida por IA. A detecção não confirma risco sanitário ou hidráulico e não substitui inspeção humana qualificada."
}
```

Ausência de detecção não comprova que o local está seguro. Ela significa apenas
que o modelo não retornou objetos acima do limiar naquela imagem.

## `GET /vision/events`

Parâmetros:

| Parâmetro | Padrão | Limite |
|---|---:|---:|
| `limit` | 50 | 1 a 500 |
| `offset` | 0 | maior ou igual a 0 |
| `point_id` | vazio | até 80 caracteres |

```powershell
Invoke-RestMethod "http://127.0.0.1:8000/vision/events?limit=20&offset=0&point_id=garagem-bloco-a-01"
```

Resposta:

```json
{
  "items": [
    {
      "event_id": "00000000-0000-0000-0000-000000000000",
      "point_id": "garagem-bloco-a-01",
      "timestamp": "2026-11-16T18:42:00Z",
      "input_type": "image",
      "model_version": "best-<12-primeiros-caracteres-do-sha256>",
      "detection_count": 1,
      "detected_classes": ["obstrucao_visivel"],
      "max_confidence": 0.87,
      "inspection_recommended": true,
      "status": "revisao_recomendada",
      "evidence_uri": "00000000-0000-0000-0000-000000000000.jpg"
    }
  ],
  "total": 1
}
```

## `GET /vision/events/{event_id}`

Retorna o mesmo contrato completo da predição. Um identificador inexistente
retorna HTTP 404.

```powershell
Invoke-RestMethod http://127.0.0.1:8000/vision/events/00000000-0000-0000-0000-000000000000
```

Eventos de vídeo são criados pelo processador local usado no Streamlit ou por
`scripts/process_video.py --database ... --evidence-dir ... --point-id ...`.
Eles usam `input_type: "video"`; as detecções representativas incluem
`track_id`, `frame_index` e `timestamp_seconds`. O manifesto completo do vídeo
continua ao lado do MP4.

## `GET /vision/evidence/{event_id}`

Retorna a evidência derivada do evento: JPEG anotado para imagem ou JSON
resumido para vídeo. A imagem/vídeo original não é retornado nem persistido pela
API. Eventos sem evidência ou UUID inexistente retornam HTTP 404.

```powershell
Invoke-WebRequest `
  http://127.0.0.1:8000/vision/evidence/00000000-0000-0000-0000-000000000000 `
  -OutFile evidencia
```

## Erros

| HTTP | Condição | Ação |
|---:|---|---|
| 400 | bytes não formam imagem segura/decodificável | reexportar JPEG/PNG e repetir |
| 404 | evento não encontrado | conferir o UUID |
| 413 | upload excede limite | reduzir/resamplear no cliente |
| 415 | MIME não permitido | enviar JPEG ou PNG |
| 422 | formulário/parâmetros inválidos | revisar contrato OpenAPI |
| 503 | checkpoint ausente/incompatível | treinar e configurar `best.pt` |

Formato geral para os erros produzidos explicitamente pela aplicação:

```json
{
  "detail": "mensagem sem caminho interno ou segredo"
}
```

Erros de validação HTTP 422 seguem o contrato padrão do FastAPI e podem trazer
uma lista estruturada em `detail`; consulte o OpenAPI em execução.

## Regra de inspeção

Na versão atual:

```text
inspection_recommended = existe ao menos uma detecção aceita pelo limiar
```

A regra não pondera chuva, sensor, criticidade do local, profundidade ou risco
hidráulico. Uma integração futura pode combinar sinais, mas deve manter a
origem de cada evidência e evitar apresentar uma heurística como probabilidade.

## Segurança para publicação

O MVP limita tipo/tamanho/pixels, mas uma implantação pública ainda precisa de:

- autenticação e autorização;
- TLS e proxy reverso;
- rate limiting e limites de concorrência;
- CORS restrito à URL real da interface;
- scanner de dependências e atualização de imagens base;
- segregação de banco/evidências;
- retenção, descarte e trilha de auditoria;
- logs sem bytes de imagem, token, endereço ou caminho absoluto;
- política de privacidade e base legal.

Não use `--reload` em publicação. O servidor local descrito aqui é para
demonstração controlada.

