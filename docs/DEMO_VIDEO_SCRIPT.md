# Roteiro do vídeo técnico - DrenaVision

## Escopo

Este roteiro dura aproximadamente **4 minutos e 30 segundos**. A faixa de 3 a
5 minutos é uma decisão do projeto para manter a demonstração objetiva; o
enunciado de Applied Computer Vision não fixa duração para este vídeo.

O objetivo é comprovar funcionamento real, não apenas apresentar telas. Grave
somente depois de executar dataset, treino, avaliação, API e interface. Se uma
classe não tiver exemplo confiável ou desempenho suficiente, mostre a limitação
e explique o próximo passo.

## Evidências que devem existir antes de gravar

- `artifacts/dataset_validation.json` válido;
- `datasets/manifests/dataset_manifest.csv` e `dataset_summary.json`;
- `artifacts/metrics/training_manifest.json`;
- `artifacts/models/best.pt` e seu SHA-256;
- `artifacts/metrics/evaluation_metrics.json`;
- matriz de confusão e exemplos de erro reais;
- uma imagem de teste por classe, nunca usada no treino;
- um vídeo de demonstração com origem/licença/consentimento registrados;
- testes passando;
- API e Streamlit funcionando localmente.

Não grave o valor de uma métrica se o arquivo autoritativo não existir.

## Preparação de tela

Abra, nesta ordem:

1. terminal no diretório do projeto;
2. editor com `dataset_summary.json`, `training_manifest.json` e `metrics.json`;
3. navegador em `http://127.0.0.1:8000/docs`;
4. navegador em `http://127.0.0.1:8501`;
5. pasta de exemplos de teste e vídeo de entrada;
6. pasta de saída `artifacts/predictions`.

Inicie serviços antes de começar:

```powershell
# Terminal 1
.\.venv\Scripts\Activate.ps1
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000

# Terminal 2
.\.venv\Scripts\Activate.ps1
python -m streamlit run app/ui/streamlit_app.py
```

Use zoom de 125% a 150%, aumente a fonte do terminal e feche notificações,
e-mail e janelas com dados pessoais.

## Cena 1 - Contexto e proposta (0:00-0:30)

**Na tela**

- título do README e Ficha Técnica;
- diagrama resumido da arquitetura;
- três classes.

**Narração sugerida**

> Este é o DrenaVision, módulo de visão computacional da DrenaHub. Ele analisa
> imagens de pontos de drenagem e sinaliza obstrução visível, água acumulada ou
> animal visível para inspeção humana. O sistema não prevê alagamentos nem mede
> profundidade; ele organiza evidências para prevenção.

**Arquivo**

- `README.md`.

## Cena 2 - Dados e treinamento customizado (0:30-1:20)

**Na tela**

- manifesto com `source_key`, licença, split e hash;
- resumo medido do dataset;
- validação sem erro;
- `configs/train.yaml`;
- manifesto de treino com hashes inicial/final e `weights_changed_from_base`.

**Comandos para deixar no histórico do terminal**

```powershell
python scripts/validate_dataset.py `
  --dataset-root datasets/processed/drenavision `
  --yaml datasets/processed/drenavision/dataset.yaml `
  --manifest datasets/manifests/dataset_manifest.csv `
  --phash `
  --report artifacts/dataset_validation.json

python scripts/train.py --config configs/train.yaml
```

**Narração sugerida**

> Os dados vêm de fontes versionadas com licença registrada. A separação por
> grupos evita que imagens relacionadas atravessem treino, validação e teste.
> Este manifesto comprova o fine-tuning: mostra a configuração efetiva, o
> dispositivo, o fingerprint do dataset e hashes diferentes entre os pesos de
> partida e o melhor checkpoint customizado.

**Arquivos**

- `datasets/manifests/dataset_manifest.csv`;
- `datasets/manifests/dataset_summary.json`;
- `artifacts/dataset_validation.json`;
- `configs/train.yaml`;
- `artifacts/metrics/training_manifest.json`.

Não precisa repetir um treino completo durante a gravação. Mostre uma execução
real anterior e, se desejar, rode `--help` para provar o ponto de entrada.

## Cena 3 - Obstrução visível (1:20-1:55)

**Na tela**

- aba “Imagem” do Streamlit;
- imagem de teste de ralo/grelha com resíduo;
- original, imagem anotada, classe e confiança;
- ID do evento.

**Narração sugerida**

> Esta imagem está no conjunto de teste e não participou do treino. O modelo
> customizado localizou material visível. A confiança pertence à detecção e não
> representa probabilidade de alagamento. O evento apenas recomenda inspeção.

**Arquivos**

- imagem identificada no manifesto como `split=test`;
- evidência criada em `artifacts/events/evidence`.

Se não houver detecção real, não troque por uma imagem de treino sem avisar.
Mostre o falso negativo e registre-o como limitação.

## Cena 4 - Água acumulada (1:55-2:25)

**Na tela**

- segunda imagem de teste;
- caixa `agua_acumulada` e confiança;
- aviso de limitação.

**Narração sugerida**

> Aqui o detector sinaliza uma região de água acumulada. Uma foto sem referência
> métrica não permite estimar centímetros de profundidade, volume ou vazão. A
> leitura deve ser combinada com inspeção e, no produto completo, com sensores e
> dados meteorológicos.

**Arquivo**

- exemplo real de teste da classe 1.

## Cena 5 - Animal e vídeo com tracking (2:25-3:05)

**Na tela**

- imagem ou vídeo de teste com animal;
- `track_id` ou resumo de IDs únicos;
- vídeo exportado.

**Comando**

```powershell
python scripts/process_video.py `
  --input caminho\video_teste.mp4 `
  --output artifacts/predictions/demo.mp4 `
  --model artifacts/models/best.pt `
  --track
```

**Narração sugerida**

> A presença do animal gera ocorrência de inspeção, não diagnóstico de
> obstrução. No vídeo usamos rastreamento para evitar somar o mesmo animal a
> cada quadro. O conjunto público desta classe é concentrado em roedores, então
> não generalizamos o resultado para toda espécie.

**Arquivos**

- vídeo de entrada com origem registrada;
- `artifacts/predictions/demo.mp4`;
- resumo JSON do processamento, se gerado.

## Cena 6 - Integração por API e histórico (3:05-3:40)

**Na tela**

- Swagger em `/docs` ou terminal com `curl.exe`;
- JSON real da predição;
- aba de histórico do Streamlit.

**Comando**

```powershell
curl.exe -X POST http://127.0.0.1:8000/vision/predict `
  -F "point_id=garagem-demo-01" `
  -F "file=@caminho\imagem_teste.jpg;type=image/jpeg"
```

**Narração sugerida**

> A integração retorna um identificador, ponto monitorado, horário UTC, versão
> do modelo, caixas, classes e confiança. Mesmo sem detecções, o contrato retorna
> uma lista vazia. O histórico permite rastrear cada ocorrência sem expor
> caminhos absolutos do servidor.

**Arquivos/rotas**

- `app/api/main.py`;
- `GET /vision/events`;
- banco local de demonstração.

## Cena 7 - Resultado técnico e limites (3:40-4:20)

**Na tela**

- `metrics.json` real;
- matriz de confusão;
- pelo menos um exemplo de falso positivo ou falso negativo;
- model card.

**Narração sugerida**

> No conjunto de teste separado, obtivemos estes valores reais de precision,
> recall, mAP@50 e mAP@50-95. O desempenho por classe mostra [ler somente os
> valores gerados]. Este exemplo de erro revela [explicar o caso]. Os principais
> limites são cobertura de domínio, iluminação e variedade da classe animal.

Substitua o trecho entre colchetes pelos resultados reais. Se o resultado for
baixo, diga isso claramente e explique dados necessários para melhorar.

**Arquivos**

- `artifacts/metrics/evaluation_metrics.json`;
- matriz de confusão gerada;
- `artifacts/metrics/error_examples/`;
- `docs/MODEL_CARD.md`.

## Encerramento (4:20-4:30)

**Na tela**

- arquitetura ou interface com o aviso de responsabilidade.

**Narração sugerida**

> O DrenaVision entrega evidência visual reproduzível para a DrenaHub, mantendo
> revisão humana e limites explícitos. Próximos passos são ampliar dados próprios,
> validar em novos locais e integrar sensores e o painel principal.

## Plano alternativo honesto

Se treino ou GPU falharem no dia:

1. não use pesos genéricos como se fossem customizados;
2. mostre o erro, logs e o estado degradado do `/health`;
3. use o notebook Colab para executar o pipeline;
4. transfira `best.pt` e confira o hash;
5. grave somente depois da inferência real.

Se não houver tempo para reexecutar, entregue o pipeline e documente o bloqueio;
não fabrique métrica ou bounding box.

## Checklist OBS

### Antes

- [ ] resolução 1920x1080 e 30 fps;
- [ ] microfone testado, sem clipping ou ruído intenso;
- [ ] captura apenas da janela/tela necessária;
- [ ] notificações e mensageiros fechados;
- [ ] zoom e fonte legíveis em tela de celular;
- [ ] `.env`, tokens, nomes de usuário e caminhos privados ocultos;
- [ ] imagens com pessoas/placas anonimizadas ou autorizadas;
- [ ] serviços iniciados e `/health` com `model_loaded=true`;
- [ ] arquivos de teste e comandos preparados;
- [ ] espaço em disco e energia conferidos;
- [ ] ensaio cronometrado.

### Durante

- [ ] mostrar original e resultado;
- [ ] mostrar ao menos uma execução real;
- [ ] provar fine-tuning por manifesto/hash;
- [ ] indicar que os exemplos são de teste;
- [ ] exibir JSON e histórico;
- [ ] mostrar métricas e um erro real;
- [ ] dizer as limitações sem minimizar falhas;
- [ ] evitar cortes que façam execução simulada parecer real.

### Depois

- [ ] conferir áudio, texto e sincronização;
- [ ] confirmar duração e resolução;
- [ ] remover qualquer segredo/dado pessoal visível;
- [ ] verificar se números coincidem com `metrics.json`;
- [ ] adicionar créditos/licenças de imagens usadas;
- [ ] exportar MP4 H.264 compatível;
- [ ] publicar em URL acessível sem login;
- [ ] testar em janela anônima e outro dispositivo;
- [ ] manter o link público durante a correção;
- [ ] registrar URL final no README e no relatório posterior.

