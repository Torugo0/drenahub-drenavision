# DrenaVision - DrenaHub

## Ficha Técnica do MVP

| Campo | Definição |
|---|---|
| Produto | **DrenaVision**, módulo de visão computacional da DrenaHub |
| Cidadão e território | Síndicos, equipes de manutenção, operadores de garagens, galpões, estacionamentos, condomínios e centros logísticos |
| Problema | Inspeções visuais de pontos de drenagem são esparsas e podem deixar passar resíduos, água acumulada ou animais que merecem verificação presencial |
| Proposta de valor | Transformar imagens e vídeos de pontos críticos em evidências rastreáveis para priorizar inspeções preventivas |
| Classes | `obstrucao_visivel`, `agua_acumulada`, `animal_visivel` |
| Componentes | Aquisição e validação de dados, fine-tuning YOLO, avaliação, inferência, API FastAPI, interface Streamlit, histórico SQLite e exportação de vídeo |
| Fontes de dados | CLICK Trash v2, Dataset of Stagnant Water and Wet Surface v4 e iRodent; origem e licença são registradas no manifesto |
| Modelo de IA | YOLO26n customizado; 70 épocas concluídas de até 100, com early stopping e melhor checkpoint na época 50 |
| URLs | [GitHub privado](https://github.com/Torugo0/drenahub-drenavision); aplicação, vídeo narrado e acesso público permanecem pendentes |
| Indicadores no teste | Precision 0,4447; recall 0,4796; mAP@50 0,3781; mAP@50-95 0,2196; 3,39 ms de inferência/imagem na RTX 4060 |
| ODS relacionados | ODS 3, 6, 11 e 13, como contribuição potencial descrita abaixo |
| Estado desta entrega | Dataset real validado, fine-tuning e avaliação executados, API/UI/vídeo funcionais e evidências curadas versionadas; dados completos e pesos ficam fora do Git |
| Prazo geral da GS | **19/11/2026 às 23h55**, no Teams da disciplina; links devem permanecer públicos durante a correção |

> **Limite de uso:** uma detecção é um sinal para inspeção. Ela não comprova
> risco hidráulico, profundidade da água, entupimento ou causalidade.

## Relação com os ODS

O DrenaVision não mede impacto social ou ambiental por si só. Ele pode apoiar:

- **ODS 3 - Saúde e bem-estar:** reduzir o tempo até a inspeção de água parada e
  presença de animais em áreas monitoradas, sem diagnosticar risco sanitário;
- **ODS 6 - Água potável e saneamento:** organizar evidências para manutenção de
  infraestrutura de drenagem e manejo responsável de resíduos;
- **ODS 11 - Cidades e comunidades sustentáveis:** apoiar rotinas locais de
  prevenção e resiliência em edificações e áreas logísticas;
- **ODS 13 - Ação contra a mudança global do clima:** fortalecer preparação
  operacional diante de eventos de precipitação, sem alegar previsão climática.

As metas e indicadores de impacto devem ser definidos pela DrenaHub e avaliados
em operação; métricas do detector não substituem indicadores de resultado
público.

## O que é requisito e o que é decisão do projeto

O enunciado oficial de **Applied Computer Vision - Enxergando o mundo real**
exige detecção de objetos e/ou realidade aumentada, treinamento customizado de
arquitetura compatível, repositório GitHub com documentação e código completo e
vídeo da solução funcionando. O PDF técnico, a ser produzido depois, vale até
5 pontos; a demonstração em vídeo vale até 5 pontos.

As três classes, YOLO, FastAPI, Streamlit, SQLite, métricas detalhadas, API
versionada e roteiro de 3 a 5 minutos são decisões do DrenaVision para tornar a
entrega verificável. O enunciado de Computer Vision não define duração para o
vídeo técnico. Consulte [os requisitos mapeados](docs/FIAP_REQUIREMENTS.md).

## Estado verificável

| Evidência | Situação no repositório |
|---|---|
| Código de aquisição, conversão e validação | Disponível |
| Código de treino customizado | Disponível |
| Código de avaliação e análise de erros | Disponível |
| API e interface | Implementadas e testadas com o checkpoint real; sem ele, falham explicitamente em modo degradado |
| Dataset bruto/processado | 432 imagens (308/62/62), não versionadas; manifests e scripts reproduzíveis estão no Git |
| Checkpoint customizado | Treinado e validado; SHA-256 `fe439a3a...f6bf7`; distribuído pela release privada, não no histórico Git |
| Métricas reais | Disponíveis em `artifacts/metrics/evaluation_metrics.json`, com curvas, matrizes e análise de erros estratificada |
| Demonstração automática | MP4 anotado real de 72 quadros gerado; manifesto e evento de histórico preservados |
| Vídeo técnico narrado | **PENDENTE:** gravar 3–5 min no OBS e publicar sem login antes da correção |
| Publicação pública | **PENDENTE por decisão do usuário:** o repositório permanece privado nesta etapa |

O sistema falha de forma explícita se `artifacts/models/best.pt` não existir ou
se o mapa de classes do checkpoint for incompatível. Pesos genéricos COCO não
são apresentados como se reconhecessem as classes DrenaVision.

O checkpoint e o MP4 anotado ficam na
[release `v0.1.0-model`](https://github.com/Torugo0/drenahub-drenavision/releases/tag/v0.1.0-model).
Enquanto o repositório estiver privado, o download exige acesso à conta; depois
da abertura para correção, teste o link sem autenticação. Com GitHub CLI:

```powershell
gh release download v0.1.0-model `
  --repo Torugo0/drenahub-drenavision `
  --pattern best.pt `
  --dir artifacts/models
```

## Arquitetura

```text
fontes licenciadas -> download -> conversão YOLO -> validação de integridade
                                                        |
                                                        v
                                              fine-tuning customizado
                                                        |
                                     conjunto de teste -> avaliação real
                                                        |
                                                        v
                                                models/best.pt
                                                  /           \
                                      FastAPI + SQLite      Streamlit
                                                  \           /
                                              evidências e vídeo
```

Uma visão operacional está em [docs/ARQUITETURA.md](docs/ARQUITETURA.md), com
detalhes de responsabilidade e fronteiras em
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Início rápido no Windows PowerShell

### 1. Ambiente

Use CPython 3.12. O Python 3.10 chegou ao fim de suporte em outubro de 2026 e
não é a base recomendada para este projeto.

```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

As versões de topo estão fixadas em `requirements.txt`. Neste host Windows com
RTX 4060 foi usado o wheel oficial CUDA 13.0 do PyTorch:

```powershell
python -m pip install --force-reinstall --no-deps -r requirements-cuda.txt
```

Em outro hardware, confirme no
[seletor oficial do PyTorch](https://pytorch.org/get-started/locally/) qual
índice de wheels corresponde ao sistema e registre o comando usado.

Verifique o ambiente:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
python -m pytest -q
python -m ruff check .
```

### 2. Dados

Os downloads somam centenas de megabytes e não entram no Git. Há três fontes
aprovadas e verificadas por tamanho/hash quando a fonte oferece esses dados.

```powershell
python scripts/download_datasets.py --source all
python scripts/convert_annotations.py --seed 42
python scripts/validate_dataset.py `
  --dataset-root datasets/processed/drenavision `
  --yaml datasets/processed/drenavision/dataset.yaml `
  --manifest datasets/manifests/dataset_manifest.csv `
  --phash `
  --report artifacts/dataset_validation.json
```

Pare se a validação retornar código diferente de zero. O relatório detecta
classes e caixas inválidas, arquivos ausentes, duplicatas exatas, duplicatas
visuais aproximadas entre splits e vazamento de `source_group`.

O processo, as contagens e as licenças estão detalhados em `docs/DATASETS.md`.

### 3. Treino customizado

Revise `configs/train.yaml` e execute:

```powershell
python scripts/train.py --config configs/train.yaml
```

O treino usa seed 42, detecta CUDA/MPS/CPU e registra configuração, versões,
hashes, horários e dispositivo. Nesta execução, 70 épocas foram concluídas com
early stopping; o manifesto curado está em
`artifacts/metrics/training_manifest.json` e o checkpoint local é
`artifacts/models/best.pt`.

Não declare o treinamento concluído apenas porque o script existe. A evidência
é a execução, seus logs e o checkpoint com hash registrado.
Veja o protocolo completo em [docs/TREINAMENTO.md](docs/TREINAMENTO.md).

### 4. Avaliação em teste separado

```powershell
python scripts/evaluate.py --config configs/evaluate.yaml
```

A avaliação escreve em `artifacts/evaluation/` as métricas globais e por
classe, matriz de confusão, gráficos e exemplos de erro quando os dados
permitem. Publique números somente a partir do arquivo real `metrics.json`.

| Métrica | Resultado real no teste (62 imagens, 333 boxes) |
|---|---|
| Precision | 0,4447 |
| Recall | 0,4796 |
| mAP@50 | 0,3781 |
| mAP@50-95 | 0,2196 |
| `obstrucao_visivel` P/R/mAP50/mAP50-95 | 0,5091 / 0,6728 / 0,5860 / 0,4046 |
| `agua_acumulada` P/R/mAP50/mAP50-95 | 0,3999 / 0,1159 / 0,1642 / 0,0360 |
| `animal_visivel` P/R/mAP50/mAP50-95 | 0,4250 / 0,6500 / 0,3843 / 0,2181 |
| Tempo de inferência | 3,39 ms/imagem na RTX 4060, sem I/O/UI/rede |

O [model card](docs/MODEL_CARD.md) registra o protocolo, hashes, resultados e
limitações. A baixa sensibilidade para água impede uso autônomo.
Consulte também [docs/METRICAS.md](docs/METRICAS.md).

### 5. Inferência em imagem e vídeo

```powershell
python scripts/predict.py `
  --input caminho\imagem.jpg `
  --model artifacts/models/best.pt `
  --output-dir artifacts/predictions

python scripts/process_video.py `
  --input caminho\video.mp4 `
  --output artifacts/predictions/demo.mp4 `
  --model artifacts/models/best.pt `
  --database artifacts/events/drenavision.db `
  --evidence-dir artifacts/events/evidence `
  --point-id ponto-demo-01 `
  --track
```

No vídeo, a contagem usa o maior valor entre IDs únicos e o pico por quadro,
evitando soma ingênua e subcontagem em quadros sem ID. Com `--database`, um
evento-resumo com frame/timestamp representativo e evidência JSON entra no
mesmo histórico da API.

### 6. API FastAPI

Terminal 1:

```powershell
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

Verificação e predição:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health

curl.exe -X POST http://127.0.0.1:8000/vision/predict `
  -F "point_id=garagem-bloco-a-01" `
  -F "file=@caminho\imagem.jpg;type=image/jpeg"
```

Documentação interativa: `http://127.0.0.1:8000/docs`. Quando o checkpoint não
está disponível, `/health` retorna `degraded` e a predição retorna HTTP 503.
Veja [docs/API.md](docs/API.md).

### 7. Interface Streamlit

Terminal 2, com a API já em execução:

```powershell
python -m streamlit run app/ui/streamlit_app.py
```

Abra `http://127.0.0.1:8501`. A interface oferece upload de imagem, captura por
câmera, vídeo, confiança por detecção e histórico. Ela consome a mesma API e o
mesmo checkpoint; não existe modo de demonstração com resultados simulados.

## Contrato resumido da API

| Método | Caminho | Função |
|---|---|---|
| GET | `/health` | estado da API, banco e checkpoint |
| GET | `/vision/classes` | mapa estável das três classes |
| POST | `/vision/predict` | imagem JPEG/PNG e `point_id` opcional |
| GET | `/vision/events` | histórico paginado/filtrado |
| GET | `/vision/events/{event_id}` | evento individual |

Uma predição válida sem detecções continua retornando HTTP 200 e
`"detections": []`. Caminhos absolutos locais não fazem parte da resposta.

## Estrutura principal

```text
app/
  api/                 FastAPI e validação de uploads
  core/                contratos, persistência e serviço de aplicação
  data/                download, conversão e validação YOLO
  inference/           treino, avaliação, predição e seleção de dispositivo
  ui/                  interface Streamlit
configs/               configurações versionadas de treino e avaliação
datasets/              manifests e dados locais ignorados pelo Git
scripts/               CLIs reproduzíveis
notebooks/             fluxo Colab chamando os mesmos scripts
tests/                 testes unitários e de contrato
artifacts/             evidências curadas no Git; pesos, runs, eventos e vídeos ignorados
docs/                  requisitos, arquitetura, ética e demonstração
```

## Reproduzir a demonstração

1. valide o ambiente e execute os testes;
2. apresente o manifesto, o relatório de validação e a separação dos splits;
3. mostre a evidência de treino e o hash de `best.pt`;
4. rode a avaliação no conjunto de teste e abra `metrics.json`;
5. inicie API e Streamlit;
6. analise exemplos de teste que não participaram do treino;
7. exiba o JSON do evento e o histórico;
8. reproduza `artifacts/videos/drenavision_demo_annotated.mp4` com
   `scripts/build_demo_video.py` e `scripts/process_video.py`;
9. declare limitações e itens ainda pendentes.

O roteiro canônico está em
[docs/VIDEO_ROTEIRO.md](docs/VIDEO_ROTEIRO.md), com notas expandidas em
[docs/DEMO_VIDEO_SCRIPT.md](docs/DEMO_VIDEO_SCRIPT.md). O notebook equivalente
está em `notebooks/drenavision_training.ipynb`.

## Segurança, privacidade e uso responsável

- arquivos enviados são limitados por tipo, tamanho e número de pixels;
- nomes originais não são usados como caminhos de evidência;
- a API não persiste a imagem original, apenas evidência anotada e metadados;
- o banco SQLite e os artefatos operacionais ficam fora do Git;
- o `point_id` não deve conter endereço, nome de morador ou outro dado pessoal;
- imagens com pessoas, placas ou endereços exigem minimização, retenção e base
  legal antes de uso real;
- desempenho pode mudar por iluminação, câmera, domínio, clima e classe;
- revisão humana permanece obrigatória.

Veja o resumo em [docs/LIMITACOES.md](docs/LIMITACOES.md) e a análise completa
em [docs/ETHICS_AND_LIMITATIONS.md](docs/ETHICS_AND_LIMITATIONS.md).

## Licenças

O código é distribuído sob **AGPL-3.0-only**, coerente com a licença padrão da
Ultralytics. A AGPL impõe obrigações de disponibilização de código-fonte,
inclusive quando software modificado é oferecido por rede. Para transformar a
DrenaHub em produto proprietário/fechado, é necessária revisão jurídica e,
conforme o caso, licença Enterprise da Ultralytics ou substituição do detector.

As imagens têm licenças próprias. Em particular, o conflito entre o registro
Apache-2.0 do iRodent e a indicação CC BY-NC para imagens-fonte é tratado de
forma conservadora como **CC BY-NC**; uso comercial desse material não é
autorizado por este projeto. Consulte [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Solução de problemas

### `/health` mostra `degraded`

Confirme que `artifacts/models/best.pt` existe, que foi produzido pelo treino
customizado e que `DRENAVISION_MODEL_PATH` aponta para ele. Verifique também se
o checkpoint contém exatamente as três classes esperadas.

### CUDA não aparece

Execute o comando de diagnóstico do PyTorch acima, confira o driver NVIDIA e
reinstale o wheel recomendado pelo seletor oficial. A versão de CUDA exibida
por `nvidia-smi` não precisa coincidir com a versão embutida no wheel.

### Dataset inválido

Não ignore o código de saída. Revise `artifacts/dataset_validation.json`,
remova vazamento entre splits e reconverta a fonte. Frames semelhantes do mesmo
vídeo devem permanecer no mesmo `source_group`.

### OpenCV não abre o MP4

Instale FFmpeg no sistema ou teste outro codec/contêiner. Preserve o vídeo de
entrada e registre qualquer transcodificação usada na demonstração.

### OneDrive sincroniza arquivos grandes

O diretório do projeto está sob OneDrive. Mantenha `datasets/raw`, checkpoints,
runs e vídeos fora de sincronização quando possível, sem alterar os manifests
versionados.

## Checklist antes da entrega

- [ ] substituir campos de integrantes no relatório PDF posterior;
- [x] executar dataset, treino, avaliação e testes no ambiente documentado;
- [x] preencher `docs/MODEL_CARD.md` apenas com métricas reais;
- [ ] gravar e revisar o vídeo técnico;
- [ ] criar o relatório PDF exigido pela FIAP em etapa posterior;
- [ ] publicar repositório, aplicação e vídeo;
- [ ] testar todos os links em janela anônima;
- [ ] manter os links públicos e funcionando durante a correção;
- [ ] um integrante enviar pelo Teams até 19/11/2026 às 23h55.

O checklist operacional completo está em
[docs/CHECKLIST_FIAP.md](docs/CHECKLIST_FIAP.md).

## Índice da documentação

| Documento | Conteúdo |
|---|---|
| [FIAP_REQUIREMENTS.md](docs/FIAP_REQUIREMENTS.md) | requisito oficial versus decisão do projeto |
| [CHECKLIST_FIAP.md](docs/CHECKLIST_FIAP.md) | verificação final e publicação |
| [ARQUITETURA.md](docs/ARQUITETURA.md) | visão operacional |
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | componentes, fluxos e estados de erro |
| `docs/DATASETS.md` | fontes, licenças, preparação e anotação |
| [TREINAMENTO.md](docs/TREINAMENTO.md) | execução e prova de fine-tuning |
| [METRICAS.md](docs/METRICAS.md) | protocolo de avaliação e interpretação |
| `docs/MODEL_CARD.md` | ficha do checkpoint e resultados reais |
| [API.md](docs/API.md) | contrato HTTP e exemplos |
| [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | hashes, ambiente e evidências |
| [LIMITACOES.md](docs/LIMITACOES.md) | limites essenciais |
| [ETHICS_AND_LIMITATIONS.md](docs/ETHICS_AND_LIMITATIONS.md) | ética, LGPD e riscos |
| [VIDEO_ROTEIRO.md](docs/VIDEO_ROTEIRO.md) | roteiro canônico e checklist OBS |

## Referências essenciais

- [Ultralytics - modelos e treinamento](https://docs.ultralytics.com/)
- [Ultralytics - opções de licença](https://www.ultralytics.com/license)
- [PyTorch - instalação oficial](https://pytorch.org/get-started/locally/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://docs.streamlit.io/)
- [Objetivos de Desenvolvimento Sustentável da ONU](https://sdgs.un.org/goals)

