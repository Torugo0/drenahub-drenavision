# Instalação e reprodução

Execute os comandos na raiz do repositório. Para apenas usar o produto com o
modelo pronto, siga a instalação e depois rode `iniciar.ps1`. As demais etapas
permitem auditar e repetir o trabalho de visão computacional.

## Instalação

Ambiente de referência: Windows, Python 3.12, RTX 4060, PyTorch 2.14.1,
torchvision 0.29.1, Ultralytics 8.4.215 e OpenCV 4.14.0.94. As versões de topo
estão fixadas em `requirements.txt`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Neste host foi usado o wheel oficial CUDA 13.0 do PyTorch:

```powershell
python -m pip install --force-reinstall --no-deps -r requirements-cuda.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Em outro hardware, use o [seletor oficial](https://pytorch.org/get-started/locally/)
para escolher o wheel. CPU também é aceita, com processamento mais lento.

### Obter o modelo publicado

O modelo customizado é distribuído na
[release v0.1.0-model](https://github.com/Torugo0/drenahub-drenavision/releases/tag/v0.1.0-model).
Enquanto o repositório estiver privado, é necessário acesso à conta.

```powershell
New-Item -ItemType Directory -Force resultados/local/models | Out-Null
gh release download v0.1.0-model `
  --repo Torugo0/drenahub-drenavision `
  --pattern best.pt `
  --dir resultados/local/models
Get-FileHash resultados/local/models/best.pt -Algorithm SHA256
```

Hash esperado do checkpoint desta versão:
`fe439a3a722e73db5d841cc199026c8d42f4bbf806bf95924c2eb8c09a9f6bf7`.
As entradas de demonstração e os créditos ficam em `dados/examples/`.

```powershell
.\iniciar.ps1
```

O iniciador usa as portas 8000 (API) e 8501 (interface). Uma segunda tentativa
de iniciar o mesmo serviço pode produzir o erro Windows 10048: a porta já está
ocupada. Nesse caso, use a instância que já está aberta.

## Dados e proveniência

| Classe destino | Fonte | Licença adotada | Conversão |
|---|---|---|---|
| `obstrucao_visivel` | [CLICK Trash v2](https://data.mendeley.com/datasets/htcvvz6fm2/2), Manjit Baishya | CC BY 4.0 | `trash` → ID 0 |
| `agua_acumulada` | [Stagnant Water v4](https://data.mendeley.com/datasets/y6zyrnxbfm/4), Kailas Patil e Sonali Bhutad | CC BY 4.0 | `water` → ID 1; `wet surface` excluída |
| `animal_visivel` | [iRodent v1.0](https://doi.org/10.5281/zenodo.8250392), Ye et al. | CC BY-NC por cautela | `rodent` → ID 2 |

O registro Zenodo do iRodent indica Apache-2.0, enquanto os metadados internos
e imagens-fonte indicam CC BY-NC. O projeto mantém a condição mais restritiva.
Os dados públicos representam lixo, água parada e roedores; não validam por
si só a detecção em ralos reais.

```powershell
python ferramentas/scripts/download_datasets.py --source all --seed 42
python ferramentas/scripts/convert_annotations.py --seed 42
python ferramentas/scripts/validate_dataset.py `
  --dataset-root dados/processed/drenavision `
  --yaml dados/processed/drenavision/dataset.yaml `
  --manifest dados/manifests/dataset_manifest.csv `
  --phash --phash-threshold 4 `
  --report dados/manifests/validation_report_phash.json
python ferramentas/scripts/visualize_annotations.py --samples-per-split 6
```

O downloader retoma transferências e reutiliza arquivos verificados. Para CLICK,
usa HTTP Range e baixa 100/20/20 pares de treino/validação/teste de um ZIP maior.
O conversor preserva splits nativos quando existem, agrupa imagens relacionadas,
aplica seed 42 e usa SHA-256 nos nomes finais. Não transforma classificação
em caixas artificiais e não usa predição automática como ground truth.

O validador confere imagens, labels YOLO, classes, coordenadas, hashes, grupos
e semelhança perceptual entre splits. Erro impede o treino. Cada imagem tem
fonte, licença, split, `source_group`, hash, dimensão e contagens no manifesto.

### Preparação medida nesta entrega

| Split | Imagens | Boxes de resíduos | Boxes de água | Boxes de animal |
|---|---:|---:|---:|---:|
| Treino | 308 | 182 | 989 | 100 |
| Validação | 62 | 40 | 223 | 20 |
| Teste | 62 | 37 | 276 | 20 |

São 432 imagens, 1.887 caixas e 13 imagens negativas. A validação registrou
zero erros/avisos, zero duplicatas SHA-256 e zero vazamentos perceptuais no
pHash com distância ≤ 4. O relatório real fica em `dados/manifests/`.

Na água, a preparação removeu 167 imagens duplicadas antes da amostragem,
um rótulo sem imagem e três caixas de área zero. Imagens só de superfície
molhada entraram como negativos. No iRodent, os arquivos contêm 447 imagens,
apesar de o registro mencionar 443; uma anotação de área zero foi descartada.
As discrepâncias ficam registradas nos relatórios de aquisição/conversão.

Outras fontes investigadas constam em `dados/manifests/research_report.json`.
`drain_data` não tinha versão exportável; alternativas com login, licença
inconclusiva ou maior distância do domínio não entraram no treinamento.

## Treinamento customizado

```powershell
python ferramentas/scripts/train.py --config ferramentas/configs/train.yaml
```

Configuração principal: YOLO26n pré-treinado, `imgsz=640`, batch 8, máximo de
100 épocas, `patience=20`, seed 42 e dispositivo `auto`. O treino exige os três
splits válidos e registra argumentos, versões, dispositivo, fingerprint dos
dados, horários e hashes dos pesos inicial/final.

Esta versão foi retomada após a época 28, concluiu 70 épocas e selecionou o
melhor checkpoint na época 50. O snapshot interrompido permanece separado.
`training_manifest.json` comprova mudança dos pesos; o pipeline recarrega o
checkpoint, verifica as classes e publica a cópia em `resultados/local/models/best.pt`.

Se faltar memória, reduza batch e registre nova configuração/run. Use a
validação para escolher hiperparâmetros. O teste permanece independente.

## Avaliação e erros

```powershell
python ferramentas/scripts/evaluate.py --config ferramentas/configs/evaluate.yaml
```

A avaliação usa `split=test`, checkpoint customizado e mapa exato de classes.
Produz métricas globais/por classe, matriz, curvas, exemplos de erro e hashes.
A execução completa vai para `resultados/local/evaluation/`; a seleção versionada
fica em `resultados/metrics/`.

| Classe | Objetos no teste | Precisão | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|---:|
| Resíduos | 37 | 0,5091 | 0,6728 | 0,5860 | 0,4046 |
| Água | 276 | 0,3999 | 0,1159 | 0,1642 | 0,0360 |
| Animal | 20 | 0,4250 | 0,6500 | 0,3843 | 0,2181 |
| Agregado | 333 | 0,4447 | 0,4796 | 0,3781 | 0,2196 |

Fonte: `resultados/metrics/evaluation_metrics.json`; 62 imagens, tamanho 640,
batch 8 e RTX 4060. Inferência do modelo: 3,39 ms/imagem, sem I/O, rede ou tela.
Precisão mede a fração de marcações corretas; recall mede a fração dos objetos
anotados encontrados. mAP resume qualidade de detecção e localização.

Os exemplos de erro incluem TP, FP e FN por classe. No limiar 0,35 e IoU 0,50,
a mineração exploratória contou 32 TP, 23 FP e 301 FN; esses totais não
substituem o AP oficial. Não houve imagem de água inteiramente correta nesse
limiar. Ver [MODEL_CARD.md](MODEL_CARD.md) para protocolo e limitações.

## Comandos diretos e testes

```powershell
python ferramentas/scripts/predict.py `
  --input caminho/foto.jpg --model resultados/local/models/best.pt `
  --output-dir resultados/local/predictions

python ferramentas/scripts/process_video.py `
  --input caminho/video.mp4 --output resultados/local/videos/analise.mp4 `
  --model resultados/local/models/best.pt --track `
  --database resultados/local/events/drenavision.db `
  --evidence-dir resultados/local/events/evidence --point-id garagem-demo-01

python -m pytest -q
python -m ruff check .
```

API/interface, quando necessário iniciar individualmente:

```powershell
python -m uvicorn drenavision.api.main:app --host 127.0.0.1 --port 8000
python -m streamlit run drenavision/ui/streamlit_app.py
```

## Evidência de uma execução

Preserve commit, configuração efetiva, seed, versões (`pip freeze`), hardware,
fontes, fingerprints e hashes. O checkpoint e o manifesto de avaliação devem
apontar para os mesmos dados/modelo. Evite sobrescrever runs e histórico usados
como evidência. Uma nova decisão orientada pelo teste exige novo teste reservado.

O [notebook Colab](drenavision_training.ipynb) chama os mesmos scripts. Salve
dados e resultados antes de a sessão expirar. Não contém saídas fabricadas.
