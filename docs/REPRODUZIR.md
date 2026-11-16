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

O modelo ativo v2 é distribuído na
[release v0.3.0-model-v2](https://github.com/Torugo0/drenahub-drenavision/releases/tag/v0.3.0-model-v2).
Enquanto o repositório estiver privado, é necessário acesso à conta.

```powershell
New-Item -ItemType Directory -Force resultados/local/models | Out-Null
gh release download v0.3.0-model-v2 `
  --repo Torugo0/drenahub-drenavision `
  --pattern best.pt `
  --dir resultados/local/models
Get-FileHash resultados/local/models/best.pt -Algorithm SHA256
```

Hash esperado do checkpoint desta versão:
`01ea7c578783b2a08fc585051b0768a1e9c0d510cb5652173fab0979f8d7d27a`.
As entradas de demonstração e os créditos ficam em `dados/examples/`.
Se já existir `best.pt`, preserve sua cópia antes de substituí-lo. A promoção
realizada nesta versão manteve o v1 em `best_v1.pt`.

```powershell
.\iniciar.ps1
```

O iniciador usa as portas 8000 (API) e 8501 (interface). Uma segunda tentativa
de iniciar o mesmo serviço pode produzir o erro Windows 10048: a porta já está
ocupada. Nesse caso, use a instância que já está aberta.

## Dados e proveniência

A execução v1 e suas métricas são históricas. A auditoria posterior encontrou
caixas iRodent com fundo incorporado por keypoints ausentes; consulte a seção
de protocolo v2 abaixo antes de gerar novos dados. O conversor atual corrige
essas caixas e não reproduz os rótulos/fingerprint do treinamento v1.

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
  --dataset-root dados/processed/drenavision_corrected `
  --yaml dados/processed/drenavision_corrected/dataset.yaml `
  --manifest dados/manifests/dataset_manifest_corrected.csv `
  --phash --phash-threshold 4 `
  --report dados/manifests/validation_report_corrected.json
python ferramentas/scripts/visualize_annotations.py --dataset-root dados/processed/drenavision_corrected --output-root resultados/local/dataset_preview_corrected --samples-per-split 6
```

Essa conversão isolada grava em `drenavision_corrected/`, sem sobrescrever o
dataset e manifesto históricos v1. Ela não inclui a expansão e os frames de
domínio do experimento v2. Para reproduzir o candidato treinado nesta revisão,
use `prepare_adaptation.py` descrito adiante.

O downloader retoma transferências e reutiliza arquivos verificados. Para CLICK,
usa HTTP Range e baixa 100/20/20 pares de treino/validação/teste de um ZIP maior.
O conversor preserva splits nativos quando existem, agrupa imagens relacionadas,
aplica seed 42 e usa SHA-256 nos nomes finais. Não transforma classificação
em caixas artificiais e não usa predição automática como ground truth.

O validador confere imagens, labels YOLO, classes, coordenadas, hashes, grupos
e semelhança perceptual entre splits. Erro impede o treino. Cada imagem tem
fonte, licença, split, `source_group`, hash, dimensão e contagens no manifesto.

### Preparação histórica medida na entrega v1

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

## Treinamento customizado histórico v1

Os comandos e resultados desta seção descrevem o protocolo v1. Para um novo
experimento com anotações corrigidas e sem sobrescrever os dados v1, use a
preparação v2 abaixo.

```powershell
# Referência histórica v1; não execute para treinar o modelo atual:
# python ferramentas/scripts/train.py --config ferramentas/configs/train.yaml
```

Configuração principal: YOLO26n pré-treinado, `imgsz=640`, batch 8, máximo de
100 épocas, `patience=20`, seed 42 e dispositivo `auto`. O treino exige os três
splits válidos e registra argumentos, versões, dispositivo, fingerprint dos
dados, horários e hashes dos pesos inicial/final.

Esta versão foi retomada após a época 28, concluiu 70 épocas e selecionou o
melhor checkpoint na época 50. O snapshot interrompido permanece separado.
Naquela execução, `training_manifest.json` comprovou mudança dos pesos e a
cópia publicada foi `best.pt`, hoje preservada como `best_v1.pt`. A configuração
histórica atual gera `candidate_v1_reproduction.pt` e não substitui o ativo.

Se faltar memória, reduza batch e registre nova configuração/run. Use a
validação para escolher hiperparâmetros. O teste permanece independente.

## Preparação e avaliação do modelo v2

A causa documentada em
[annotation_audit.json](../resultados/metrics/v2/annotation_audit.json) está
nas caixas originais da fonte iRodent: 305 de 447 retângulos incluem os pontos
ausentes `(0,0,0)`. A conversão v2 usa somente keypoints com visibilidade maior
que zero, margem de 10% por lado (mínimo 2 px) e clipping. Cinco anotações com
menos de dois pontos visíveis são excluídas. O retângulo resultante pode
representar somente a parte visível anotada, não um corpo inteiro revisado
independentemente por uma pessoa.

Primeiro gere a auditoria e confira a montagem de 12 imagens:

```powershell
python ferramentas/scripts/prepare_adaptation.py --audit-only
```

Saídas: `resultados/local/adaptation/annotation_audit.json` e
`irodent_source_vs_v2.jpg`. O resumo e uma montagem estão versionados em
`resultados/metrics/v2/`, com hashes, coordenadas de exemplo, origem e licença.

Para adicionar imagens/cortes de domínio revisados, prepare uma pasta com
`images/`, `labels/` YOLO e `sources.json` contendo `items`, cada um com
`file_name`, `source_url`, `source_license`, `source_group`, `annotation_note`
e `split: "train"`. A adaptação não aceita frames de domínio em validação ou
teste nem usa predições do detector como rótulo. Os rótulos desta execução
foram produzidos por inspeção visual assistida e não são revisão humana
independente.

```powershell
python -c "from pathlib import Path; import shutil; from ultralytics.utils.downloads import attempt_download_asset; p=Path('resultados/local/models'); p.mkdir(parents=True,exist_ok=True); shutil.copy2(attempt_download_asset(p/'yolo26n.pt'),p/'yolo26n-base.pt')"
python ferramentas/scripts/prepare_negative_crops.py
python ferramentas/scripts/prepare_domain_annotations.py
python ferramentas/scripts/prepare_adaptation.py --domain-root dados/interim/adaptation
python ferramentas/scripts/train.py --config ferramentas/configs/train_v2.yaml
```

O primeiro comando mantém o peso inicial oficial do Ultralytics no cache local,
com o nome esperado pela configuração; ele não usa o checkpoint customizado
ativo como peso inicial. O preparador de domínio reúne os cinco cortes
negativos revisados e os onze frames/cortes positivos, todos apenas no treino.

Em um clone novo, execute antes `download_datasets.py --source all --seed 42`.
O preparador recupera as imagens e labels de resíduos/água diretamente das
fontes brutas quando o dataset processado v1 não está disponível. Confere o
SHA-256 da imagem e as contagens de caixas contra o manifesto v1 versionado;
não é necessário reconstruir o dataset defeituoso para gerar o v2.

O preparador recusa sobrescrever o dataset v1 ou uma pasta v2 já preenchida;
para outro experimento, escolha `--output-root` e `--audit-dir` novos e ajuste
a configuração. Mantém os IDs dos splits v1 de validação/teste, corrige suas
caixas de roedor e expande o treino somente com imagens nativas de treino
iRodent fora dos grupos/hashes reservados. O manifesto fica dentro do dataset
em `data_manifest.csv`, com uma cópia em `resultados/local/adaptation/`.
Validação de estrutura, proveniência, duplicatas, grupos e pHash precede o treino.

Nesta execução foram preparados **557/62/62 itens** de treino/validação/teste,
373 imagens iRodent no total, 234 imagens nativas de treino adicionais e 16
imagens/cortes de domínio. Só uma imagem do treino v1 foi removida por
geometria insuficiente. Os domínios `rato_close`, `rato_residuos`, `canal_residuos`
e `drenagem_chuva` entraram apenas no treino. `rato_rua` e `sarjeta_residuos`
ficaram fora da adaptação e servem como regressão já inspecionada durante o
desenvolvimento, não teste de campo intocado.

O treinamento usa o YOLO26n inicial, seed 42, tamanho 640, batch 12, máximo de
100 épocas e patience 25. A execução concluída parou na época 60 e selecionou
a época 35. O destino do treino é `resultados/local/models/candidate_v2.pt`;
o treino não substitui automaticamente o modelo ativo. Após a comparação,
esse checkpoint foi promovido a `best.pt`, preservando o antigo em `best_v1.pt`.

Para repetir a comparação no teste corrigido, obtenha o baseline histórico se
ele não estiver no computador:

```powershell
if (-not (Test-Path resultados/local/models/best_v1.pt)) {
  New-Item -ItemType Directory -Force resultados/local/models/v1 | Out-Null
  gh release download v0.1.0-model --repo Torugo0/drenahub-drenavision --pattern best.pt --dir resultados/local/models/v1
  Copy-Item resultados/local/models/v1/best.pt resultados/local/models/best_v1.pt
}
python ferramentas/scripts/evaluate.py --config ferramentas/configs/evaluate_v2_baseline.yaml
python ferramentas/scripts/evaluate.py --config ferramentas/configs/evaluate_v2.yaml --model resultados/local/models/candidate_v2.pt
```

O v1 tem hash `fe439a3a722e73db5d841cc199026c8d42f4bbf806bf95924c2eb8c09a9f6bf7`.
Ambas as configurações usam `dados/processed/drenavision_v2/dataset.yaml` e
`split=test`. Consulte os arquivos versionados
[baseline_test_metrics.json](../resultados/metrics/v2/baseline_test_metrics.json),
[candidate_test_metrics.json](../resultados/metrics/v2/candidate_test_metrics.json)
e [comparison.json](../resultados/metrics/v2/comparison.json). O mAP@50 agregado
nesse teste foi de 0,2563 no baseline para 0,3581 no v2; animal foi de 0,0183
para 0,3102. Esses números não são o AP de validação usado para escolher a época.
Água e alguns exemplos regrediram; a [revisão dos clipes](../resultados/metrics/v2/product_validation_v2.json)
registra ganhos e falhas. Métricas v1 calculadas com caixas antigas permanecem
como histórico e não são diretamente comparáveis ao protocolo v2.

## Avaliação e erros históricos v1

```powershell
# Referência histórica: exige os dados v1 preservados.
# python ferramentas/scripts/evaluate.py --config ferramentas/configs/evaluate.yaml --model resultados/local/models/best_v1.pt
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

Os comandos gerais agora usam o protocolo **v2**: `train.py`/`run_training()`
carregam `train_v2.yaml` e geram somente `candidate_v2.pt`; `evaluate.py`/
`run_evaluation()` usam `evaluate_v2.yaml`, o `best.pt` ativo e os rótulos v2.
Validador, visualizador e construtor de sequência também usam o dataset e
manifesto locais v2 por padrão. As configurações `train.yaml` e `evaluate.yaml`
são históricas e precisam ser solicitadas explicitamente; a primeira gera
`candidate_v1_reproduction.pt`, e a segunda exige o backup `best_v1.pt`.

```powershell
python ferramentas/scripts/validate_dataset.py --phash
python ferramentas/scripts/visualize_annotations.py
python ferramentas/scripts/evaluate.py

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
