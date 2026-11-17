# Instalação e reprodução

Execute os comandos na raiz do repositório. A versão 0.4.0 usa o detector YOLO
v3 e o segmentador de água SegFormer B1. Para abrir o produto, basta instalar as
dependências e obter os modelos. Baixar datasets é uma etapa de reprodução
acadêmica, separada do uso normal.

## Instalar o ambiente

Ambiente de referência: Windows, Python 3.12 e RTX 4060. PyTorch 2.14.1,
torchvision 0.29.1, Ultralytics 8.4.215 e Transformers 4.57.3 estão fixados,
com as demais versões testadas, nos arquivos `requirements*.txt`.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Neste computador foi usado o wheel oficial CUDA 13.0:

```powershell
python -m pip install --force-reinstall --no-deps -r requirements-cuda.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

CPU também é aceita, com processamento mais lento. Em outro hardware, consulte
o [seletor oficial do PyTorch](https://pytorch.org/get-started/locally/). A
instalação CUDA acima descreve este treinamento e não é obrigatória para todo
computador.

## Obter os modelos e iniciar

Os arquivos estão na
[release v0.4.0-pipeline-v3](https://github.com/Torugo0/drenahub-drenavision/releases/tag/v0.4.0-pipeline-v3).
O acesso a um repositório privado exige uma conta autorizada. Os comandos abaixo
usam o GitHub CLI autenticado. Também é possível baixar os assets pela página
da release. Em uma atualização, preserve seus modelos e `.env` anteriores.

```powershell
New-Item -ItemType Directory -Force resultados/local/release/v0.4.0 | Out-Null
New-Item -ItemType Directory -Force resultados/local/models | Out-Null
New-Item -ItemType Directory -Force dados/examples | Out-Null
gh release download v0.4.0-pipeline-v3 `
  --repo Torugo0/drenahub-drenavision `
  --pattern best.pt --pattern best_v2.pt `
  --pattern water_model.zip --pattern exemplos_reais.zip `
  --dir resultados/local/release/v0.4.0
Copy-Item resultados/local/release/v0.4.0/best.pt resultados/local/models/best.pt
Copy-Item resultados/local/release/v0.4.0/best_v2.pt resultados/local/models/best_v2.pt
Expand-Archive resultados/local/release/v0.4.0/water_model.zip `
  -DestinationPath resultados/local/models/water_segformer_b1_ade
Expand-Archive resultados/local/release/v0.4.0/exemplos_reais.zip `
  -DestinationPath dados/examples -Force
python ferramentas/scripts/prepare_water_model.py
```

`best_v2.pt` é necessário para repetir o treino; o uso normal utiliza `best.pt`
e a pasta do segmentador. O ZIP de água tem exatamente cinco arquivos na raiz:
`model.safetensors`, `config.json`, `preprocessor_config.json`,
`download_manifest.json` e `LICENSE-NVIDIA-SegFormer.txt`. Não crie uma pasta
extra dentro de `water_segformer_b1_ade/` ao extrair. O preparador verifica uma
instalação correta; se ela estiver ausente, obtém a revisão oficial fixada e
reproduz os mesmos cinco arquivos. A API não baixa modelos durante análises.

O SegFormer NVIDIA acompanha licença de **pesquisa e avaliação não comercial**.
Esta distribuição acadêmica não concede uso comercial dos pesos. A licença
integral e a origem estão no pacote e em
`resultados/metrics/v3/water_model_source/`.

| Artefato | SHA-256 esperado |
|---|---|
| YOLO v3 ativo, `best.pt` | `12b9e86f223366c3f2ef8c2308f0e1140cda08c55ab55e281e58d300fd554b4b` |
| Inicial histórico v2, `best_v2.pt` | `01ea7c578783b2a08fc585051b0768a1e9c0d510cb5652173fab0979f8d7d27a` |
| Água B1, `model.safetensors` | `e0802e077c195d9dbaf48c69b94946ef6a3ae5b6665176fedf363e5e63a1ec93` |

```powershell
Get-FileHash resultados/local/models/best.pt -Algorithm SHA256
Get-FileHash resultados/local/models/best_v2.pt -Algorithm SHA256
.\iniciar.ps1
```

Abra a [interface](http://127.0.0.1:8501) ou a
[documentação da API](http://127.0.0.1:8000/docs). Para reiniciar instâncias
antigas, execute:

```powershell
.\iniciar.ps1 -Reiniciar
```

Se já tinha `.env`, confira o perfil atual: confiança geral/resíduos 0,35,
animal 0,60, IoU 0,45, imagem YOLO 640, água com pixel 0,95, área mínima 0,05 e
modo `augment`. Imagens usam `pipeline-1314fcfa94f5f8cf`; vídeos acrescentam
confirmação de água por 0,5 s e usam `pipeline-4d613d5a1d82cb61`.
O perfil completo está em
[active_profile.json](../resultados/metrics/v3/active_profile.json). Um treino
repetido pode gerar outro hash; não atribua a versão publicada a pesos novos.

## Preparar os dados v3 em um clone novo

CLICK Trash v2 e Stagnant Water v4 usam CC BY 4.0. O registro iRodent indica
Apache-2.0, mas metadados/imagens de origem indicam CC BY-NC; o projeto mantém a
condição mais restritiva. Clipes e fotos conservam autores, URLs e licenças em
`dados/examples/sources.json`. As regras estão em [VALIDACAO_V3.md](VALIDACAO_V3.md).

Use o ZIP de exemplos da release para reprodução exata: ele preserva os bytes
dos MP4 revisados. `prepare_examples.py` pode reconstruir uma biblioteca das
fontes originais, mas diferenças de FFmpeg entre plataformas podem mudar os
bytes. Os preparadores de contexto recusam vídeos com hash diferente do revisado.

Primeiro obtenha as fontes e reconstrua o v2, base imutável da revisão v3.
Não é necessário reconstruir o v1 defeituoso: o preparador v2 recupera imagens
e labels brutos pelos IDs/hashes do manifesto v1 versionado.

```powershell
python ferramentas/scripts/download_datasets.py --source all --seed 42
python ferramentas/scripts/prepare_negative_crops.py
python ferramentas/scripts/prepare_domain_annotations.py
python ferramentas/scripts/prepare_adaptation.py --domain-root dados/interim/adaptation
```

Isso gera `dados/processed/drenavision_v2/` com 557/62/62 imagens. O v2 corrige
caixas iRodent que incorporavam keypoints ausentes `(0,0,0)`, usando pontos
visíveis e margem de 10%. Pode cobrir só a parte anotada do animal; não é uma
referência humana independente de corpo inteiro.

Em seguida prepare os quadros completos e o conjunto limpo:

```powershell
python ferramentas/scripts/prepare_rat_context_v3.py
python ferramentas/scripts/prepare_water_v3.py --extra-negative 0
python ferramentas/scripts/prepare_v3_dataset.py `
  --domain-root dados/interim/rat_context_v3 `
  --domain-root dados/interim/water_v3 `
  --fetch-click `
  --output dados/processed/drenavision_v3_clean `
  --audit-dir resultados/local/preparation_v3_clean
python ferramentas/scripts/validate_dataset.py --phash `
  --report resultados/local/preparation_v3_clean/validation_report.json
python ferramentas/scripts/visualize_annotations.py
```

`--fetch-click` baixa/verifica 250 pares nativos de treino por HTTP Range,
sem baixar o ZIP inteiro. Os 100 já usados são deduplicados, deixando 150 novos.
A água recebe 400 imagens positivas e oito quadros de contexto adicionais.
Cinco recortes pequenos do rato junto ao lixo são substituídos por cinco
quadros completos com rato e pilha de resíduos revisados. Não entram os 24
negativos de superfície molhada inicialmente considerados. Oito negativos
antigos de treino e dois de cada partição protegida também são removidos: a
fonte pode conter água visível mesmo com rótulo vazio.

O resultado medido tem **1.107/60/60 imagens**. As restantes de validação e teste
mantêm pixels e partições. Caixas de água originais que se tocam ou sobrepõem
formam componentes; o retângulo pode conter espaços secos e não é uma máscara.
A montagem verifica hashes, grupos e pHash. Se uma pasta processada já estiver
preenchida, escolha outro destino vazio e ajuste a configuração; preserve os
arquivos e relatórios históricos.

## Repetir treinamento e avaliação

O v3 parte de `best_v2.pt`, não do ativo `best.pt`. Usa tamanho 640, batch 12,
seed 42, máximo de 80 épocas e patience 20. A execução publicada concluiu
**63 épocas**, selecionando a **época 43**. Outro ambiente pode gerar pesos
diferentes; registre suas versões, manifests e hashes.

```powershell
python ferramentas/scripts/train.py --config ferramentas/configs/train_v3.yaml
python ferramentas/scripts/evaluate.py `
  --config ferramentas/configs/evaluate_v3.yaml `
  --model resultados/local/models/candidate_v3.pt `
  --name candidate_v3_reproduction_test
```

O treino exporta `candidate_v3.pt` e não substitui o ativo automaticamente.
`evaluate.py` usa o teste e mede **somente YOLO**, sem B1 nem os limiares por
classe do produto. A coleta de AP usa confiança 0,001 e NMS 0,60, diferentes dos
limiares da interface. O pipeline completo tem seu perfil em `active_profile.json`
e avaliação própria. Fontes podem omitir água em fotos de animais ou outros
animais presentes; interprete as métricas com os limites documentados.

A calibração de água usa exclusivamente 59 imagens claras de validação revisadas
antes de consultar previsões; uma imagem incerta fica separada. O primeiro grid
falhou no piso de especificidade e foi preservado. A extensão do grid na mesma
validação escolheu pixel 0,95/área 0,05, sem usar o teste adicional para escolher
parâmetros. Consulte
[water_presence_calibration.json](../resultados/metrics/v3/water_presence_calibration.json).
`calibrate_water.py` exige a referência selada e o benchmark YOLO de validação;
não muda rótulos nem procura parâmetros no teste.

O conjunto adicional é reconstruído pela lista selada, sem nova amostragem dos
negativos:

```powershell
python ferramentas/scripts/prepare_extra_test.py
python ferramentas/scripts/prepare_extra_test.py --verify-only `
  --exclude-dataset dados/processed/drenavision_v3_clean
```

São 119 imagens: 69 do teste nativo iRodent, 40 de água e dez negativos revisados.
Elas ficam fora do treino e da calibração, mas pertencem às mesmas fontes
públicas; não equivalem a estudo de campo. Após consultá-las, uma nova escolha
de parâmetros exige outra avaliação reservada.

## Comandos diretos

Defaults de treino, avaliação YOLO, validação, visualização e sequência usam
`drenavision_v3_clean`. Para analisar entradas com o perfil publicado:

```powershell
python ferramentas/scripts/predict.py `
  --input caminho/foto.jpg --model resultados/local/models/best.pt `
  --water-model resultados/local/models/water_segformer_b1_ade `
  --water-pixel-confidence 0.95 --water-min-area-fraction 0.05 `
  --animal-conf 0.60 --water-mode augment `
  --output-dir resultados/local/predictions
python ferramentas/scripts/process_video.py `
  --input caminho/video.mp4 --output resultados/local/videos/analise.mp4 `
  --model resultados/local/models/best.pt `
  --water-model resultados/local/models/water_segformer_b1_ade `
  --water-pixel-confidence 0.95 --water-min-area-fraction 0.05 `
  --animal-conf 0.60 --water-mode augment --water-confirmation-seconds 0.5 --track `
  --database resultados/local/events/drenavision.db `
  --evidence-dir resultados/local/events/evidence --point-id garagem-demo-01
python -m pytest -q
python -m ruff check .
```

Os exemplos em vídeo são filmagens reais. `build_demo_video.py` cria uma
sequência de fotografias do teste, identificada como sequência; não deve ser
apresentada como uma nova filmagem independente.

## Histórico e evidências

Configurações antigas precisam ser solicitadas explicitamente. `train.yaml`
exporta `candidate_v1_reproduction.pt`; `evaluate.yaml` usa `best_v1.pt` e dados
v1. `train_v2.yaml` exporta `candidate_v2.pt`; `evaluate_v2.yaml` usa o backup
`best_v2.pt` e rótulos v2. Não troque checkpoints/labels históricos pelos atuais
para reproduzir números antigos.

Avaliações v1/v2 continuam em `resultados/metrics/` e `resultados/metrics/v2/`.
As regras de caixas mudaram entre protocolos: compare checkpoints nos mesmos
rótulos, sem comparar diretamente mAP antigo e novo. Preserve commit,
configuração, seed, versões, dispositivo, fingerprints, hashes e relatórios
completos em `resultados/local/`.

O [notebook Colab](drenavision_training.ipynb) chama os mesmos preparadores e o
treino v3. Seus outputs estão vazios; foi validado por análise de sintaxe e Ruff,
**não executado no Colab nesta entrega**. Uma execução futura deve obter fontes
e assets autorizados, verificar hashes, passar pelas validações e salvar os
artefatos antes de a sessão expirar.
