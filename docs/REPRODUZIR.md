# Instalação e reprodução

Execute os comandos na raiz do repositório. A versão 0.5.0 mantém o YOLO
customizado v3 e o SegFormer B1 de água e acrescenta Grounding DINO Tiny com
verificação de recortes SigLIP 2. A revisão muda a inferência, sem novo treino.
Para usar o produto, instale as dependências e prepare os modelos. Baixar
datasets pertence à reprodução acadêmica, separada do uso normal.

Neste computador, a API 0.5.0 está ativa e a suíte passou em 239 testes.
O benchmark de 219 imagens está concluído em coortes separadas: CLICK40
reservado e 179 regressões conhecidas. A verificação funcional dos seis vídeos,
duas fotos do catálogo e fluxo real na interface foi concluída.

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

Os pesos YOLO v3, a água B1 e os exemplos continuam na
[release v0.4.0-pipeline-v3](https://github.com/Torugo0/drenahub-drenavision/releases/tag/v0.4.0-pipeline-v3).
Para o repositório público, os assets podem ser baixados diretamente por HTTP,
sem login, token ou instalação do GitHub CLI. Também é possível usar o botão
de download de cada asset na página da release. Em uma atualização, preserve
seus modelos e `.env` anteriores.

```powershell
New-Item -ItemType Directory -Force resultados/local/release/v0.4.0 | Out-Null
New-Item -ItemType Directory -Force resultados/local/models | Out-Null
New-Item -ItemType Directory -Force dados/examples | Out-Null
$releaseUrl = "https://github.com/Torugo0/drenahub-drenavision/releases/download/v0.4.0-pipeline-v3"
foreach ($artifactName in "best.pt", "best_v2.pt", "water_model.zip", "exemplos_reais.zip") {
  Invoke-WebRequest -Uri "$releaseUrl/$artifactName" `
    -OutFile (Join-Path "resultados/local/release/v0.4.0" $artifactName)
}
Copy-Item resultados/local/release/v0.4.0/best.pt resultados/local/models/best.pt
Copy-Item resultados/local/release/v0.4.0/best_v2.pt resultados/local/models/best_v2.pt
Expand-Archive resultados/local/release/v0.4.0/water_model.zip `
  -DestinationPath resultados/local/models/water_segformer_b1_ade
Expand-Archive resultados/local/release/v0.4.0/exemplos_reais.zip `
  -DestinationPath dados/examples -Force
python ferramentas/scripts/prepare_water_model.py
python ferramentas/scripts/prepare_object_model.py
python ferramentas/scripts/prepare_verifier_model.py
```

Quem já utiliza o GitHub CLI pode substituir somente o bloco de download por
esta alternativa opcional, mantendo as etapas de cópia, extração e preparação:

```powershell
gh release download v0.4.0-pipeline-v3 `
  --repo Torugo0/drenahub-drenavision `
  --pattern best.pt --pattern best_v2.pt `
  --pattern water_model.zip --pattern exemplos_reais.zip `
  --dir resultados/local/release/v0.4.0
```

O caminho principal de instalação acima não depende de autenticação no GitHub.

`best_v2.pt` é necessário para repetir o treino. O uso normal 0.5 utiliza
`best.pt` e três pastas: `water_segformer_b1_ade`, `grounding_dino_tiny` e
`siglip2_base_patch16_224`. Os dois novos preparadores usam as revisões
oficiais fixadas, verificam hash/tamanho e não sobrescrevem uma pasta não vazia
incompatível. O download adicional é aproximadamente **2,2 GB**: 689 MB de
pesos Grounding DINO e 1,5 GB de SigLIP 2, além dos tokenizadores/configurações.
Eles já estão presentes neste computador; repetir a preparação os verifica.
Em outro computador, faça a preparação antes de iniciar os serviços.

O ZIP de água tem exatamente cinco arquivos na raiz:
`model.safetensors`, `config.json`, `preprocessor_config.json`,
`download_manifest.json` e `LICENSE-NVIDIA-SegFormer.txt`. Não crie uma pasta
extra dentro de `water_segformer_b1_ade/` ao extrair. O preparador verifica uma
instalação correta; se ela estiver ausente, obtém a revisão oficial fixada e
reproduz os mesmos cinco arquivos. **A API e a interface não baixam modelos
durante análises**: executam os arquivos locais preparados. Um componente
configurado ausente ou incompatível deve ser corrigido antes de analisar.

O SegFormer NVIDIA acompanha licença de **pesquisa e avaliação não comercial**.
Esta distribuição acadêmica não concede uso comercial dos pesos. A licença
integral e a origem estão no pacote e em
`resultados/metrics/v3/water_model_source/`.

| Artefato | SHA-256 esperado |
|---|---|
| YOLO v3 ativo, `best.pt` | `12b9e86f223366c3f2ef8c2308f0e1140cda08c55ab55e281e58d300fd554b4b` |
| Inicial histórico v2, `best_v2.pt` | `01ea7c578783b2a08fc585051b0768a1e9c0d510cb5652173fab0979f8d7d27a` |
| Água B1, `model.safetensors` | `e0802e077c195d9dbaf48c69b94946ef6a3ae5b6665176fedf363e5e63a1ec93` |
| Grounding DINO Tiny, `model.safetensors` | `1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3` |
| SigLIP 2, `model.safetensors` | `612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b` |

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

Se outro aplicativo já usar 8501, preserve-o e escolha outra porta para a
interface: `.\iniciar.ps1 -PortaInterface 8502`. A API continua na porta
configurada em `.env`; esse parâmetro altera somente o endereço da tela.

Confira `http://127.0.0.1:8000/health`: no ambiente validado retornou
`status: "ok"`, `model_loaded: true` e `version: "0.5.0"`. A interface envia
fotos e vídeos à API por HTTP. As redes grandes são carregadas uma vez no
serviço; só o rastreador YOLO pequeno tem estado próprio por vídeo. A UI não
abre subprocesso com outra cópia de Grounding DINO, SigLIP 2 ou SegFormer.

Para testar vídeo diretamente, aguarde a resposta do POST e use `video_url`
para obter o MP4 anotado (a requisição só termina após o processamento):

```powershell
curl.exe -X POST http://127.0.0.1:8000/vision/predict/video `
  -F "point_id=verificacao-local" `
  -F "file=@C:\dados\video.mp4;type=video/mp4"
curl.exe http://127.0.0.1:8000/vision/videos/ID_RETORNADO_NO_POST `
  --output analise.mp4
```

`ID_RETORNADO_NO_POST` deve ser substituído pelo `event_id` real. Uploads
aceitos são MP4, MOV, AVI e MKV com MIME compatível. O limite padrão é 500 MB;
os temporários de entrada são removidos ao final. Veja os contratos e erros
em [TECNICO.md](TECNICO.md#api).

Se já tinha `.env`, compare seus valores com `.env.example` e com
[inference_v4.json](../ferramentas/configs/inference_v4.json); não apague
configurações próprias sem conferir. O YOLO mantém confiança base/resíduos
0,35, animal 0,60, IoU 0,45 e tamanho 640. Água mantém pixel 0,95, área mínima
0,05 e modo `augment`. O especialista usa score de resíduos 0,30/animal 0,25,
texto 0,20 e arestas 640/1024; o verificador exige margens 0,04/0,08. O modo
`animal_replace` complementa resíduos e substitui as caixas de animal do mestre.

O perfil de imagem congelado é **`pipeline-4c5c993223219929`**, conforme
[frozen_candidate.json](../resultados/metrics/v4/frozen_candidate.json).
As identidades e a revisão do holdout foram seladas antes desse congelamento.
A calibração usou a validação; o perfil ficou fixado antes das previsões
reservadas, que não foram usadas para retuná-lo. Os horários e escopos estão
em [VALIDACAO_V4.md](VALIDACAO_V4.md).
O [perfil ativo](../resultados/metrics/v4/active_profile.json) reúne estado,
hashes e provas funcionais da implantação local verificada.
O vídeo mantém a confirmação de água por 0,5 s e registra seu perfil adicional.
Não atribua o perfil 0.5 a uma execução que desative os componentes ou altere
pesos/parâmetros. Os perfis 0.4 `pipeline-1314fcfa94f5f8cf` e
`pipeline-4d613d5a1d82cb61` continuam históricos em
[active_profile.json v3](../resultados/metrics/v3/active_profile.json).

Depois de preparar os arquivos, reinicie os serviços para sair de uma instância
antiga. A primeira análise pode incluir o carregamento dos componentes; a
latência mediana de 0,965 s registrada no novo teste CLICK40, ante 0,615 s no
perfil 0.4, é uma medição por foto na RTX 4060 e não uma promessa para CPU ou vídeo.
Consulte [VALIDACAO_V4.md](VALIDACAO_V4.md) para escopos e resultados.

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

## Repetir o treinamento histórico v3 e sua avaliação

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
`evaluate.py` usa o teste e mede **somente YOLO**, sem B1, Grounding DINO,
SigLIP 2 nem os limiares por classe do produto. A coleta de AP usa confiança
0,001 e NMS 0,60, diferentes dos limiares da interface. O pipeline 0.5 tem seu
perfil congelado em `resultados/metrics/v4/frozen_candidate.json` e avaliação
própria. Fontes podem omitir água em fotos de animais ou outros
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
`drenavision_v3_clean`. Para analisar entradas com os componentes 0.5:

```powershell
python ferramentas/scripts/predict.py `
  --input caminho/foto.jpg --model resultados/local/models/best.pt `
  --imgsz 640 --conf 0.35 --iou 0.45 `
  --object-model resultados/local/models/grounding_dino_tiny `
  --object-verifier resultados/local/models/siglip2_base_patch16_224 `
  --object-residue-confidence 0.30 --object-animal-confidence 0.25 `
  --object-residue-min-margin 0.04 --object-animal-min-margin 0.08 `
  --object-text-confidence 0.20 --object-mode animal_replace `
  --object-shortest-edge 640 --object-longest-edge 1024 `
  --water-model resultados/local/models/water_segformer_b1_ade `
  --water-pixel-confidence 0.95 --water-min-area-fraction 0.05 `
  --animal-conf 0.60 --water-mode augment `
  --output-dir resultados/local/predictions
python ferramentas/scripts/process_video.py `
  --input caminho/video.mp4 --output resultados/local/videos/analise.mp4 `
  --model resultados/local/models/best.pt `
  --imgsz 640 --conf 0.35 --iou 0.45 --residue-conf 0.35 `
  --object-model resultados/local/models/grounding_dino_tiny `
  --object-verifier resultados/local/models/siglip2_base_patch16_224 `
  --object-residue-confidence 0.30 --object-animal-confidence 0.25 `
  --object-residue-min-margin 0.04 --object-animal-min-margin 0.08 `
  --object-text-confidence 0.20 --object-mode animal_replace `
  --object-shortest-edge 640 --object-longest-edge 1024 `
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

Os defaults também leem `.env`; os caminhos e parâmetros acima explicitam os
componentes adicionais, mas a identidade final depende de todos os valores
efetivos. `--no-objects` permite solicitar uma execução sem o especialista
quando esse diagnóstico é necessário; tal execução não é o perfil 0.5 completo.
API e interface usam o runtime compartilhado e os mesmos componentes/limiares.
A execução por CLI acima é uma alternativa técnica e carrega seu próprio processo: evite
rodá-la junto dos serviços em uma máquina com memória limitada.

O relatório [product_validation.json](../resultados/metrics/v4/product_validation.json)
concluiu os oito itens do catálogo, com 1.388 quadros integralmente decodificados,
H.264/início rápido, HTTP Range 206, persistência de eventos e API saudável.
O perfil de vídeo é `pipeline-380203274bf9c28c`, baseado no perfil de imagem
`pipeline-4c5c993223219929`. Em
[ui_verification.json](../resultados/metrics/v4/ui_verification.json), a prova
de tela cobre as duas fotos do usuário e Hanoi reproduzido até 12/12 segundos.

Essas verificações são funcionais e de regressão conhecida, sem referência
completa de acurácia por quadro. Rua teve só uma marcação animal ambígua em
240 quadros, sem acerto confirmado; Hanoi teve 144/240, contra 187/240 na 0.4.
Não há alegação de ganho temporal. A revisão confirmou falsos resíduos em
lixeira fixa/PVC instalado, além de ambiguidades com vegetação, perdas de
outros resíduos/água rasa em `ralo_lixo` e água incluindo concreto em Canal.
A gravação narrada anterior continua 0.4;
os testes atuais não produzem automaticamente uma nova gravação pública.

## Histórico e evidências

Configurações antigas precisam ser solicitadas explicitamente. `train.yaml`
exporta `candidate_v1_reproduction.pt`; `evaluate.yaml` usa `best_v1.pt` e dados
v1. `train_v2.yaml` exporta `candidate_v2.pt`; `evaluate_v2.yaml` usa o backup
`best_v2.pt` e rótulos v2. Não troque checkpoints/labels históricos pelos atuais
para reproduzir números antigos.

Avaliações v1/v2 continuam em `resultados/metrics/` e `resultados/metrics/v2/`.
As métricas e o treinamento v3 permanecem em `resultados/metrics/v3/`; a revisão
de inferência 0.5, os componentes fixados, a calibração e o novo teste ficam em
`resultados/metrics/v4/`. A gravação local de 9min12s registra a **versão 0.4.0**
e deve ser identificada como histórica, sem ser usada como evidência da 0.5.
As regras de caixas mudaram entre protocolos: compare checkpoints nos mesmos
rótulos, sem comparar diretamente mAP antigo e novo. Preserve commit,
configuração, seed, versões, dispositivo, fingerprints, hashes e relatórios
completos em `resultados/local/`.

O [notebook Colab](drenavision_training.ipynb) chama os mesmos preparadores e o
treino v3. Seus outputs estão vazios; foi validado por análise de sintaxe e Ruff,
**não executado no Colab nesta entrega**. Uma execução futura deve obter fontes
e assets autorizados, verificar hashes, passar pelas validações e salvar os
artefatos antes de a sessão expirar.

## Documento técnico e arquivos da entrega

O [PDF técnico](relatorio/DrenaVision_Applied_Computer_Vision_FIAP.pdf), a
[fonte editável](relatorio/relatorio_fonte.md) e o
[checklist](relatorio/checklist_requisitos.md) ficam em `docs/relatorio/`.
O código e essa documentação são disponibilizados no
[repositório GitHub](https://github.com/Torugo0/drenahub-drenavision).

A entrega é organizada em três arquivos separados, sem ZIP:

- `demo_drena_vision.mp4`: vídeo original de 2min13,20s, com trilha silenciosa,
  entregue como anexo, sem edição ou recodificação.
- `LINK_REPOSITORIO.txt`: endereço do repositório e do PDF, nomes/RMs e instruções.
- `DrenaVision_Applied_Computer_Vision_FIAP.pdf`: relatório técnico, em anexo e
  também na documentação do GitHub.

As cópias locais de entrega ficam em `resultados/local/entrega/`, pasta ignorada
pelo Git. O vídeo não precisa ser obtido pelo repositório. A gravação narrada
de 9min12s permanece histórica da versão 0.4, separada do vídeo recente.
