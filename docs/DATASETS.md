# Pesquisa, seleção e governança dos datasets

## Contrato de classes

O DrenaVision usa um único esquema YOLO normalizado:

| ID destino | Nome | Evidência visual esperada |
|---:|---|---|
| 0 | `obstrucao_visivel` | lixo ou material sólido visível que possa obstruir drenagem |
| 1 | `agua_acumulada` | região visível de água parada |
| 2 | `animal_visivel` | animal visível; nesta versão, roedores |

Os rótulos representam apenas caixas publicadas pelas fontes. Nenhum rótulo de
classificação foi promovido artificialmente a bounding box e nenhuma saída de
modelo não revisada foi tratada como ground truth.

## Resultado real da preparação

| Split | Imagens | Obstrução | Água | Animal |
|---|---:|---:|---:|---:|
| treino | 308 | 182 | 989 | 100 |
| validação | 62 | 40 | 223 | 20 |
| teste | 62 | 37 | 276 | 20 |
| **total** | **432** | **259** | **1.488** | **140** |

O dataset final tem 1.887 boxes e 13 imagens negativas. A validação registrou
0 erros, 0 avisos, 0 grupos duplicados por SHA-256 e 0 vazamentos perceptuais
entre splits no pHash com distância de Hamming ≤ 4. O relatório autoritativo
está em `datasets/manifests/validation_report_phash.json`.

## Fontes selecionadas

### Água acumulada — Stagnant Water and Wet Surface v4

- Fonte: [Mendeley Data, DOI 10.17632/y6zyrnxbfm.4](https://data.mendeley.com/datasets/y6zyrnxbfm/4)
- Autores: Kailas Patil e Sonali Bhutad.
- Licença: CC BY 4.0.
- Publicação: 1.976 imagens RGB anotadas em YOLO, com 256 × 256 pixels e vistas
  superior/lateral.
- Mapeamento: `water` (origem 0) → `agua_acumulada` (destino 1).
- Exclusões: diretório `Raw data` e todas as caixas `wet surface` (origem 1).

A seleção determinística usa 100/20/20 imagens positivas. Sequências numéricas
consecutivas são agrupadas em blocos de dez antes do sorteio para que quadros
potencialmente semelhantes não atravessem treino, validação e teste. Além disso,
8/2/2 imagens contendo somente `wet surface` entram com arquivo de rótulo vazio
como negativos difíceis; elas não são falsamente rotuladas como água.

A auditoria do material bruto encontrou um arquivo de rótulo sem imagem
(`image99r.txt`) e três caixas de área zero em duas imagens. O conversor registra
e exclui esses itens inválidos. Também foram removidas 167 ocorrências de imagem
exatamente duplicada antes da amostragem (SHA-256); a saída processada continua
sujeita ao validador estrito.

### Obstrução — CLICK Trash Dataset v2

- Fonte: [Mendeley Data, DOI 10.17632/htcvvz6fm2.2](https://data.mendeley.com/datasets/htcvvz6fm2/2)
- Autor: Manjit Baishya.
- Licença: CC BY 4.0.
- Publicação: 1.126 fotografias externas, com anotações YOLO/COCO; as propostas
  iniciais foram verificadas e corrigidas na fonte.
- Mapeamento: `trash` (origem 0) → `obstrucao_visivel` (destino 0).

O ZIP publicado tem vários gigabytes. `download_datasets.py` usa requisições
HTTP Range por meio do RemoteZip e transfere apenas uma amostra determinística
(seed 42) de 100 imagens do treino, 20 da validação e 20 do teste. Os splits
nativos são preservados. O hash de cada imagem baixada e o nome remoto de seu
rótulo ficam em `datasets/raw/click_trash/sample/selection.json`.

Limitação: o CLICK representa lixo visível em ambientes externos, não confirma
que cada objeto esteja fisicamente bloqueando um ralo. A aplicação deve tratar a
detecção como triagem visual, nunca como comprovação hidráulica.

### Animal — iRodent v1.0

- Fonte: [Zenodo, DOI 10.5281/zenodo.8250392](https://doi.org/10.5281/zenodo.8250392)
- Autores: Shaokai Ye, Anastasiia Filippova, Jessy Lauer, Maxime Vidal, Steffen
  Schneider, Tian Qiu, Alexander Mathis e Mackenzie Weygandt Mathis.
- Formato: COCO, com caixas, keypoints e máscaras publicadas.
- Mapeamento: categoria `rodent` (origem 1) → `animal_visivel` (destino 2).

São escolhidas 100 imagens para treino e 20 para validação a partir do
`train.json` original; as 20 imagens de teste vêm exclusivamente do `test.json`
original. Assim, o conjunto de teste publicado nunca alimenta treino ou seleção
de validação.

Os metadados publicados declaram 443 imagens, mas o arquivo baixado contém 357
entradas em `train.json` e 90 em `test.json` (447 no total). A auditoria COCO
encontrou uma anotação de área zero (`train.json`, anotação 31,
`new_obs_id_00289.jpg`); ela é registrada e excluída, deixando 446 imagens com
caixa utilizável antes da amostragem. A discrepância 443/447 é preservada como
problema de proveniência, não arredondada silenciosamente.

Há uma divergência de licença que exige prudência. A página do registro Zenodo
declara Apache-2.0, enquanto `meta_data.json` e a procedência das fotografias do
iNaturalist indicam CC BY-NC 4.0 para as imagens. O projeto aplica a condição mais
restritiva: uso não comercial, atribuição e revisão jurídica antes de qualquer
produto comercial ou redistribuição de imagens/pesos derivados.

Limitação: o conjunto cobre roedores em ambientes naturais variados, e não é um
dataset específico de ratos em ralos, garagens ou condomínios. A classe permanece
preparada para ampliação com dados revisados do domínio-alvo.

## Fontes investigadas e não utilizadas

O conjunto obrigatório
[Roboflow `drain_data`](https://universe.roboflow.com/-jd8tk/drain_data) foi
inspecionado em 16/11/2026. A página mostrava 139 imagens de prévia, classes
`debris` e `drain`, licença marcada como domínio público e **zero versões de
dataset**. Sem uma versão exportável, não havia um pacote reproduzível de imagens
e anotações para baixar; ele foi excluído sem contornar autenticação ou limites
da plataforma.

Outras buscas foram registradas sem contornar autenticação:

| Fonte | Evidência observada | Decisão |
|---|---|---|
| [Roboflow: busca UncleanDrain](https://universe.roboflow.com/search?q=UncleanDrain) | projetos de drenagem encontrados; exportação exigia login | não baixado; licença deve ser verificada por versão |
| [Roboflow: busca rat detection](https://universe.roboflow.com/search?q=rat%20detection) | múltiplos projetos de detecção de ratos; exportação exigia login | iRodent preferido por ter arquivo direto e versão fixa |
| [Roboflow: busca water leakage](https://universe.roboflow.com/search?q=water%20leakage) | projetos de água/superfície molhada; exportação exigia login | Mendeley v4 preferido por reprodutibilidade |
| [Open Images V7](https://storage.googleapis.com/openimages/web/download.html), classe Mouse `/m/04rmv` | bounding boxes pesquisáveis; anotações CC BY 4.0, licença da imagem por item | não baixado: seria necessária auditoria individual de direitos |
| [Hugging Face DAMM mouse detection](https://huggingface.co/datasets/gauravkaul/DAMM_mouse_detection) | caixas/máscaras estilo COCO, CC BY 4.0, muitos frames de laboratório/vídeo | reservado; exigiria split por sequência e tem menor aderência ao domínio |
| [Kaggle 26 Class Object Detection](https://www.kaggle.com/datasets/mohamedgobara/26-class-object-detection-dataset) | download público funcionou e inclui `Rat`; metadado Kaggle indica Apache-2.0 | excluído: direitos VIAD/Roboflow não reconciliados, augmentations quase duplicadas e gap de domínio |
| [Mendeley Floating Waste v1](https://data.mendeley.com/datasets/j26w4m645z/1) | caixas YOLO `floating_waste`, CC BY 4.0; fotos e frames de vídeo | alternativa futura; lixo flutuante é menos aderente a obstrução sólida de ralo e exige agrupamento temporal |

Essas consultas são descoberta, não autorização automática de uso. Para esta
primeira versão foram preferidas as três fontes selecionadas por oferecerem
download direto, anotações de detecção, versão fixa e licença rastreável. Novas
fontes só devem entrar após a mesma verificação de formato, proveniência e
direitos.

## Conversão, deduplicação e prevenção de vazamento

1. O mapeamento de classes é aplicado antes da escrita e todo rótulo final passa
   pelo parser YOLO estrito (cinco campos, números finitos, classe válida, área
   positiva e caixa dentro da imagem).
2. Arquivos finais recebem prefixo da fonte e fragmento do SHA-256, eliminando
   colisões de nomes entre datasets.
3. Imagens idênticas por SHA-256 são proibidas, inclusive dentro do mesmo split.
4. O manifesto impede que um mesmo `source_group` apareça em splits diferentes.
5. A validação opcional por pHash (DCT, 64 bits) detecta imagens quase idênticas
   entre splits com limiar configurável.
6. O relatório contabiliza imagens, caixas e negativos a partir dos arquivos
   realmente produzidos. Metas de amostragem não são apresentadas como métricas.

O formato final é:

```text
datasets/processed/drenavision/
├── images/{train,val,test}/
├── labels/{train,val,test}/
└── dataset.yaml
```

`datasets/dataset.yaml` é um template versionado que aponta para essa saída. A
conversão grava também o arquivo efetivo
`datasets/processed/drenavision/dataset.yaml`, sem caminhos absolutos; treino,
avaliação e validação usam o arquivo efetivo por padrão.

## Reprodutibilidade e auditoria

```powershell
python scripts/research_datasets.py --verify-local --output datasets/manifests/research_report.json
python scripts/download_datasets.py --source all --seed 42
python scripts/convert_annotations.py --seed 42
python scripts/validate_dataset.py --report datasets/manifests/validation_report.json
python scripts/validate_dataset.py --phash --phash-threshold 4
python scripts/visualize_annotations.py --samples-per-split 6
```

`dataset_sources.csv` é a trilha de fontes. `dataset_manifest.csv` é a trilha por
imagem. `dataset_summary.json` e o relatório do validador contêm somente medidas
calculadas na execução local.

