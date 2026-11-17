# Dados do DrenaVision

Este diretório guarda configuração, rastreabilidade, manifestos e os exemplos
de teste em `examples/`. Os datasets completos brutos e processados são
obtidos localmente e permanecem fora do Git.
O contrato final é YOLO, com as classes:

| ID | Classe |
|---:|---|
| 0 | `obstrucao_visivel` |
| 1 | `agua_acumulada` |
| 2 | `animal_visivel` |

## Reprodução

Execute a partir da raiz do repositório:

```powershell
python ferramentas/scripts/research_datasets.py --verify-local
python ferramentas/scripts/download_datasets.py --source all --seed 42
python ferramentas/scripts/prepare_negative_crops.py
python ferramentas/scripts/prepare_domain_annotations.py
python ferramentas/scripts/prepare_adaptation.py --domain-root dados/interim/adaptation
python ferramentas/scripts/prepare_rat_context_v3.py
python ferramentas/scripts/prepare_water_v3.py
python ferramentas/scripts/prepare_v3_dataset.py --fetch-click --domain-root dados/interim/rat_context_v3 --domain-root dados/interim/water_v3
python ferramentas/scripts/validate_dataset.py
python ferramentas/scripts/visualize_annotations.py --samples-per-split 6
```

O download convencional é retomável por HTTP Range. O CLICK é um ZIP de vários
gigabytes; o script abre o diretório central remotamente e baixa somente 140
pares da base histórica; `--fetch-click` prepara 250 pares nativos de treino
em uma pasta suplementar, sendo 150 adicionais. Não transfere o ZIP completo. Repetir os comandos é
seguro no downloader: arquivos completos e verificados são reutilizados.
Os preparadores recusam sobrescrever uma saída já preenchida. Para repetir,
escolha destinos vazios (`--output-root` no v2, `--output` no v3).
Use os vídeos exatos versionados/da release: reencodar os originais pode alterar
os hashes exigidos pelos quadros revisados. A instalação detalhada está em
[REPRODUZIR.md](../docs/REPRODUZIR.md).

A conversão simples `convert_annotations.py` cria `drenavision_corrected` com
manifesto separado. Ela não substitui a preparação v2 acima nem sobrescreve
as evidências v1. O v2 corrige caixas de roedores que incorporavam pontos
ausentes na origem e registra esse protocolo em
[resultados/metrics/v2](../resultados/metrics/v2/README.md).

Para uma inspeção mais cara de imagens visualmente semelhantes entre splits:

```powershell
python ferramentas/scripts/validate_dataset.py --phash --phash-threshold 4
```

O validador termina com código diferente de zero se encontrar imagem corrompida,
rótulo ausente/órfão, caixa YOLO inválida, classe fora do contrato, duplicata
exata, vazamento de grupo, divergência no manifesto ou configuração incorreta.

## Estrutura local

```text
dados/
├── dataset.yaml
├── dataset_sources.csv
├── manifests/
│   ├── dataset_manifest.csv       # gerado pela conversão
│   └── dataset_summary.json       # contagens medidas, nunca estimadas
├── examples/                       # oito entradas pequenas versionadas + créditos
├── raw/                            # ignorado pelo Git
│   ├── stagnant_water/
│   ├── click_trash/
│   └── irodent/
└── processed/drenavision_v3_clean/ # ignorado pelo Git; 1107/60/60 imagens
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    ├── dataset.yaml                    # gerado; usado por treino e avaliação
    └── data_manifest.csv               # proveniência específica do v3
```

O `dados/dataset.yaml` versionado é o template legível que aponta para a
saída v3. O preparador grava o YAML efetivo ao lado das imagens em
`dados/processed/drenavision_v3_clean/dataset.yaml`; é esse arquivo portátil que
treino, avaliação e validação consomem por padrão. Os manifests na raiz de
`dados/manifests/` preservam o histórico v1; o manifesto atual tem uma cópia
versionada em `resultados/metrics/v3/data_manifest.csv`. O v2 permanece local e
em `resultados/metrics/v2/` como histórico. Os componentes conectados de água,
as exclusões de negativos e as 119 imagens adicionais têm protocolo próprio em
[VALIDACAO_V3.md](../docs/VALIDACAO_V3.md).

Cada nome processado começa com a fonte e inclui os 16 primeiros caracteres do
SHA-256 da imagem. O manifesto preserva URL, versão, licença, caminho original,
grupo anti-vazamento, dimensões, hash e contagens de caixas.

## Licenças

Água e CLICK são CC BY 4.0 e exigem atribuição. O registro Zenodo do iRodent
declara Apache-2.0, mas os metadados internos e a origem iNaturalist indicam
CC BY-NC 4.0 para as imagens. Por prudência, o conteúdo iRodent é tratado como
**não comercial**. Consulte `dataset_sources.csv` e [REPRODUZIR.md](../docs/REPRODUZIR.md) antes de
redistribuir dados ou pesos derivados.

