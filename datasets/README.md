# Dados do DrenaVision

Este diretório guarda apenas configuração, rastreabilidade e manifestos. Os
arquivos brutos e processados são obtidos localmente e permanecem fora do Git.
O contrato final é YOLO, com as classes:

| ID | Classe |
|---:|---|
| 0 | `obstrucao_visivel` |
| 1 | `agua_acumulada` |
| 2 | `animal_visivel` |

## Reprodução

Execute a partir da raiz do repositório:

```powershell
python scripts/research_datasets.py --verify-local
python scripts/download_datasets.py --source water
python scripts/download_datasets.py --source click --seed 42 --click-train 100 --click-val 20 --click-test 20
python scripts/download_datasets.py --source irodent
python scripts/convert_annotations.py --seed 42
python scripts/validate_dataset.py
python scripts/visualize_annotations.py --samples-per-split 6
```

O download convencional é retomável por HTTP Range. O CLICK é um ZIP de vários
gigabytes; o script abre o diretório central remotamente e baixa somente 140
pares selecionados, sem transferir o arquivo completo. Repetir os comandos é
seguro: arquivos completos e verificados são reutilizados.

Para uma inspeção mais cara de imagens visualmente semelhantes entre splits:

```powershell
python scripts/validate_dataset.py --phash --phash-threshold 4
```

O validador termina com código diferente de zero se encontrar imagem corrompida,
rótulo ausente/órfão, caixa YOLO inválida, classe fora do contrato, duplicata
exata, vazamento de grupo, divergência no manifesto ou configuração incorreta.

## Estrutura local

```text
datasets/
├── dataset.yaml
├── dataset_sources.csv
├── manifests/
│   ├── dataset_manifest.csv       # gerado pela conversão
│   └── dataset_summary.json       # contagens medidas, nunca estimadas
├── raw/                            # ignorado pelo Git
│   ├── stagnant_water/
│   ├── click_trash/
│   └── irodent/
└── processed/drenavision/          # ignorado pelo Git
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    └── dataset.yaml                    # gerado; usado por treino e avaliação
```

O `datasets/dataset.yaml` versionado é o template legível que aponta para a
saída padrão. O conversor sempre grava o YAML efetivo ao lado das imagens em
`datasets/processed/drenavision/dataset.yaml`; é esse arquivo portátil, sem
caminho absoluto, que treino, avaliação e validação consomem por padrão.

Cada nome processado começa com a fonte e inclui os 16 primeiros caracteres do
SHA-256 da imagem. O manifesto preserva URL, versão, licença, caminho original,
grupo anti-vazamento, dimensões, hash e contagens de caixas.

## Licenças

Água e CLICK são CC BY 4.0 e exigem atribuição. O registro Zenodo do iRodent
declara Apache-2.0, mas os metadados internos e a origem iNaturalist indicam
CC BY-NC 4.0 para as imagens. Por prudência, o conteúdo iRodent é tratado como
**não comercial**. Consulte `dataset_sources.csv` e `docs/DATASETS.md` antes de
redistribuir dados ou pesos derivados.

