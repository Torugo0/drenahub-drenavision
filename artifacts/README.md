# Artefatos e saídas

Esta pasta reúne **resultados produzidos pelo projeto**. Ela não contém código
da aplicação.

## Evidências versionadas

| Pasta | Conteúdo |
|---|---|
| `dataset_preview/` | montagem pequena para conferir visualmente as anotações |
| `demo/` | exemplos de inferência e manifests usados na demonstração |
| `metrics/` | métricas, curvas, matrizes e evidências do treino real |

## Saídas locais não versionadas

| Pasta | Conteúdo | Como recriar |
|---|---|---|
| `models/` | `best.pt` e outros checkpoints | treino ou release do GitHub |
| `videos/` | vídeos de entrada/saída e JSON sidecar | release, `build_demo_video.py` e `process_video.py` |
| `events/` | banco SQLite e evidências da API/UI | executar API e interface |
| `predictions/` | imagens anotadas avulsas | `predict.py` |
| `training/` | runs completos e snapshots | `train.py` |
| `evaluation/` | avaliação completa antes da curadoria | `evaluate.py` |
| `tmp/` | arquivos temporários de upload | gerenciada automaticamente |

Essas saídas locais são ignoradas pelo Git para evitar duplicação de arquivos
grandes. Não mova manualmente arquivos entre as pastas: use os scripts, que
também registram hashes e proveniência.
