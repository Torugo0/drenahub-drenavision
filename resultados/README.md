# Resultados do projeto

Esta pasta contém o que o DrenaVision produz. Para conferir a entrega, comece
por `metrics/` e `demo/`. Para usar a aplicação, o checkpoint fica em
`local/models/`.

| Pasta | Conteúdo | Git |
|---|---|---|
| `metrics/` | métricas oficiais, curvas, matrizes, manifesto do treino e erros selecionados | evidências pequenas versionadas |
| `demo/` | imagens anotadas e manifests da demonstração acadêmica | exemplos selecionados versionados |
| `dataset_preview/` | montagem para conferir visualmente as anotações | seleção versionada |
| `local/models/` | checkpoint `best.pt` utilizado pela aplicação | local; obter por release ou treino |
| `local/videos/` | vídeos analisados e seus resumos JSON | local; gerado pela aplicação |
| `local/events/` | banco SQLite e evidências do histórico | local; gerado pela aplicação |
| `local/predictions/` | análises avulsas executadas por comando | local |
| `local/training/` | logs, checkpoints e snapshots completos de treinamento | local |
| `local/evaluation/` | avaliação completa antes da seleção para publicação | local |
| `local/tmp/` | cópias temporárias de upload/conversão | gerenciada automaticamente |
| `local/logs/` | logs dos serviços iniciados por `iniciar.ps1` | local |

Os exemplos de entrada ficam em `dados/examples/`; aqui ficam as saídas.
As pastas locais são ignoradas pelo Git para não publicar histórico ou duplicar
arquivos grandes. O modelo e os vídeos distribuídos estão na
[release](https://github.com/Torugo0/drenahub-drenavision/releases/tag/v0.2.0-product).

`metrics/evaluation_metrics.json` é a fonte dos números publicados. Os JSONs
contêm hashes e referências à execução original; seus caminhos históricos
podem mencionar a organização anterior do repositório. A reorganização das
pastas não representa um novo treinamento ou uma nova avaliação.
