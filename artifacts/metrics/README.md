# Evidências publicadas de treino e avaliação

`evaluation_metrics.json` é a saída autoritativa da avaliação do checkpoint
`fe439a3a...f6bf7` nas 62 imagens do split de teste. O campo `plots` inventaria
todas as figuras produzidas na execução local. O subconjunto efetivamente
versionado está declarado em `publication_manifest.json`: curvas, matrizes e
um exemplo por `categoria × classe`.

A análise de erros percorreu as 62 imagens. No limiar 0,35 e IoU de pareamento
0,50, a varredura exploratória contou 32 TP, 23 FP e 301 FN. Esses totais servem
para mineração de exemplos e **não substituem** as métricas oficiais calculadas
pela avaliação Ultralytics completa.

Nas imagens de erro, o plot do detector é complementado com ground truth verde,
FN vermelho e FP magenta. A categoria `true_positive` exige ao menos uma
correspondência válida da classe, mas a mesma imagem ainda pode conter outros
erros. Não houve imagem de água integralmente correta no limiar operacional.

As figuras contêm amostras transformadas de CLICK Trash v2 (CC BY 4.0),
Stagnant Water and Wet Surface v4 (CC BY 4.0) e iRodent (tratado como CC BY-NC
4.0). `publication_manifest.json` registra o identificador original de cada
imagem, que pode ser cruzado com `datasets/manifests/dataset_manifest.csv`; veja também
`THIRD_PARTY_NOTICES.md`. O uso é acadêmico e não concede direitos comerciais
sobre iRodent.
