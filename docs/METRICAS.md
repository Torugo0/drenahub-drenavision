# Métricas e análise de erros

## Regra de publicação

Os valores abaixo foram gerados por uma avaliação real e não foram estimados.
O arquivo autoritativo versionado é
`artifacts/metrics/evaluation_metrics.json`; a execução local completa fica em
`artifacts/evaluation/test_yolo26n_drenavision/metrics.json`.

```powershell
python scripts/evaluate.py --config configs/evaluate.yaml
```

## Métricas reportadas

| Métrica | Interpretação | Cuidado |
|---|---|---|
| Precision | proporção de detecções aceitas que correspondem a objetos anotados | pode subir ao custo de recall |
| Recall | proporção dos objetos anotados que foi detectada | depende da qualidade dos labels |
| mAP@50 | AP médio com IoU 0,50 | tolera localização menos precisa |
| mAP@50-95 | AP médio em IoUs 0,50 a 0,95 | mais rigoroso na localização |
| Matriz de confusão | confusões entre classes e fundo | interpretar com suporte por classe |
| Tempo por imagem | pré-processamento, inferência e pós-processamento | depende de hardware, batch e warm-up |

Apresente valores agregados e para cada uma das três classes. Uma média pode
ocultar baixo desempenho de `animal_visivel`.

## Protocolo

- usar somente o split `test` independente;
- registrar checkpoint/dataset por SHA-256;
- registrar `imgsz`, batch, confidence, IoU, dispositivo e versões;
- não ajustar o modelo depois de olhar o teste sem criar nova versão;
- guardar matriz, curvas e exemplos de erro com o JSON;
- informar número de imagens/objetos por classe.

## Exemplos de erro

O avaliador deve exportar:

- verdadeiro positivo representativo;
- falso positivo por classe quando existir;
- falso negativo por classe quando existir;
- casos com baixa confiança/oclusão;
- confusão água/reflexo e resíduo/textura;
- falhas de animal pequeno, distante ou parcialmente oculto.

Não escolha somente imagens favoráveis. Ausência de um tipo de erro em amostra
pequena não prova que ele não ocorre.

## Comparação com o modelo original

O detector COCO original e o DrenaVision não compartilham o mesmo espaço de
classes. Uma comparação direta de mAP é geralmente inválida. Se for útil,
compare latência/tamanho e mostre qualitativamente que pesos genéricos não
fornecem o contrato das três classes. Não renomeie classes COCO para fingir um
baseline equivalente.

## Resultado no teste separado

| Classe | Suporte no teste | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|---:|
| `obstrucao_visivel` | 37 | 0,5091 | 0,6728 | 0,5860 | 0,4046 |
| `agua_acumulada` | 276 | 0,3999 | 0,1159 | 0,1642 | 0,0360 |
| `animal_visivel` | 20 | 0,4250 | 0,6500 | 0,3843 | 0,2181 |
| agregado | 333 | 0,4447 | 0,4796 | 0,3781 | 0,2196 |

Protocolo: 62 imagens, `imgsz=640`, batch 8, RTX 4060, checkpoint SHA-256
`fe439a3a722e73db5d841cc199026c8d42f4bbf806bf95924c2eb8c09a9f6bf7` e
dataset SHA-256
`e0c605ead9c68936bdd0cdfe5ef400a8e7e398ec7e116b2f51f49d73aeea2772`.
A inferência do modelo levou 3,39 ms/imagem; o valor não inclui I/O, rede,
persistência ou renderização da interface.

## Análise de erros executada

A mineração qualitativa percorreu as 62 imagens, estratificou exemplos por
`categoria × classe` e publicou um TP, FP e FN de cada classe. No limiar 0,35 e
IoU de pareamento 0,50, a varredura contou 32 TP, 23 FP e 301 FN. Esses totais
dependem do limiar e do pareamento guloso; não são as métricas oficiais de AP.

Não houve imagem de `agua_acumulada` integralmente correta no limiar 0,35. Há
três imagens com ao menos um TP de água, mas elas também contêm erros. O recall
0,1159 e mAP@50-95 0,0360 tornam água a principal limitação medida. Animal tem
apenas 20 instâncias de teste e forte concentração em roedores, portanto a
incerteza é alta.

## Limiares operacionais

O limiar da API (padrão 0,35) não precisa ser igual ao usado para construir a
curva de avaliação. Escolha o ponto operacional na validação, considerando o
custo de perder uma anomalia versus gerar inspeções extras. Confiança não é
probabilidade de risco hidráulico.

