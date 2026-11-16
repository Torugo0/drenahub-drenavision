# Model Card - DrenaVision YOLO26n

## Estado em 16/11/2026

Existe um checkpoint customizado real em `artifacts/models/best.pt`. O treino
foi retomado de um run interrompido na epoca 28 e terminou por early stopping
apos 70 de 100 epocas. O snapshot interrompido foi preservado separadamente e
nao e apresentado como run concluido.

- dispositivo: NVIDIA GeForce RTX 4060 (`cuda:0`);
- seed: 42;
- SHA-256 do YOLO26n inicial:
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`;
- SHA-256 do `best.pt` publicado:
  `fe439a3a722e73db5d841cc199026c8d42f4bbf806bf95924c2eb8c09a9f6bf7`;
- manifesto versionado: `artifacts/metrics/training_manifest.json`.

Os hashes diferentes e o avanco de epocas comprovam que houve ajuste de pesos.
O checkpoint e adequado a demonstracao academica, mas os resultados abaixo nao
sustentam uso autonomo ou producao.

## Uso pretendido

O detector transforma imagens de pontos de drenagem em evidencias para revisao
humana. As classes sao fixas e ordenadas:

| ID | Classe | Significado permitido |
|---:|---|---|
| 0 | `obstrucao_visivel` | residuos, folhas ou material visivel |
| 1 | `agua_acumulada` | agua visivelmente acumulada, sem medida de profundidade |
| 2 | `animal_visivel` | animal presente, sem diagnostico de obstrucao |

O modelo nao preve alagamento, vazao, profundidade, risco hidraulico ou causa de
uma ocorrencia. Toda deteccao recomenda apenas inspecao.

## Arquitetura e ajuste

- arquitetura inicial: Ultralytics YOLO26n (`yolo26n.pt`);
- tarefa: deteccao por bounding boxes;
- pesos iniciais: pre-treinados, seguidos de fine-tuning nas tres classes;
- resolucao padrao: 640 px;
- seed obrigatoria: 42;
- dispositivo: primeira GPU CUDA disponivel, com fallback explicito para CPU;
- checkpoint publicado: `artifacts/models/best.pt`;
- configuracao versionada: `configs/train.yaml`.

Cada treino concluido grava hash SHA-256 dos pesos iniciais, `best.pt` e copia
publicada, versoes do ambiente, fingerprint do dataset, argumentos efetivos e
metricas finais em `training_manifest.json`. Hashes iguais entre peso inicial e
`best.pt` invalidam a alegacao de fine-tuning.

Uma execucao interrompida pode ser retomada com `scripts/train.py --resume
<run>/weights/last.pt`. Antes de continuar, o pipeline copia um snapshot marcado
como `interrupted`, valida a epoca/meta e so publica o modelo quando a retomada
avanca e altera os pesos.

## Dados e separacao

O dataset padrao e `datasets/processed/drenavision/dataset.yaml`. A execucao
exige `train`, `val` e `test`, mapa de classes exato e ao menos uma imagem em
cada split. O fingerprint inclui imagens, labels e YAML. O agrupamento por
origem/video e a verificacao de vazamento pertencem ao pipeline de dados e
devem passar antes do treino. A avaliacao registrou validacao estrutural sem
erros ou avisos e nenhum grupo de duplicatas SHA-256.

| Split | Imagens | Boxes de obstrucao | Boxes de agua | Boxes de animal |
|---|---:|---:|---:|---:|
| Treino | 308 | 182 | 989 | 100 |
| Validacao | 62 | 40 | 223 | 20 |
| Teste | 62 | 37 | 276 | 20 |

O conjunto completo possui 13 imagens negativas. O fingerprint SHA-256 do
dataset avaliado e
`e0c605ead9c68936bdd0cdfe5ef400a8e7e398ec7e116b2f51f49d73aeea2772`.

Fontes atualmente preparadas pelo projeto incluem agua parada (CC BY 4.0),
CLICK Trash (CC BY 4.0) e iRodent. O conjunto iRodent traz conflito de
licenciamento entre o registro e as imagens-fonte (CC BY-NC); seu uso comercial
depende de revisao e ele nao deve ser tratado como permissao comercial
automatica. Imagens proprias e negativas representativas continuam essenciais.

## Protocolo de avaliacao

`python scripts/evaluate.py --config configs/evaluate.yaml` usa apenas `test` e
exporta:

- precision e recall agregados;
- mAP@50 e mAP@50-95 agregados;
- as quatro metricas por classe;
- tempo por imagem;
- matriz de confusao e graficos Ultralytics;
- exemplos reais corretos, falsos positivos e falsos negativos;
- hashes do modelo e dataset, configuracao e versoes.

Valores ausentes devem permanecer como “nao avaliados”; nunca devem ser
substituidos por zero, estimativas ou resultados sinteticos.

### Resultado real no split de teste

Fonte unica desta tabela:
`artifacts/metrics/evaluation_metrics.json`. A execucao usou
62 imagens, 333 instancias anotadas, `imgsz=640` e a mesma RTX 4060. O hash do
checkpoint conferiu com o publicado.

| Escopo | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---:|---:|---:|---:|
| Agregado | 0,4447 | 0,4796 | 0,3781 | 0,2196 |
| `obstrucao_visivel` | 0,5091 | 0,6728 | 0,5860 | 0,4046 |
| `agua_acumulada` | 0,3999 | 0,1159 | 0,1642 | 0,0360 |
| `animal_visivel` | 0,4250 | 0,6500 | 0,3843 | 0,2181 |

O tempo de inferencia do modelo foi 3,39 ms por imagem; preprocessamento e
pos-processamento consumiram 0,75 ms e 1,48 ms, respectivamente. Esses tempos
nao incluem upload, persistencia, renderizacao da interface ou rede.

A avaliacao tambem produziu curvas, matriz de confusao, predicoes JSON e
exemplos estratificados por categoria e classe. A analise percorreu as 62
imagens e, no limiar 0,35 com IoU de pareamento 0,50, contabilizou 32 TP, 23 FP
e 301 FN. Esses totais exploratorios dependem do limiar e nao substituem as
metricas oficiais do split completo. Nao houve imagem de agua integralmente
correta nesse limiar, embora tres imagens tenham ao menos um TP de agua.

## Limitacoes e riscos conhecidos

- agua rasa, reflexos, piso brilhante e sombras podem se confundir;
- `agua_acumulada` teve recall 0,1159 e mAP@50-95 0,0360; e a principal
  limitacao medida e precisa de mais diversidade, revisao de boxes e dados do
  dominio real de drenagem;
- residuos parcialmente ocultos ou fora do ralo podem ter baixa cobertura;
- `animal_visivel` teve somente 20 instancias no teste; as metricas possuem alta
  incerteza e o conjunto e concentrado em roedores;
- as fontes publicas representam lixo, agua parada e roedores, mas nao cobrem
  suficientemente ralos, grelhas, garagens e canais da implantacao pretendida;
- dominios, cameras, alturas, iluminacoes e climas nao vistos reduzem a
  generalizacao;
- tracking perde ou troca IDs em oclusoes; quando nao ha IDs, a contagem de
  video usa o maximo simultaneo conservador, nunca a soma entre frames;
- o processador recusa videos acima de 18.000 frames por padrao; o limite e
  ajustavel por `--max-frames` e fica registrado no JSON sidecar;
- imagens podem conter pessoas, placas e enderecos, exigindo minimizacao,
  retencao controlada e revisao LGPD.

## Criterios minimos antes de demonstrar

1. concluido: validar estrutura, proveniencia e licencas conhecidas;
2. concluido: treino com manifesto e hash diferente do peso inicial;
3. concluido: avaliacao no split de teste separado;
4. concluido: exportacao de exemplos corretos, FP e FN;
5. concluido para a demonstracao automatica: sequencia de imagens reais do
   teste processada como video anotado, com origem e limitacoes declaradas;
6. pendente para a entrega: gravar a apresentacao narrada e, idealmente,
   acrescentar filmagem propria do dominio de drenagem;
7. obrigatorio na apresentacao: destacar a baixa qualidade medida para agua e
   o tamanho reduzido do teste de animais.

## Licenciamento

Ultralytics usa AGPL-3.0 ou licenca Enterprise. O caminho academico pressupoe
repositorio e servico compativeis com AGPL; um produto comercial fechado exige
revisao juridica, licenca apropriada ou substituicao do detector. Licencas de
bibliotecas nao substituem as licencas individuais de imagens e datasets.

