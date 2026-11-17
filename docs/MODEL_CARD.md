# Model Card — DrenaVision

## Produto ativo — versão 0.4.0

O produto combina um YOLO26n customizado com segmentação de superfícies de
água por SegFormer B1 ADE20K. As caixas são propostas reais dos modelos;
o azul preenche somente os pixels aceitos da máscara neural. Não existem
regras por nome de arquivo, coordenadas fixas ou redução artificial de caixas.

| Componente | Identificação |
|---|---|
| YOLO v3, `best.pt` | SHA-256 `12b9e86f223366c3f2ef8c2308f0e1140cda08c55ab55e281e58d300fd554b4b` |
| Origem do ajuste | v2 preservado em `best_v2.pt`, SHA-256 `01ea7c578783b2a08fc585051b0768a1e9c0d510cb5652173fab0979f8d7d27a` |
| Segmentador B1 | revisão NVIDIA `a775b4f5dcb2179e57006dca0e268281306412ef` |
| B1 safetensors | SHA-256 `e0802e077c195d9dbaf48c69b94946ef6a3ae5b6665176fedf363e5e63a1ec93` |
| Perfil de imagem | `pipeline-1314fcfa94f5f8cf`, [manifesto](../resultados/metrics/v3/active_profile.json) |

O treino YOLO concluiu **63 épocas**, selecionando a **43**, de uma meta de 80
com paciência 20, semente 42 e RTX 4060. Dados: **1.107/60/60** imagens.
A primeira montagem com negativos de água incorretos foi rejeitada após seis
épocas; esse checkpoint não foi usado como origem. O treino válido recomeçou
do v2 preservado. O B1 é pré-treinado; não alegamos ter treinado seus pesos.

O protocolo v3 agrupa componentes de caixas de água que já se tocam e retira
negativos inconsistentes da fonte. Esses envelopes podem incluir espaços secos;
não são máscaras certificadas. Originais, v1 e v2 permanecem preservados.
[VALIDACAO_V3.md](VALIDACAO_V3.md) registra diagnóstico e isolamento.

### Parâmetros selecionados antes do teste adicional

YOLO: entrada 640, confiança 0,35 para resíduos/água e 0,60 para animal,
IoU de supressão 0,45. A mudança de animal foi escolhida na validação: F1
0,7895 → 0,8000, com recall 0,75 → 0,70; reduz falsos alertas e pode perder
animais de menor confiança.

B1: processamento limitado a 512, probabilidade de pixel 0,95, componente
mínimo de 5% da imagem e complemento às propostas YOLO de água. Suprimir
envelopes contidos retém uma proposta existente, sem criar uma caixa de união.
Os modelos, processador, classes e parâmetros entram no hash do perfil.

A calibração usou **59 referências claras de presença de água**, revisadas
nas imagens de validação antes da inferência: 24 positivas e 35 negativas.
Uma imagem incerta foi excluída previamente. O primeiro grid falhou no alvo
de especificidade; a segunda rodada, nos mesmos mapas, obteve 23 TP, 0 FP,
1 FN e 35 TN. Isso é resultado de **validação**, não de teste.

### Testes separados e limites

O perfil de imagem foi fixado antes da avaliação das 119 imagens adicionais.
Elas incluem 69 fotos nativas de teste iRodent antes não utilizadas, 40 novas
fotos de água e dez negativos revisados. Foram verificadas identidade,
grupos e distância perceptual contra treino/validação.

| Presença nas imagens adicionais | v2 | Produto 0.4.0 |
|---|---:|---:|
| Água encontrada nas 40 positivas | 2/40 | 33/40 |
| Falso alerta de água nos dez negativos revisados | 0/10 | 2/10 |
| Animal encontrado nas 69 positivas | 47/69 | 46/69 |

As 69 imagens de animal **não são negativos de água**: suas referências não
anotam todas as categorias. Algumas também contêm mais animais que as caixas
da fonte, derivadas de pontos visíveis. TP/FP/FN por caixa medem concordância
com referências parciais; não certificam contagem de todos os animais nem
precisão de espécie. Revisão assistida não equivale a anotação humana independente.

O **YOLO isolado**, nas mesmas 60 imagens de teste com caixas v3, teve mAP@50
0,3349 → 0,4845 e mAP@50–95 0,1748 → 0,2574. O AP@50 de resíduos caiu
0,6233 → 0,5710; água aumentou 0,0688 → 0,3565 e animal 0,3125 → 0,5260.
Esses valores não representam mAP do produto composto. Os relatórios
[do produto](../resultados/metrics/v3/independent_comparison.json) e
[do detector](../resultados/metrics/v3/master_map_comparison.json) separam os escopos.

Nos exemplos conhecidos, o rato junto ao lixo recebe caixas ajustadas e a
foto do bueiro passa a destacar água. Permanecem perdas no rato entre
vegetação, na foto `ralo_lixo`, em pequenos trechos de água e em quadros
desfocados. Há resíduos marcados indevidamente sobre tubulação fixa e falsos
alertas de água em pisos molhados. Fontes de adaptação não são teste independente.

Os vídeos acrescentam confirmação temporal de água e apresentam o sinal por
classe ao operador. Contagens técnicas de IDs não equivalem a vários ratos ou
objetos físicos. A avaliação de 179 imagens não mede essa etapa temporal.

### Uso e licença

`agua_acumulada` mantém compatibilidade da API e significa água visível,
parada ou em movimento. Nenhum componente confirma estagnação, profundidade,
risco sanitário ou hidráulico. Alertas requerem verificação de uma pessoa.

O B1 usa a [licença NVIDIA de pesquisa/avaliação não comercial](https://github.com/NVlabs/SegFormer/blob/master/LICENSE).
O original oficial é verificado por hash e convertido com `weights_only=True`
para safetensors. A aplicação carrega somente os cinco arquivos locais
verificados, sem código remoto ou download durante a análise. O código AGPL
não relicencia esses pesos. Este é um protótipo acadêmico; uso comercial exige
resolver essas restrições e as dos dados iRodent.

## Histórico — versão 0.3.0, protocolo v2

O checkpoint v2 foi preservado em `resultados/local/models/best_v2.pt`, SHA-256
`01ea7c578783b2a08fc585051b0768a1e9c0d510cb5652173fab0979f8d7d27a`.
É um único YOLO26n, sem ensemble. O treino novo partiu do peso pré-treinado,
concluiu **60 épocas**, parou por early stopping e selecionou a época **35**.
O v1 foi preservado em `resultados/local/models/best_v1.pt` e nos artefatos
históricos. Distribuição: [release v0.3.0-model-v2](https://github.com/Torugo0/drenahub-drenavision/releases/tag/v0.3.0-model-v2).

O dataset v2 tem 557/62/62 imagens. Seu fingerprint é
`af3927e1f3b24de81984cf4a302eb764dd09eb4e0380b7105ffd013ed0496e49`.
Os dois modelos foram avaliados nas mesmas 62 imagens de **teste**, com as
caixas v2 derivadas de keypoints visíveis. A seleção corrige um defeito de
anotação da fonte; não torna a caixa uma anotação humana de corpo inteiro.

| Escopo no teste corrigido | Modelo | Precisão | Recall | mAP@50 | mAP@50–95 |
|---|---|---:|---:|---:|---:|
| Agregado | v1 reavaliado | 0,2906 | 0,2842 | 0,2563 | 0,1490 |
| Agregado | v2 histórico | 0,3828 | 0,4251 | 0,3581 | 0,1810 |
| Animal | v1 reavaliado | 0,0322 | 0,0500 | 0,0183 | 0,0059 |
| Animal | v2 histórico | 0,3769 | 0,4500 | 0,3102 | 0,1454 |
| Resíduos | v2 histórico | 0,5347 | 0,6486 | 0,6233 | 0,3611 |
| Água | v2 histórico | 0,2368 | 0,1765 | 0,1409 | 0,0364 |

Fontes: [comparação](../resultados/metrics/v2/comparison.json),
[avaliação do v2](../resultados/metrics/v2/candidate_test_metrics.json),
[v1 nas caixas corrigidas](../resultados/metrics/v2/baseline_test_metrics.json)
e [manifesto do treino](../resultados/metrics/v2/training_manifest.json).
Os valores de **validação** usados para selecionar a época não são métricas
de teste: o AP de animal no teste separado é 0,3102.
O tempo de inferência medido para o v2 foi 3,91 ms/imagem na RTX 4060, sem
upload, persistência ou renderização.

A melhora é parcial. A comparação ganha em localização de animal e mAP
agregado, mas água segue fraca: seu mAP@50 caiu de 0,1645 para 0,1409.
Resíduos melhoram em mAP@50, mas pioram em mAP@50–95 (0,4048 → 0,3611).
Nos exemplos, o close usado na adaptação tem uma caixa de animal nos 220
quadros; isso é consistência numa fonte de treino, não estimativa independente
de acurácia. Telha, carro e espuma deixaram de gerar os falsos alertas anteriores,
mas o rato na cena com lixo continua perdido, `rato_rua` segue sem marcação e
a foto de lixo em Queens regrediu. A outra foto de bueiro mantém três embalagens
marcadas. Consulte a [revisão visual v2](../resultados/metrics/v2/product_validation_v2.json),
que mantém esses erros e a origem de cada exemplo.

## Modelo v1: estado histórico em 16/11/2026

Uma auditoria posterior encontrou um defeito nas caixas de roedores da fonte
iRodent: pontos ausentes `(0,0,0)` foram incorporados ao retângulo publicado.
Os números v1 abaixo são preservados como resultado histórico desse protocolo;
as métricas de animal não comprovam localização correta do corpo do roedor.
O experimento v2, descrito adiante, usa anotações corrigidas separadas e exige
comparar os dois modelos no mesmo conjunto de rótulos v2.

O checkpoint customizado v1 foi preservado em `resultados/local/models/best_v1.pt`. O treino
foi retomado de um run interrompido na epoca 28 e terminou por early stopping
apos 70 de 100 epocas. O snapshot interrompido foi preservado separadamente e
nao e apresentado como run concluido.

- dispositivo: NVIDIA GeForce RTX 4060 (`cuda:0`);
- seed: 42;
- SHA-256 do YOLO26n inicial:
  `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`;
- SHA-256 do `best.pt` publicado:
  `fe439a3a722e73db5d841cc199026c8d42f4bbf806bf95924c2eb8c09a9f6bf7`;
- manifesto versionado: `resultados/metrics/training_manifest.json`.

Os hashes diferentes e o avanco de epocas comprovam que houve ajuste de pesos.
Essa execução comprova ajuste de pesos, mas seus resultados de localização
animal dependem da anotação defeituosa e ficam somente como histórico.

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
- checkpoint histórico: `resultados/local/models/best_v1.pt` (publicado originalmente como `best.pt`);
- configuracao versionada: `ferramentas/configs/train.yaml`.

Cada treino concluido grava hash SHA-256 dos pesos iniciais, `best.pt` e copia
publicada, versoes do ambiente, fingerprint do dataset, argumentos efetivos e
metricas finais em `training_manifest.json`. Hashes iguais entre peso inicial e
`best.pt` invalidam a alegacao de fine-tuning.

Uma execucao interrompida pode ser retomada com `ferramentas/scripts/train.py --resume
<run>/weights/last.pt`. Antes de continuar, o pipeline copia um snapshot marcado
como `interrupted`, valida a epoca/meta e so publica o modelo quando a retomada
avanca e altera os pesos.

## Dados e separacao

O dataset padrao e `dados/processed/drenavision/dataset.yaml`. A execucao
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

O comando histórico `python ferramentas/scripts/evaluate.py --config ferramentas/configs/evaluate.yaml --model resultados/local/models/best_v1.pt` usa apenas `test` v1 e
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

### Resultado histórico v1 no split de teste

Fonte unica desta tabela:
`resultados/metrics/evaluation_metrics.json`. A execucao usou
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

## Verificação funcional com exemplos externos

A biblioteca em `dados/examples/` reúne **seis filmagens reais e duas
fotografias**, com créditos, licenças e hashes em `sources.json`. Não são
sequências simuladas de fotografias nem entradas previamente marcadas. Esses
exemplos ficaram fora do treino, validação e teste **na execução v1** e não têm caixas de referência
revisadas para calcular novas métricas de precisão, recall ou mAP.

O [relatório de validação do produto](../resultados/demo/product_validation.json)
registrou o mesmo checkpoint e limiar 0,35, sem alteração para favorecer os
exemplos. As oito análises foram concluídas. Os seis vídeos exportados usam
H.264 com faststart; os 1.388 quadros processados foram decodificados
independentemente por FFmpeg, sem perda. Cada saída também possui um quadro
anotado de destaque. Isso verifica inferência/exportação e compatibilidade do
formato; não certifica reprodução em todos os navegadores.

Na revisão visual, `rato_close` e `rato_rua` continham roedores visíveis, mas
não tiveram detecções. Em `rato_residuos`, o destaque animal de maior confiança
enquadrou uma telha, e não o rato; esse resultado não é apresentado como acerto
confirmado de animal. Em `drenagem_chuva`, espuma/água recebeu alerta de animal;
em `canal_residuos`, um automóvel foi marcado como obstrução. A marcação de
`sarjeta_residuos` permaneceu ambígua.

As fotografias `ralo_lixo` e `bueiro_agua` tiveram marcações visualmente válidas
de resíduos, porém outros resíduos e a água da cena ficaram sem marcação. Os
contadores do vídeo representam marcações/IDs de rastreamento, não uma contagem
garantida de objetos distintos. A validação expõe falhas de generalização e
mantém necessária a revisão humana, mesmo quando a cadeia de processamento
funciona corretamente.

## Auditoria e experimento de localização v2

A [auditoria versionada](../resultados/metrics/v2/annotation_audit.json) examinou
as 447 anotações originais do iRodent (357 imagens nativas de treino e 90 de
teste). Em **305/447**, a caixa publicada coincide com o envelope de todos os
keypoints, inclusive os ausentes representados por `(0,0,0)`. Não foi encontrada
troca COCO `xywh`/`xyxy`, divergência de dimensão ou rotação EXIF nessa fonte.
Das 100 imagens de roedor usadas no treino v1, 75 tinham caixa na origem;
isso ajuda a explicar o aprendizado de fundo e os retângulos excessivos.

No v2, `irodent_annotation_box()` deriva um retângulo apenas dos keypoints
anotados com visibilidade maior que zero. Acrescenta margem de **10% por lado**
(mínimo de 2 pixels em cada eixo) e limita as coordenadas à imagem. Casos com
menos de dois pontos visíveis ou geometria de área zero são excluídos: cinco
anotações da fonte ficaram sem caixa utilizável. A
[montagem de 12 comparações](../resultados/metrics/v2/irodent_source_vs_v2.jpg)
mostra a caixa publicada em vermelho e a derivada em azul.

Este alvo é um **envelope de pontos visíveis anotados**. Pode cobrir somente a
cabeça ou outra parte visível quando há poucos pontos. Não garante um retângulo
de corpo inteiro e não é ground truth independente revisado por uma pessoa.
A correção usa a anotação de pose da fonte, sem predição automática como rótulo.

O dataset separado `dados/processed/drenavision_v2/` tem **557/62/62 imagens**
de treino/validação/teste. Contém 373 imagens iRodent no total, incluindo 234
imagens adicionais somente do treino nativo. Uma imagem do treino v1 foi
excluída por pontos insuficientes; os IDs de imagens dos 62 itens de validação
e 62 de teste foram preservados, com caixas de roedor corrigidas. Dados e
métricas v1 não foram sobrescritos. Estrutura, hashes, grupos e pHash entre
splits passaram na validação.

Foram acrescentadas **16 imagens/cortes de adaptação**, apenas ao treino,
derivadas de `rato_close`, `rato_residuos`, `canal_residuos` e `drenagem_chuva`.
Esses rótulos vieram de inspeção visual assistida; não são anotação humana
independente. Portanto esses quatro clipes deixam de ser teste externo para
o candidato v2. `rato_rua` e `sarjeta_residuos` ficaram fora da adaptação e
servem como regressão; já foram inspecionados no desenvolvimento e também
não representam um teste de campo intocado.

O treino `ferramentas/configs/train_v2.yaml` gerou `candidate_v2.pt` separado.
Após a avaliação comparativa e a revisão dos clipes, os mesmos bytes foram
promovidos a `best.pt`, com backup do v1. A configuração `evaluate_v2.yaml`
usa `best_v2.pt`; `evaluate_v2_baseline.yaml` usa `best_v1.pt` e o mesmo
dataset corrigido. Os mAP v1 e v2 com rótulos distintos não são diretamente
comparáveis; a tabela inicial reavalia ambos nos rótulos v2. A promoção e seus
resultados ficam separados da auditoria inicial, cujo estado era anterior ao treino.

## Limitacoes e riscos conhecidos

- agua rasa, reflexos, piso brilhante e sombras podem se confundir;
- `agua_acumulada` tem recall 0,1765 e mAP@50-95 0,0364 no teste v2; continua
  limitada e precisa de mais diversidade, revisão de caixas e dados reais de drenagem;
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
5. concluido para a demonstracao automatica: biblioteca de seis filmagens reais
   e duas fotografias, com exportacao H.264, origem e falhas documentadas;
6. pendente para a entrega: gravar a apresentacao narrada e, idealmente,
   acrescentar filmagem propria do dominio de drenagem;
7. obrigatorio na apresentacao: destacar a baixa qualidade medida para agua e
   o tamanho reduzido do teste de animais.

## Licenciamento

Ultralytics usa AGPL-3.0 ou licenca Enterprise. O caminho academico pressupoe
repositorio e servico compativeis com AGPL; um produto comercial fechado exige
revisao juridica, licenca apropriada ou substituicao do detector. Licencas de
bibliotecas nao substituem as licencas individuais de imagens e datasets.

