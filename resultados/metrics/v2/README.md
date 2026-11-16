# Auditoria das anotações e protocolo v2

`annotation_audit.json` registra a causa observada das caixas grandes de animal:
305 de 447 caixas publicadas pelo iRodent incluem a origem `(0,0)`, coincidindo
com o envelope de todos os keypoints, inclusive pontos ausentes com visibilidade
zero. Isso ensina o detector a incluir fundo que não pertence ao animal.

`irodent_source_vs_v2.jpg` compara 12 imagens: vermelho é a caixa publicada;
azul é o envelope dos keypoints anotados com visibilidade maior que zero,
acrescido de 10% por lado e recortado nos limites da imagem. Esse envelope
pode abranger apenas cabeça ou parte visível; não é uma revisão humana de
corpo inteiro. Cinco anotações com menos de dois pontos visíveis foram excluídas.

Os arquivos completos de trabalho ficam em `resultados/local/adaptation/`.
O dataset v1 e suas métricas permanecem preservados. A avaliação v2 precisa
comparar os modelos antigo e candidato usando a mesma anotação corrigida;
os números calculados com os rótulos v1 não são diretamente comparáveis.
A auditoria foi registrada antes do treino; a decisão posterior de promoção
está em `comparison.json`. O modelo v2 ativo tem SHA-256
`01ea7c578783b2a08fc585051b0768a1e9c0d510cb5652173fab0979f8d7d27a`.

| Evidência | Arquivo |
|---|---|
| Treino de 60 épocas, melhor modelo na época 35 | `training_manifest.json`, `training_results.csv` |
| Mesmas 62 imagens e mesmos rótulos para os dois modelos | `baseline_test_metrics.json`, `candidate_test_metrics.json` |
| Ganhos, regressões e decisão de promoção | `comparison.json` |
| Fontes, hashes e divisões dos 681 itens | `data_manifest.csv`, `preparation_report.json` |
| Revisão dos oito exemplos, incluindo falhas | `product_validation_v2.json` |
| Interface compacta verificada no navegador | `frontend_v2_desktop.jpg`, `frontend_v2_mobile.jpg` |

As métricas de teste continuam modestas: mAP@50 agregado 0,3581 e de animais
0,3102. O vídeo da grade foi fonte de adaptação; sua detecção não representa
teste independente. Os roedores pequeno/encoberto, água e alguns resíduos
continuam sendo limitações. Há regressão documentada no saco plástico da foto
de Queens. O modelo anterior foi preservado em `resultados/local/models/best_v1.pt`.

## Origem e licença da montagem

Imagens do **iRodent v1.0**, Ye et al./Adaptive Motor Control Lab,
[registro e download](https://doi.org/10.5281/zenodo.8250392), com origem em
observações iNaturalist. O projeto adota conservadoramente **CC BY-NC 4.0**
para as imagens-fonte, conforme os metadados internos. O registro tem indicação
divergente de Apache-2.0; essa indicação não substitui os direitos das imagens.
Consulte [os avisos de terceiros](../../../THIRD_PARTY_NOTICES.md).

A montagem foi redimensionada e recebeu caixas coloridas e legendas pelo
DrenaVision. Os identificadores originais de suas 12 imagens estão no JSON
ao lado; a auditoria local preserva os 447 registros e os hashes de `train.json`
e `test.json`. A montagem é evidência técnica de uso acadêmico e não concede
direitos comerciais adicionais. Créditos individuais da origem iNaturalist
devem ser conferidos antes de reutilização externa das fotografias.

As capturas `frontend_v2_desktop.jpg` e `frontend_v2_mobile.jpg` incluem a
filmagem **Rat.ogv**, de Ykmyks, CC BY-SA 4.0, recortada e convertida para
H.264. Fonte, versão da licença e hashes estão em `dados/examples/sources.json`.
O DrenaVision acrescentou a interface e as caixas de detecção.
