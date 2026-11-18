# Evidências e protocolo v4

Esta pasta contém evidências pequenas, sem fotografias do usuário, capturas
da interface, imagens anotadas ou pesos. A interpretação está em
[VALIDACAO_V4.md](../../../docs/VALIDACAO_V4.md).

**Estado desta publicação:** perfil `pipeline-4c5c993223219929` congelado e
API 0.5.0 ativa, com 239 testes aprovados. O benchmark de 219 imagens está
concluído em coortes separadas e a verificação funcional de fotos/vídeos
foi concluída, mantendo os erros de detecção documentados.
O perfil e a configuração foram selados antes da inferência no novo holdout
e da reexecução das regressões conhecidas do usuário. As identidades e a
revisão do holdout já estavam seladas antes de fixar o perfil; a reserva dos
dados não ocorreu após a seleção dos parâmetros. Os
resultados de calibração não comprovam generalização ou paridade validação/teste.

| Arquivo | Conteúdo |
|---|---|
| `calibration_presence_v4_v2.json` | Referência congelada VAL60 para selecionar a fusão das classes 0 e 2. `null` não é negativo. Cópia exata da referência local. |
| `holdout_manifest.json` | 40 novas identidades nativas CLICK-test e 60 caixas nativas, com hashes e origem. Cópia exata selada antes das previsões. |
| `holdout_presence_review.json` | Revisão fonte-only: 40 positivos de lixo, 38 negativos de roedores e dois incertos; água fora do escopo. Cópia exata. |
| `data_audit_summary.json` | Quantidades, negativos escassos, tamanhos e cobertura dos pontos de pose. Não altera os labels de treino. |
| `diagnosis_summary.json` | Diagnóstico numérico dos dois erros do usuário, sem republicar as imagens. Casos de desenvolvimento, não teste independente. |
| `model_sources.json` | Componentes candidatos externos, revisões, URLs, licenças e hashes. YOLO customizado e referência de água v3 preservados. |
| `frozen_candidate.json` | Registro imutável do perfil, configuração, pesos, frases e regras selecionadas na validação antes das previsões no novo holdout. |
| `evaluation_holdout40.json` | Comparação após congelamento nas 40 novas identidades CLICK, com referência, contagens, latência e preservação de água. |
| `evaluation_regression179.json` | Reexecução congelada de test60 e extra119 já conhecidos; mantém os ganhos de presença e as regressões de localização. Não é um novo teste independente. |
| `product_validation.json` | Oito casos do catálogo concluídos na API, seis vídeos/duas fotos, 1.388 quadros decodificados, H.264/início rápido, Range 206, eventos e perfis conferidos. Verificação funcional, não acurácia independente. |
| `ui_verification.json` | Envios reais das duas fotos do usuário e vídeo Hanoi reproduzido até 12/12 s na tela; referências às capturas locais, sem republicá-las. |
| `active_profile.json` | Estado `verified_running_product`, perfis de imagem/vídeo ativos, hashes, provas e limitações da implantação local 0.5.0. |
| `provenance_index.json` | Índice inicial dos JSONs de protocolo e suas fontes locais; não é o manifesto final de release. |

A configuração congelada é `ferramentas/configs/inference_v4.json` (relativa à
raiz do projeto). Resíduos do especialista: score 0,30/margem 0,04, complementam
o YOLO. Animal: score 0,25/margem 0,08, substitui as caixas de animal do mestre.
Grounding DINO usa arestas 640/1024 e texto 0,20, com SigLIP 2 nos recortes.

Na calibração, animal passou de 15/20 para 20/20 positivos sinalizados, com
zero falsos alertas em 40 negativos nos dois casos. Isso não é uma taxa geral
de detecção. Nas 40 caixas CLICK de 20 fotos de validação, com IoU 0,50, os acertos passaram
de 31 para 34, as perdas de nove para seis e as previsões sem par de dez para
14: F2 do candidato 0,8173 (`5TP/(5TP+FP+4FN)`), com redução de precisão de caixas. A presença de
resíduos permaneceu igual (21 positivas sinalizadas, zero falsos alertas em
30 negativas; nove incertas excluídas). O objetivo de maior cobertura foi
escolhido na validação antes das previsões no teste novo. Os detalhes e limites estão no
protocolo; os 179 casos são reportados separadamente da calibração e do CLICK40.

No novo holdout40, resíduos presentes foram sinalizados em 38/40→39/40 fotos.
As caixas passaram de 43 TP/16 FP/17 FN para 48/20/12 (IoU 0,50): recall
0,7167→0,8000, precisão 0,7288→0,7059 e F1 0,7227→0,7500. Há maior cobertura,
mais previsões sem par e maior latência mediana por foto: 0,615→0,965 s na
RTX 4060. Animal teve zero falsos alertas em 38 negativas claras nos dois
perfis; dois incertos ficaram fora. Não há positivos de animal nem negativos
de resíduos, portanto essa coorte não mede seus respectivos recall e falsos
alertas. A água ficou igual nas 40 imagens, dentro de `1e-5`, sem nova alegação
de acurácia. Os resultados não são de um domínio novo nem um ganho universal.

Nos 179 casos de regressão conhecida, animal passou de 14/20→19/20 positivas
no test60 e 46/69→63/69 no extra119, sem alertas nas negativas proxy de rótulo.
O F1 de caixas IoU 0,50 caiu: 0,4706→0,3913 e 0,6207→0,6014,
respectivamente. Há mais presença sinalizada e pior acordo de localização,
incluindo caixas duplicadas/amplas. Os envelopes de pontos parciais limitam a
referência, mas não demonstram que todos os FP sejam objetos reais sem rótulo.
Resíduos no test60 tiveram 23/14/14→25/20/12 TP/FP/FN (F1 0,6216→0,6098);
no extra119 sem rótulo de resíduos, alertas passaram de 1/119 a 2/119.
Água permaneceu igual nas 179 imagens dentro de `1e-5`. As negativas entre
classes dessa regressão usam ausência de label como proxy, não revisão de
ausência abrangente. Não somar as 40 imagens reservadas e as 179 conhecidas
como uma taxa única de generalização. Uma simulação local de supressão de
caixas contidas não foi adotada, e o manifesto congelado segue intacto.

Na verificação funcional, o perfil-base é `pipeline-4c5c993223219929` e o de
vídeo `pipeline-380203274bf9c28c`, com a confirmação de água por 0,5 s preservada.
Todos os oito itens do catálogo concluíram e os seis vídeos decodificaram
integralmente. A UI mostrou as duas fotos do usuário e Hanoi até seu fim de
12 segundos. Isso não comprova qualidade temporal: Hanoi teve propostas de
animal em 144/240 quadros contra 187/240 anteriormente; Rua teve uma proposta
ambígua em 240 e continua sem acerto confirmado. A revisão visual encontrou
falsos resíduos em lixeira fixa e cotovelo de PVC, ambiguidades em vegetação,
perdas de outros resíduos/água rasa em `ralo_lixo` e água incluindo concreto
em Canal. Algumas fontes de vídeo foram usadas na adaptação de treino.
Essas contagens não são recall ou objetos físicos. A gravação narrada de
9min12s permanece histórica da 0.4; nenhuma nova publicação foi realizada.

Os acertos nas duas regressões do usuário (zero→uma marcação de animal e
uma→cinco de resíduos) foram reexecutados após o congelamento, mas as fotos
já eram conhecidas. Permanecem regressões de desenvolvimento, fora de
treinamento/calibração, e não são publicadas nesta pasta.

Os paths de dados são relativos à raiz do projeto. Os arquivos completos
precisam existir localmente para reproduzir a inferência. Não se deve clonar
esta pasta e assumir que ela contém fotografias/datasets ou modelos.

A classe água não é escolhida pela coorte v4. Preserva-se a revisão v3 de
presença 24 positivos/35 negativos/um incerto e seus parâmetros operacionais.
O campo da classe `1` na coorte v4 não substitui essa referência.

O holdout novo não entra em calibração/treino. É da mesma fonte/câmera CLICK,
não de um domínio externo novo; sua revisão foi feita por assistentes e não
é ground truth humana independente. Não permite auditar a composição do
pré-treinamento dos modelos de fundação.
