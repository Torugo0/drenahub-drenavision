# Revisão v4: detecção de roedores e resíduos

Esta página registra o diagnóstico, as referências congeladas e o método de
avaliação da revisão v4. **A API 0.5.0 está ativa, o benchmark de 219 imagens
está concluído em coortes separadas e a verificação funcional de fotos/vídeos
foi concluída.** Erros de detecção permanecem documentados.
O perfil `pipeline-4c5c993223219929` foi fixado antes de
executar a inferência no novo holdout e de reexecutar as regressões do usuário nessa
configuração. Os resultados de calibração abaixo não demonstram melhoria
conclusiva fora da validação ou paridade entre validação e teste. O perfil
congelado foi ativado sem mudar parâmetros após os resultados reservados.

## Falhas reproduzidas

As duas imagens enviadas pelo usuário foram encontradas localmente em
`dados/examples/teste.jpg` e `teste1.jpg`. Foram usadas para reproduzir o
problema, não para treinar, calibrar limiares ou compor um teste independente.
Os diagnósticos numéricos estão em
[diagnosis_summary.json](../resultados/metrics/v4/diagnosis_summary.json).
Essas fotografias, capturas e versões anotadas não são republicadas na pasta
de métricas v4; apenas dimensões, hashes e diagnósticos numéricos são incluídos.

| Caso | Comportamento medido no YOLO v3 |
|---|---|
| Rato no tubo, 360 × 240 | Em 640, uma proposta com confiança 0,564954 cobre cabeça/corpo visíveis, mas é eliminada pelo corte operacional de animal 0,60. Em 960 o maior score sobe a 0,710648, enquanto a caixa cobre só a parte inferior do corpo. Aumentar a resolução não comprova melhor localização. |
| Resíduos na grelha, 960 × 640 | Em 640 e confiança mínima 0,35 resta uma proposta; em 0,15 surgem quatro. Em 960/0,35 são duas. As garrafas e a pilha continuam incompletamente cobertas. Uma proposta falsa de água com score 0,277 também aparece em 960, mostrando o risco de reduzir o limiar global. |

As propostas vieram da inferência neural real. Nenhuma caixa de correção deve
ser desenhada por nome de arquivo, hash da foto, coordenada conhecida ou regra
específica para esses dois casos. Uma contagem de propostas também não equivale
à quantidade de resíduos físicos presentes na cena.

## Auditoria dos dados existentes

O checkpoint customizado v3 foi treinado com 1.107 imagens: 250 CLICK de
resíduos, 500 de água, 333 iRodent e 24 quadros/crops de adaptação. Há 498 caixas
de resíduos, 1.453 regiões de água e 344 caixas de animais no treino.
Os dados e pesos anteriores são preservados. A revisão v4 descrita aqui não
alega a execução de um novo treinamento.

Foram identificados limites que precisam acompanhar a interpretação dos
resultados:

- **Poucos negativos confiáveis:** existem seis labels vazios no treino,
  mas somente cinco são crops de fundo previamente revisados, cerca de 0,45%
  das imagens. O sexto, CLICK `trash (1045)`, contém pequenos fragmentos
  azul/vermelho suspeitos; seu label vazio não comprova uma cena limpa.
- **Caixas parciais de animais:** o conversor deriva caixas dos pontos de
  pose visíveis com margem de 10%, e não de contornos completos revisados.
  Em 53/333 imagens iRodent de treino há no máximo um ponto corporal visível;
  26 têm no máximo quatro pontos totais. A revisão visual confirma exemplos
  em que a caixa cobre a cabeça e deixa parte do tronco visível de fora.
  Isso não significa que todas essas 53 caixas sejam incorretas: alguns
  animais estão realmente ocultos. Os COCO locais usados não têm máscaras
  de segmentação para substituir automaticamente essa referência.
- **Protocolos diferentes para lixo:** CLICK anota itens/embalagens; quadros
  de adaptação de Hanói incluem um envelope de pilha. A unidade de instância
  não é uniforme entre fontes.
- **Alvos de outras classes omitidos:** fontes de uma classe podem conter
  água, resíduos ou animais adicionais não anotados. Ausência de label não
  pode ser usada automaticamente como ausência de alvo.
- **Pouca diversidade efetiva:** as 20 imagens de água da validação pertencem
  a apenas quatro grupos numéricos de cenas. Vinte imagens não representam
  vinte locais independentes.

As contagens, distribuições de tamanho das caixas e hashes ficam em
[data_audit_summary.json](../resultados/metrics/v4/data_audit_summary.json).
As folhas de revisão permanecem locais, em `resultados/local/v4/data_audit/`.

## Candidato composto congelado

A proposta mantém o **YOLO customizado v3** e avalia dois componentes externos
pré-treinados e fixos:

1. **Grounding DINO Tiny:** propõe caixas de objetos a partir de um vocabulário
   fixado para roedores e resíduos. Vocabulário, resolução e limiares foram
   escolhidos no desenvolvimento com a validação, antes das previsões no novo holdout.
2. **SigLIP 2:** verifica recortes das caixas candidatas contra descrições
   positivas e de fundo, para examinar se a inclusão do detector aberto
   aumenta os falsos alertas. Seus scores não são probabilidades calibradas
   nem têm a mesma escala dos scores YOLO/Grounding DINO.
3. **Fusão por classe:** no modo `animal_replace`, resíduos do especialista
   complementam os do YOLO; as caixas de animal vêm do especialista verificado
   e substituem as do YOLO, evitando manter uma segunda caixa parcial do
   mestre. As caixas do especialista têm prioridade na supressão de duplicatas
   da mesma classe, com IoU 0,5. Água não participa dessa supressão de objetos.
   Só as classes 0 (resíduos) e 2 (animal/roedor) participaram da escolha.

O registro imutável está em
[frozen_candidate.json](../resultados/metrics/v4/frozen_candidate.json),
SHA-256 `94d20cce07114fe4b6c9e58b4a86daaaebea191989dba9a243c01fcbc5e4b93c`.
A configuração executável está em
[inference_v4.json](../ferramentas/configs/inference_v4.json), SHA-256
`35b43502b6957e981895c23a3ff2e2f80e0fdd2c736990f5d61ac96f06e86cbb`.

Os registros de 18 de novembro de 2026, em UTC, distinguem a reserva dos dados
da fixação dos parâmetros: as identidades do holdout foram seladas às 00:03 e
sua revisão às 00:05; a execução de calibração em validação foi registrada
às 00:13/00:14; o perfil foi congelado às 00:17 e o relatório da avaliação
reservada foi registrado às 00:25. Portanto, os dados reservados já estavam
selados antes do congelamento do perfil. A garantia experimental é que o
perfil foi fixado antes da inferência no holdout e não foi retunado depois
de observar essas previsões, não que o conjunto foi reservado após a seleção.

| Parâmetro congelado | Valor |
|---|---|
| YOLO mestre | Tamanho 640; corte base 0,35; resíduos 0,35; animal 0,60; IoU 0,45 |
| Grounding DINO | Aresta curta 640, longa máxima 1024; texto 0,20 |
| Resíduos do especialista | Score mínimo 0,30; margem SigLIP 2 mínima 0,04; complementa o mestre |
| Animal do especialista | Score mínimo 0,25; margem SigLIP 2 mínima 0,08; substitui o animal do mestre |
| Margem de verificação | Maior cosseno positivo menos maior cosseno negativo; não é probabilidade de risco |

O contexto de 10% de cada lado do recorte é usado somente na entrada do
verificador; não aumenta a caixa devolvida pelo detector. O manifesto fixa
também as frases positivas/de fundo e os hashes dos pesos. Não se trata de um
novo treinamento do YOLO ou dos componentes externos.

Esses componentes não foram treinados pela equipe nesta revisão. A origem,
as revisões fixadas, os hashes, as URLs e as licenças estão em
[model_sources.json](../resultados/metrics/v4/model_sources.json).

| Componente | Revisão ou hash fixado |
|---|---|
| YOLO customizado v3, `best.pt` | SHA-256 `12b9e86f223366c3f2ef8c2308f0e1140cda08c55ab55e281e58d300fd554b4b` |
| [Grounding DINO Tiny oficial](https://huggingface.co/IDEA-Research/grounding-dino-tiny/blob/a2bb814dd30d776dcf7e30523b00659f4f141c71/README.md) | `a2bb814dd30d776dcf7e30523b00659f4f141c71`, Apache-2.0 |
| [SigLIP 2 oficial](https://huggingface.co/google/siglip2-base-patch16-224/blob/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2/README.md) | `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, Apache-2.0 |

Os artefatos externos usam pesos `safetensors`, configuração e tokenizadores
locais fixados. Baixar os pesos não significa ativá-los nem demonstrar ganho.
O vocabulário desta avaliação concentra-se em roedores; não permite prometer
que todos os tipos de animais serão reconhecidos.

O conteúdo integral do pré-treinamento dos componentes externos não é
auditável pelo projeto. Um hash não encontrado no treino customizado não
prova que a imagem pública jamais tenha aparecido no pré-treinamento externo.

## Validação para escolha da fusão

A referência vigente é
[calibration_presence_v4_v2.json](../resultados/metrics/v4/calibration_presence_v4_v2.json),
SHA-256 `53c2d92ddfdac9b5a13b5c2a3c103257cd4fd608aba071d47c61644c6d3c7b9f`.
Ela contém as mesmas 60 identidades de validação v3: 20 por fonte.

| Presença revisada | Positivos | Negativos | Incertos |
|---|---:|---:|---:|
| Resíduos, classe 0 | 21 | 30 | 9 |
| Animal/roedor, classe 2 | 20 | 40 | 0 |

`true` significa alvo visível confirmado; `false`, ausência de alvo visível
na resolução disponível; `null`, incerteza. **Os valores `null` devem ser
excluídos, nunca convertidos em negativos.** Nove cenas de água têm fragmentos
ou pilhas de natureza incerta para resíduos. A revisão também reconhece
plástico descartado em uma cena de roedor que não tinha label da classe 0.

A revisão de presença foi feita por assistentes sobre as 60 cenas completas,
com inspeção dos originais ambíguos. A versão v2 reconcilia uma segunda revisão
conservadora antes da observação das previsões candidatas. Não é ground truth
humana independente. As referências v1 locais permanecem arquivadas.

Essa coorte serve ao desenvolvimento, à escolha de limiares e ao controle de
falsos alertas. Não é um teste independente nem uma nova taxa de generalização.
A revisão adicional é de presença por imagem. Separadamente, as caixas nativas
CLICK da validação serviram à escolha da cobertura de resíduos. Essa referência
de caixas não certifica máscara, corpo completo, contagem de animais ou todos
os itens físicos de lixo.

## Resultados de calibração e custo da maior cobertura

Para animal, a seleção maximizou F1 de presença na validação, com desempate por
menos falsos alertas e maior margem de verificação. Para resíduos, o objetivo
foi F2 das caixas CLICK, com casamento IoU 0,50, priorizando cobertura e sujeito a precisão de caixas
de pelo menos 0,70 e no máximo dois alertas adicionais nas 30 imagens negativas
revisadas. O objetivo de cobertura foi adotado usando a validação, antes de
consultar as previsões no novo holdout. Os números abaixo pertencem à **calibração**.

A fórmula usada é `F2 = 5×TP / (5×TP + FP + 4×FN)`; não é AP e não pode ser
comparada diretamente com o mAP histórico do YOLO isolado.

| Referência de validação | V3 | Candidato congelado |
|---|---|---|
| Animal: 20 imagens positivas | 15 sinalizadas; cinco perdidas | 20 sinalizadas; nenhuma perdida nessa coorte |
| Animal: 40 imagens negativas | Zero falsos alertas | Zero falsos alertas |
| Resíduos: presença em 21 positivas | 21 sinalizadas | 21 sinalizadas |
| Resíduos: 30 negativas e nove incertas | Zero falsos alertas nas negativas; incertas excluídas | Mesmo resultado de presença; incertas excluídas |
| Resíduos CLICK: 20 imagens, 40 caixas de referência | 31 acertos, dez previsões sem par, nove perdas | 34 acertos, 14 previsões sem par, seis perdas |

Nas caixas CLICK, o candidato alcança precisão 34/48 = 0,7083, recall
34/40 = 0,85 e F2 = 0,8173. O V3 tem precisão 31/41 = 0,7561 e recall
31/40 = 0,775. Há **mais cobertura e também mais previsões sem correspondência**;
a precisão de caixas diminui nesse comparativo. Um resultado de presença
igual não revela essa diferença de localização. A referência é nativa e pode
ser incompleta; previsões sem par não devem ser automaticamente descritas
como lixo inexistente na cena.

Reconhecer os 20 positivos dessa validação pequena não significa detectar
todos os roedores, nem permite anunciar uma taxa geral de acerto. Resultados
de teste, latência, revisão de caixas e paridade do runtime continuam necessários.

## Reexecução das duas regressões conhecidas

Após congelar o perfil, a execução pelo runtime passou de zero para uma
marcação de animal na foto do tubo, incluindo cabeça e corpo aparentes, e de
uma para cinco marcações de resíduos na foto da grelha, incluindo pilha e
itens. São propostas neurais reais, não caixas inseridas na interface.
O registro numérico e as comparações visuais ficam localmente em
`resultados/local/v4/user_regression/`.

As fotos já tinham sido vistas no diagnóstico original. Sua reexecução após
o congelamento não as torna independentes. Esses acertos de regressão não
provam uma contagem correta de cinco objetos ou cobertura perfeita de toda a
pilha; também não substituem o teste das outras cenas. As fotografias e versões
anotadas não são republicadas junto às métricas.

## Água preservada

O componente e a política de água v3 não são retunados nesta revisão. Devem
ser preservados o SegFormer B1, os parâmetros já publicados e a confirmação
de 0,5 segundo nos vídeos, com as versões de imagem/vídeo separadas.

Para presença de água continua valendo
[val_water_presence_review.json](../resultados/metrics/v3/val_water_presence_review.json):
24 positivos, 35 negativos e um incerto. Os campos da classe `1` existentes
na coorte v4 não substituem essa revisão legada e **não são usados para
seleção da fusão v4**. A foto CLICK borrada `ed3b7841b4666c3d` permanece incerta
para água. A preservação do comportamento de água precisa ser conferida na
comparação final; resultados das classes 0 e 2 não provam essa paridade.

## Testes e regressões separados

| Conjunto | Uso permitido |
|---|---|
| VAL60 revisada | Escolher vocabulário/limiares/verificação/fusão das classes 0 e 2. |
| 60 imagens de teste v3 + 119 adicionais já examinadas | Comparação controlada/regressão conhecida. Não voltar a utilizá-las como um teste novo intocado. |
| 40 novas imagens CLICK-test | Avaliar somente depois de congelar o candidato. Nunca treinar, escolher prompts ou ajustar limiares por esses resultados. |
| `teste.jpg` e `teste1.jpg` do usuário | Reproduzir e conferir os erros relatados; permanecem fora de treino/calibração e não compõem uma métrica independente. |
| Vídeos e fotos da interface | Verificar funcionamento, reprodução e apresentação; fontes de adaptação não são testes independentes. |

O novo holdout tem 40 imagens e 60 caixas nativas, selecionadas por semente
fixa entre 92 identidades de teste CLICK que não estavam na amostra anterior.
A seleção foi selada antes das previsões candidatas e excluiu coincidências
por identidade, hash e pHash com distância até quatro bits contra todos os
splits atuais, o teste adicional, os exemplos do usuário e itens já aceitos.

- [holdout_manifest.json](../resultados/metrics/v4/holdout_manifest.json):
  SHA-256 `03b7474f03cb4d6545d9d66db7129f693cdb215ef26bcdce004805896340269e`.
- [holdout_presence_review.json](../resultados/metrics/v4/holdout_presence_review.json):
  SHA-256 `6a3d858f744a4e47a6a522e527a84be88eaf694baf8724d37625f5a3dc03301a`.

A revisão das 40 cenas confirma 40 positivos de resíduos e 38 negativos
visuais de roedores. Duas cenas muito borradas ficam incertas para roedores.
Água não é avaliada nessa coorte. Portanto, o holdout não tem negativos de
resíduos e não permite medir sozinho seus falsos alertas, nem recall de
roedores. Esses denominadores precisam acompanhar qualquer resultado.

As imagens vêm da mesma fonte/câmera conhecida CLICK. São novas identidades
para a revisão do projeto, **não um domínio/câmera novo**. A revisão visual é
assistida e não certificada por anotadores humanos independentes.

## Congelamento, paridade e publicação de resultados

### Resultado do novo teste reservado CLICK40

O [relatório de execução](../resultados/metrics/v4/evaluation_holdout40.json)
registra o hash congelado antes da inferência e informa que esses resultados
não foram usados para selecionar os parâmetros. Baseline e candidato usam as
mesmas 40 imagens, referências e casamento de caixas com IoU 0,50.

| Medida no holdout40 | Produto 0.4 | Candidato 0.5 congelado |
|---|---:|---:|
| Fotos positivas de resíduos sinalizadas | 38/40 | 39/40 |
| Caixas de resíduos com correspondência | 43 | 48 |
| Previsões de resíduos sem correspondência | 16 | 20 |
| Caixas de resíduos perdidas | 17 | 12 |
| Precisão por caixa de resíduos | 0,7288 | 0,7059 |
| Recall por caixa de resíduos | 0,7167 | 0,8000 |
| F1 por caixa de resíduos | 0,7227 | 0,7500 |
| Falsos alertas de animal nas negativas claras | 0/38 | 0/38 |
| Latência mediana por foto, RTX 4060 | 0,615 s | 0,965 s |

O novo teste mostra recuperação de mais resíduos, com cinco referências
perdidas a menos e quatro previsões sem par a mais. A precisão de caixas
diminui. Como a fonte pode omitir objetos, uma previsão sem par não comprova
automaticamente um objeto inexistente. A coorte não permite estimar falsos
alertas de resíduos em cenas limpas ou recall de animais, pois não contém
negativos da primeira classe nem positivos da segunda. Dois casos incertos
para animal foram excluídos desse denominador.

As propostas de água mantiveram-se iguais nas 40 imagens, dentro da tolerância
numérica `1e-5`. É uma verificação de preservação de saída, não uma nova
estimativa de acurácia da água; suas presenças não foram revisadas nesse teste.
O custo de latência também aumentou: a medição por foto nesse hardware não é
uma promessa de processamento em tempo real ou de desempenho em CPU/vídeo.

### Regressão conhecida em 179 imagens

O [relatório dos 179 casos](../resultados/metrics/v4/evaluation_regression179.json)
compara o mesmo baseline e candidato congelado sobre o test60 v3 e o extra119.
Esses dados já foram examinados anteriormente: são regressão de engenharia,
**não um novo holdout nem uma base para retunar o perfil**. A ausência de
rótulo por classe é proxy negativo; pode haver objetos de outra classe não
anotados. Isso difere das negativas revisadas de VAL60 e CLICK40.

| Medida | Produto 0.4 | Perfil 0.5 congelado |
|---|---:|---:|
| Animal, presença test60 | 14/20 positivas; 0/40 alertas negativos proxy | 19/20; 0/40 |
| Animal, caixas test60, IoU 0,50 | 8 TP / 6 FP / 12 FN; F1 0,4706 | 9 / 17 / 11; F1 0,3913 |
| Animal, presença extra119 | 46/69 positivas; 0/50 alertas negativos proxy | 63/69; 0/50 |
| Animal, caixas extra119, IoU 0,50 | 36 TP / 11 FP / 33 FN; F1 0,6207 | 43 / 31 / 26; F1 0,6014 |
| Resíduos, presença test60 | 20/20 positivas; 0/40 alertas negativos proxy | 20/20; 0/40 |
| Resíduos, caixas test60, IoU 0,50 | 23 TP / 14 FP / 14 FN; F1 0,6216 | 25 / 20 / 12; F1 0,6098 |
| Resíduos, alertas em extra119 sem rótulo da classe | 1/119 | 2/119 |

A presença de animal melhora, mas **o F1 de localização cai nas duas coortes**.
Persistem caixas duplicadas, amplas e possíveis objetos incorretos. Alguns
envelopes de pontos da fonte representam cabeça/tronco parcial, o que também
limita o casamento com corpo completo. Essa limitação não absolve todos os
FP: uma localização neural grande pode incluir fundo demais, e a referência
incompleta não comprova que a previsão seja correta.

Uma auditoria offline posterior simulou supressão de caixas de animal contidas
(90% da menor área, razão de áreas até quatro), sem alterar coordenadas.
Por confiança, removeu oito caixas na VAL60 e melhorou seu F1 de 0,5714 para
0,6829, mas no agregado diagnóstico dos 179 perdeu cinco casamentos e reduziu
F1 de 0,5503 para 0,5402. Priorizar maior área piorou ainda mais. **A regra não
foi adotada**; nenhum resultado do produto foi substituído pela simulação.
JSON, método e nove painéis de dados iRodent ficam somente em
`resultados/local/v4/containment_audit/`, sem fotografias do usuário.
Uma nova política exigirá novo perfil, validação e nova avaliação reservada.

Água manteve saídas iguais em 179/179 imagens dentro de `1e-5`, sem retune.
O benchmark total reúne 40 reservadas e 179 regressões, mas os resultados
não são somados numa taxa geral de acerto. As duas fotos do usuário e os vídeos
continuam fora desses denominadores.

### Estado da execução e verificação funcional

Pesos, vocabulário, hashes, resoluções, limiares por classe, regras de
verificação e fusão/NMS foram registrados em `frozen_candidate.json` antes de
consultar as previsões no novo holdout. A avaliação de validação e teste precisa usar essa
mesma configuração congelada. Uma mudança posterior exige nova versão e
descrição do que já foi observado.

Publicar separadamente, por classe e coorte, presenças corretas, perdas e
falsos alertas com seus denominadores; identificar exclusões por incerteza.
Comparar ambos os candidatos sobre as mesmas imagens e referências. A revisão
de localização deve preservar as propostas neurais originais e declarar a
qualidade das caixas de referência. Testar também custo de execução e
consistência entre imagem, vídeo, API e interface antes de promover o candidato.

Nesta etapa, o perfil permanece congelado e a calibração, CLICK40 e regressão179
estão registrados. **A API 0.5.0 está ativa e a verificação funcional foi concluída.**
Após reiniciar API/interface, `/health` respondeu `status: "ok"`,
`model_loaded: true` e `version: "0.5.0"`. Na ativação, a suíte passou em 236 testes; a conferência final da entrega passou em 239, registrada em `docs/relatorio/validacao_relatorio.json`.
Foto e vídeo usam HTTP, com redes grandes compartilhadas na API e um rastreador
YOLO pequeno por vídeo. Os endpoints são `POST /vision/predict/video` e
`GET /vision/videos/{event_id}`; a UI não cria subprocesso com os modelos pesados.

[active_profile.json](../resultados/metrics/v4/active_profile.json) fixa o
estado `verified_running_product`, os hashes e as provas. O perfil de vídeo
é `pipeline-380203274bf9c28c`, baseado no perfil de imagem congelado e com
confirmação de água por 0,5 s preservada.

O [relatório funcional](../resultados/metrics/v4/product_validation.json),
com `status: "completed"`, cobre oito itens conhecidos do catálogo: seis
vídeos e duas fotos. Todos concluíram pela API e tiveram eventos persistidos;
os seis MP4s passaram por decodificação integral dos **1.388 quadros**, comparação
de dimensões/FPS/duração, H.264, início rápido (`moov` antes de `mdat`) e
entrega parcial HTTP Range 206. O serviço permaneceu saudável em 0.5.0.

| Vídeo conhecido | Quadros | Com proposta de animal | Com proposta de água exibida | Com proposta de resíduos |
|---|---:|---:|---:|---:|
| Hanoi, `rato_residuos` | 240 | 144 | 0 | 227 |
| `rato_close` | 220 | 220 | 0 | 0 |
| `rato_rua` | 240 | 1, ambígua | 0 | 64 |
| `drenagem_chuva` | 240 | 0 | 224 | 84 |
| `canal_residuos` | 208 | 0 | 208 | 0 |
| `sarjeta_residuos` | 240 | 0 | 139 | 40 |

As fotos do catálogo retornaram duas marcações de resíduos em `ralo_lixo` e
quatro de resíduos/duas de água em `bueiro_agua`. São quantidades previstas,
não uma certificação de todos os objetos reais. Os clipes de 10,4 a 12 segundos
demandaram aproximadamente 67 a 91 segundos de processamento nessa máquina.

A [prova de interface](../resultados/metrics/v4/ui_verification.json) registra
os envios reais das duas fotos do usuário, suas marcações/alertas e Hanoi
reproduzido até 12/12 segundos sem erro do player. Não afirma que os seis
vídeos foram vistos integralmente na tela: os outros arquivos passaram pela
verificação técnica integral de transporte e decodificação. As fotografias do
usuário e suas capturas permanecem locais, fora das métricas distribuídas.

**O êxito funcional não comprova melhora temporal de detecção.** Hanoi tem
animal em 144/240 quadros, contra 187/240 no perfil anterior. A única proposta
de animal em Rua é ambígua; esse caso não é contado como acerto confirmado.
Não existem labels completos por quadro para inferir recall ou contagem de
animais físicos. Algumas fontes forneceram quadros de adaptação de treino e
continuam regressões conhecidas, não testes independentes.

A inspeção visual final também encontrou falsos resíduos na lixeira fixa de
Chuva e no cotovelo de PVC instalado em Sarjeta, além de ambiguidades com
vegetação. Em `ralo_lixo`, outros resíduos evidentes e água rasa continuam
perdidos. A região prevista de água em Canal inclui concreto. Somam-se à
piora de F1 de caixas de animal já reportada. Esses erros são preservados;
nenhum parâmetro foi mudado depois desta avaliação.

O vídeo narrado de 9min12s continua histórico da 0.4. Não foi produzida ou
publicada uma nova gravação narrada 0.5 nesta verificação. Não há alegação
de acerto universal ou de novo treinamento v4 concluído.

## Artefatos e reprodução dos dados

[resultados/metrics/v4/](../resultados/metrics/v4/README.md) contém somente
JSON/Markdown pequenos: coortes congeladas, auditoria, diagnóstico numérico e
fontes dos componentes. Os paths das imagens são relativos à raiz do projeto;
os binários/dados completos continuam locais, fora do Git. Os arquivos v3
não foram sobrescritos.

`audit_training_data_v4.py` reproduz a auditoria sem inferência. A opção de
congelamento recusa sobrescrever a coorte existente. `prepare_litter_holdout_v4.py`
prepara a amostra nativa de teste sem executar modelos; a opção
`--review-sealed` só aceita a identidade selada revisada. Para reconstruir
uma coorte publicada, conferir os registros e hashes do manifesto original;
uma nova amostragem ou novo horário de selamento não é automaticamente a
mesma evidência. Não alterar referências depois de observar previsões.
