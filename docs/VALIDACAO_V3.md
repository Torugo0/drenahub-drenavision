# Protocolo de revisão e validação v3

Esta revisão trata das falhas observadas em água, resíduos e roedores no produto.
O protocolo abaixo foi registrado durante a preparação dos dados, antes da
avaliação do novo checkpoint. Os resultados devem ser publicados separadamente;
esta página não apresenta métricas ainda não calculadas nem promete detectar
todos os objetos.

## Diagnóstico

No vídeo do rato junto a resíduos, o v2 reconheceu os cinco recortes de treino
com confiança entre 0,453 e 0,619, mas perdeu o animal nos quadros completos.
Os recortes tinham aproximadamente 100 × 70 pixels e eram ampliados para o
treino. Faltava ensinar a localizar o animal pequeno dentro da cena. Testes em
CPU com entradas 640, 960 e 1280, além de recortes genéricos sobrepostos, não
resolveram a falha: reapareceram caixas sobre telhas e outras áreas de fundo.
Por isso, aumentar a escala ou baixar a confiança não foi adotado como correção.

Na fonte de água, uma mesma superfície pode ter várias caixas pequenas que se
tocam ou se sobrepõem. Esse alvo fragmentado difere do que o usuário espera ver
na imagem inteira. Além disso, as imagens de origem têm 256 × 256 pixels; só
aumentar sua quantidade não representa toda a variedade de um bueiro filmado.

## Dados preparados

O conjunto revisado fica separado em `dados/processed/drenavision_v3_clean/`.
O v2 e os arquivos originais são preservados. Após a exclusão dos negativos de
água sem referência confiável, a montagem concluída contém **1.107 imagens de
treino, 60 de validação e 60 de teste**, conforme o relatório de preparação.

| Alteração | Uso e proteção |
|---|---|
| Água da fonte: 100 imagens positivas de treino anteriores + 400 adicionais | 500 positivas de treino. A fonte não tem uma partição oficial; as adicionais foram atribuídas ao treino por regra determinística. |
| Exclusão das imagens da fonte de água com rótulo vazio | A pasta de superfície molhada contém também poças visíveis. Suas imagens não podem ser tratadas automaticamente como cenas sem água. Os oito negativos antigos de treino, dois de validação e dois de teste são excluídos; os 24 candidatos adicionais não entram. |
| CLICK: 250 imagens nativas de treino, sendo 150 adicionais | Rótulos publicados de resíduos, sem imagens nativas de validação/teste no suplemento. Os 100 arquivos anteriores foram deduplicados por hash. |
| Cinco quadros completos de `rato_residuos`, em 0, 2, 4, 6 e 10 s | Caixas revisadas para o rato visível e a pilha de resíduos; substituem os cinco recortes pequenos do v2. A telha isolada permanece fundo. |
| Oito quadros completos de água de `drenagem_chuva` e `canal_residuos` | Adaptação ao contexto de drenagem, quatro instantes por fonte. Os vídeos completos são fontes de treino. |

As anotações dos quadros de contexto resultam de inspeção visual assistida,
não de previsões do detector. Elas não são uma referência certificada por um
anotador humano independente. A caixa da pilha cobre os resíduos contíguos,
incluindo pequenos espaços entre itens; não prova bloqueio da drenagem.
Os cinco recortes de fundo revisados visualmente na adaptação anterior
permanecem como negativos; não são selecionados apenas pelo nome de uma pasta.

A primeira montagem v3 que ainda incluía negativos de superfície molhada foi
preservada para auditoria, e seu treinamento foi interrompido após a revisão
visual revelar o conflito. O candidato do conjunto limpo parte novamente do
v2 preservado. O treino interrompido não serve de origem nem de evidência de
desempenho do novo candidato.

Os 250 pares CLICK passaram por verificação de hashes, leitura das imagens e
validação das coordenadas dos rótulos. Foram inspecionados visualmente 16 novos
exemplos, com conferência em resolução original de três cenas mais complexas.
Essa amostra não certifica individualmente as 250 anotações.

## O que mudou nas caixas de água

O v3 une caixas da classe água quando as **caixas originais** se intersectam ou
se tocam, inclusive em uma borda ou canto. A conexão não aceita distância
positiva entre caixas e não usa o retângulo que vai crescendo para decidir
novas conexões. Assim, uma região separada que apenas intersecta o espaço
vazio de um retângulo já unido não é incorporada por esse motivo.

Cada componente conectado recebe seu menor retângulo alinhado aos eixos.
**Esse retângulo pode conter solo seco, margens ou espaços vazios internos.**
Não é uma máscara de segmentação nem uma medição da área real de água. Nas
imagens já presentes no v2, as 1.488 caixas de água tornaram-se 370 envelopes
de componentes, antes da inclusão das imagens adicionais.

O ID `agua_acumulada` continua no contrato da API. Para a triagem visual desta
revisão, o alvo inclui água visível parada ou em movimento. A presença de uma
caixa não estabelece que a água está parada, contaminada ou que haverá enchente.

Como os alvos de água mudaram, a comparação justa precisa avaliar **o v2 e o
candidato v3 sobre os mesmos rótulos v3**. A métrica histórica do v2, calculada
sobre caixas fragmentadas, não pode ser comparada diretamente com a nova.

## Isolamento e reprodução

Das 62 imagens de validação e 62 de teste do v2, duas por partição são removidas
pela regra explícita de exclusão dos negativos de água sem referência confiável.
As 60 restantes em cada partição mantêm seus pixels, hashes, identidades e
partições. Nessas imagens, somente a regra de caixas de água muda. Os rótulos
de animais e resíduos permanecem iguais. As imagens extras e os quadros dos
vídeos entram exclusivamente no treino. As comparações entre checkpoints usam
a mesma lista final de imagens e as mesmas exclusões, nunca denominadores
diferentes para favorecer um candidato.

A montagem exclui sobreposição com as partições protegidas por hash exato,
grupo de origem e pHash com distância de até quatro bits. A validação final
do conjunto montado passou sem erros ou avisos. Testes com dados sintéticos
verificam também preservação dos pixels das imagens protegidas, imutabilidade
do conjunto anterior e rejeição de suplementos indevidos.

Uma avaliação adicional foi preparada com **119 imagens** ainda não utilizadas
no v2: 69 animais do teste nativo iRodent, 40 imagens de água e dez negativos de
superfície molhada. Imagens de geometria insuficiente ou negativos visualmente
ambíguos foram separados antes das previsões. A seleção foi verificada contra
todas as imagens do v2 e do v3 por hash, grupo e pHash. Essa verificação passou
sem violações. As novas imagens de água têm uma atribuição de teste do projeto,
pois a fonte não oferece uma partição oficial.

A referência iRodent também não é uma contagem exaustiva dos animais da foto.
Na revisão visual sem previsões, `obs_id_00330.jpg` e `obs_id_0036.jpg` mostram
dois roedores, e `obs_id_00656.jpg` mostra vários hamsters; cada imagem tem
somente uma referência de pose convertida em caixa. Uma detecção verdadeira
de outro animal pode ser contabilizada como previsão sem correspondência.
O conjunto selado permanece inteiro e inalterado. Suas métricas medem acordo
com essas referências incompletas; não provam precisão de contagem de todos os
animais. A fonte contém diferentes tipos de roedores, sem certificar que todo
animal seja um rato urbano.

O registro detalhado da montagem fica em
`resultados/local/preparation_v3_clean/preparation_report.json`; a seleção adicional,
em `dados/processed/extra_test_v3/`, com seu relatório de isolamento. Os scripts
reprodutíveis são `prepare_water_v3.py`, `prepare_rat_context_v3.py`,
`prepare_v3_dataset.py` e `prepare_extra_test.py`, dentro de
`ferramentas/scripts/`. As instruções gerais de ambiente e execução ficam em
[REPRODUZIR.md](REPRODUZIR.md).

## Avaliação e limites da conclusão

O ajuste parte do checkpoint v2 preservado, com tamanho 640, semente 42 e
seleção de época pela validação. O limiar YOLO é **0,35** para resíduos/água,
e **0,60** para animal, escolhido na validação para reduzir falsos alertas.
A IoU de supressão é **0,45**. Os limiares não devem ser reduzidos depois de observar
o teste para conseguir marcar os exemplos. A avaliação de AP usa seu próprio
limiar baixo de coleta de candidatos, aplicado igualmente aos dois modelos;
esse valor não representa a configuração usada pela interface.

O teste original de 62 imagens já foi examinado durante revisões anteriores.
Sua seleção limpa de 60 imagens serve à
comparação controlada e à regressão, mas não equivale a uma amostra nova de
uso em campo. As 119 imagens adicionais ampliam a avaliação sem entrar no
treino. O checkpoint e as configurações devem ser fixados antes de calcular
seus resultados; os erros precisam continuar registrados.

Os oito exemplos disponíveis na interface devem ser processados novamente,
com inspeção das caixas e da reprodução dos vídeos, inclusive quando o retorno
for vazio. Os clipes usados na adaptação verificam a integração e a consistência
do treino; seu acerto não é uma estimativa de generalização. `rato_rua`,
`sarjeta_residuos`, `ralo_lixo` e `bueiro_agua` permanecem fora da adaptação,
mas já foram observados durante o desenvolvimento e são casos de regressão.

## Calibração adicional de presença de água

Um experimento separado usa o SegFormer B1 pré-treinado em ADE20K para reconhecer
superfícies de água. As 60 imagens de validação receberam revisão visual de
presença de água, considerando qualquer classe de origem, antes de consultar
previsões desse modelo. São 24 positivas, 35 negativas e uma incerta por
desfoque, excluída da calibração binária. Isso evita classificar como erro uma
água real que não foi anotada na fonte de roedores. A revisão assistida não é
uma certificação humana independente nem referência de segmentação por pixel.

O primeiro conjunto de parâmetros de probabilidade por pixel e área de região
não atingiu a especificidade mínima de 0,90: sua melhor especificidade foi
31/35, com quatro falsos alertas de água. O relatório foi preservado como
rejeitado. Os falsos alertas recaíam sobre partes da pelagem de roedores, mesmo
com probabilidade elevada; confiança do modelo não equivale a certeza física.

Uma segunda rodada ampliou somente os limiares na mesma validação, reutilizando
os mesmos mapas neurais em cache e sem novas inferências. Manteve-se o critério
de maximizar F1 de presença com especificidade de pelo menos 0,90, seguido de
desempates explícitos. O perfil selecionado exige probabilidade de pixel 0,95
e área de componente de pelo menos 5% da imagem, com processamento 512.
A união lógica de presença com a água do YOLO resultou em 23 verdadeiros
positivos, zero falsos positivos, uma omissão e 35 verdadeiros negativos na
validação clara. O YOLO sozinho tinha 19 acertos de água e cinco omissões nos
mesmos casos. A substituição completa da água do YOLO pelo segmentador não foi
a política selecionada.

Esse resultado mede **presença na imagem**, não a qualidade das caixas, a
precisão de cada pixel nem o desempenho em campo. O ajuste de parâmetros usa
validação e torna sua pontuação otimista. O teste selado de 119 imagens não
entrou nessa escolha. Não houve busca de outro modelo nessa calibração, mudança
dos rótulos após as previsões, preenchimento de lacunas ou regras especiais
para imagens de animais. O registro e os hashes ficam em
`resultados/metrics/v3/water_presence_calibration.json`; a referência visual
selada fica em `val_water_presence_review.json` no mesmo diretório.

Animais ocultos, objetos pequenos, reflexos, baixa luz, imagens comprimidas e
diferenças entre a fonte e uma drenagem real continuam sendo limitações. Sem
marcações significa que o modelo não encontrou sinais acima do limiar, e não
que a cena está livre de animais, resíduos ou água.

## Confirmação temporal posterior, somente em vídeo

O perfil de **imagem** `pipeline-1314fcfa94f5f8cf` e seus resultados em 179
imagens permanecem congelados. Depois dessa avaliação, a revisão dos vídeos
completos conhecidos encontrou sete propostas indevidas de água em quadros
desfocados de Hanoi. Esse material é regressão/adaptação, não teste de vídeo
independente.

A etapa de vídeo passou a exigir uma sequência de água com pelo menos 0,5 s
(`ceil(FPS × 0,5)` quadros). Se a sequência máxima for menor, todas as propostas
de água desse vídeo são retiradas da apresentação e a exportação é redesenhada
dos quadros originais com as demais classes. As propostas brutas e seus
metadados ficam preservados no JSON. Não há alteração de pesos, parâmetros
da imagem, nomes de arquivo ou coordenadas fixas nessa regra.

O perfil de vídeo é `pipeline-4d613d5a1d82cb61`, com base `1314…`. Em Hanoi,
a sequência máxima era quatro quadros, contra dez exigidos: sete propostas
brutas e zero exibidas. As 187 previsões de animal e 227 de resíduos, com suas
coordenadas/confianças/IDs, são idênticas às do processamento bruto. A água
real de chuva, canal e sarjeta continua confirmada. Oito arquivos e 1.388
quadros foram reprocessados e decodificados sem perda.

Essa etapa não melhora retroativamente as métricas das 179 imagens nem
comprova generalização de vídeo. Persistem alertas de resíduos sobre tubulação
fixa, o rato oculto e o lixo não marcado em `ralo_lixo`. A interface mostra
presença do sinal nos vídeos; IDs técnicos não significam vários ratos físicos.
Veja `video_confirmation_scope.json` e `product_validation_v3_confirmed.json`
em `resultados/metrics/v3/`.
