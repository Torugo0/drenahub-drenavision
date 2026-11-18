# Gravação da demonstração - DrenaVision

Este roteiro orienta uma gravação real da versão 0.4.0. Os tempos, a narração,
as legendas e a seleção de exemplos são propostas do projeto. O pedido do
usuário é mostrar os vídeos, as imagens, a espera, os resultados e os alertas.

## O que o professor pede

Fonte conferida no PDF original `Desafio-FIAP-GS-CitizenTech 4ES.pdf`, páginas
43-44, com revisão visual das páginas. A disciplina Applied Computer Vision
pede treinamento customizado de uma arquitetura compatível com o problema,
link do GitHub com documentação e código completo e vídeo da solução
funcionando. O vídeo vale até 5 pontos. O PDF técnico vale outros 5 pontos e
deve conter título, integrantes, resumo de 250 palavras, introdução, objetivo
de 50 palavras, ferramentas, metodologia passo a passo, resultados e conclusão.

O enunciado de Computer Vision não fixa duração, plataforma de publicação,
presença da equipe na câmera ou narração. As regras de até cinco minutos,
equipe aparecendo e YouTube não listado pertencem ao pitch de Business
Intelligence, páginas 41-42. Não confundir os dois vídeos. Na página 44, a
expressão "detecção de objetos em tempo" aparece incompleta no próprio PDF;
este projeto demonstra processamento de arquivos, sem prometer vídeo ao vivo.

As regras gerais da página 24 pedem envio por um integrante no Teams da
disciplina até 19/11 às 23h55; até cinco alunos da mesma turma e o mesmo
produto nas disciplinas; repositórios, aplicações e vídeos públicos e
funcionando durante a correção, em até uma semana após o encerramento.
A página 20 pede a ficha técnica no início do README e da documentação.
Gravar localmente não cumpre, por si só, as etapas de publicação e envio.

## Preparação

1. Executar `iniciar.ps1`. Abrir a interface em `http://127.0.0.1:8501` e a API
   em `http://127.0.0.1:8000/docs`. Aguardar o modelo ficar pronto.
2. Deixar os oito arquivos originais de `dados/examples/` disponíveis para o
   seletor de arquivos. Os originais não contêm caixas do modelo.
3. Gravar a tela em tamanho legível. Usar o modo **Meu arquivo**, carregar o
   arquivo original, preencher um local de demonstração e clicar em
   **Analisar foto ou vídeo**. Manter o indicador de processamento visível
   até terminar. Repetir cada ação de verdade durante a gravação.
4. Mostrar o resultado antes de trocar de arquivo. Nos vídeos, clicar em
   reproduzir e aguardar o trecho marcado; mostrar também o fim do clipe.
5. Não iniciar um novo treinamento durante a gravação. Abrir somente as
   evidências do treinamento concluído e das avaliações já registradas.

Os resultados abaixo são os observados na validação v3. A gravação deve
descrever o que aparecer na execução atual, mesmo que seja diferente. Não
substituir a execução por resultado pré-renderizado. Se houver edição de
uma espera, informar na tela "Espera acelerada" e preservar a gravação
completa. O vídeo principal pode durar mais de cinco minutos em Computer
Vision; os tempos abaixo são apenas metas editoriais.

## Versão completa: todos os exemplos

Começar com uma explicação curta: "O DrenaVision ajuda quem cuida da drenagem
a revisar fotos e vídeos. Ele destaca possíveis resíduos, água e animais.
Os sinais precisam ser conferidos por uma pessoa."

| Ordem | Arquivo original | Mostrar depois da espera | Fala ou legenda curta |
|---|---|---|---|
| 1 | `bueiro_agua.jpg` | Resíduos e regiões de água; alerta; imagem inteira; botão para salvar | "A foto recebeu marcações de resíduos e água. A equipe pode conferir o local e salvar o resultado." |
| 2 | `rato_residuos.mp4` | Reprodução de 12 s; caixa compacta no rato à direita da telha; resíduos; alerta de animal | "O vídeo foi analisado. A marcação acompanha o possível animal. Os sinais orientam uma verificação no local." |
| 3 | `drenagem_chuva.mp4` | Reprodução; marcações de água; alerta; ausência de animal | "Aqui o sinal principal é água. O sistema não mede profundidade, vazão ou risco de enchente." |
| 4 | `canal_residuos.mp4` | Vídeo vertical completo; regiões de água | "O mesmo fluxo funciona com um vídeo vertical. Este exemplo mostra água, não um rato." |
| 5 | `sarjeta_residuos.mp4` | Água; revisar marcações em objeto branco e saída do tubo | "Algumas marcações podem estar erradas. O objeto na lateral e o tubo precisam de revisão humana; não confirmamos o material apenas pela caixa." |
| 6 | `rato_close.mp4` | Rato atrás da grade; reprodução; possíveis variações da caixa | "Neste close o animal é mais visível, mesmo com a grade. Esta fonte também forneceu quadros para a adaptação do modelo." |
| 7 | `rato_rua.mp4` | Resultado sem animal marcado; vídeo original para comparação | "Este é um caso difícil, com o roedor pequeno e parcialmente oculto. A ausência de marcação não prova ausência de animal." |
| 8 | `ralo_lixo.jpg` | Nenhuma marcação apesar dos resíduos visíveis | "Esta foto expõe uma falha: existem resíduos, mas o modelo deixou de marcar. Por isso o usuário também deve olhar o registro." |

Depois dos oito casos, abrir **Registros anteriores**, filtrar pelo nome do
local usado na gravação e mostrar uma ocorrência com data e resultado.
Salvar uma evidência, quando o botão estiver disponível. Mostrar
**Como funciona** para explicar o cuidado com a análise e com o uso humano.

Fechar com uma passagem breve pela API e pelas evidências técnicas. Não é
necessário explicar JSON para o usuário final; esta parte comprova a
implementação para o professor.

## Resumo sugerido: aproximadamente 4min50s

Esta versão prioriza quatro entradas reais: uma imagem com resíduos e água,
dois vídeos com assuntos diferentes e uma imagem de falha. As esperas reais
podem aumentar o tempo final. A versão completa atende melhor ao pedido de
mostrar todos os exemplos.

| Tempo de referência | Ação na tela | Narração sugerida |
|---|---|---|
| 0:00-0:20 | Página inicial; título; três tipos de sinal | "Este é o DrenaVision, componente de inspeção visual do DrenaHub. Ele ajuda zeladores e equipes de manutenção a conferir pontos de drenagem por foto ou vídeo." |
| 0:20-1:00 | Carregar `bueiro_agua.jpg`, informar local, analisar, aguardar; mostrar alerta e marcações | "Vou enviar uma foto original e iniciar a análise. Depois da espera, aparecem resíduos e regiões de água. A pessoa pode revisar a imagem e salvar o resultado para a equipe responsável." |
| 1:00-2:05 | Carregar `rato_residuos.mp4`, analisar, aguardar; reproduzir o resultado | "Agora envio um vídeo com um roedor junto a resíduos. O processamento ocorre quadro a quadro. A caixa destaca o possível animal durante a reprodução, e o aviso chama atenção para uma verificação. Este clipe é demonstração de uma fonte usada na adaptação, não teste independente." |
| 2:05-3:00 | Carregar `drenagem_chuva.mp4`, aguardar, reproduzir; mostrar alerta de água | "Neste segundo vídeo, o assunto é água na drenagem. O resultado destaca o sinal visual. Ele não confirma obstrução hidráulica nem prevê enchentes. Cabe à equipe conferir o local." |
| 3:00-3:30 | Carregar `ralo_lixo.jpg`, analisar, comparar com a foto | "Aqui aparece uma limitação real: há resíduos visíveis, mas o modelo não marcou. Não encontrar um sinal não garante que o local está livre. A análise precisa da revisão de uma pessoa." |
| 3:30-3:50 | Histórico; filtro por local; abrir resultado ou salvar evidência | "As análises ficam no histórico para consulta. Podemos localizar o ponto e recuperar a evidência para acompanhar a manutenção." |
| 3:50-4:10 | Swagger: `GET /health`, `GET /vision/classes`, `GET /vision/events`; mostrar um evento real | "A API permite integrar a análise a outros módulos do DrenaHub. Aqui consultamos a saúde do serviço, as classes e uma ocorrência realmente registrada nesta demonstração." |
| 4:10-4:35 | Código e relatórios v3: treinamento, dados e avaliação | "O detector YOLO recebeu treinamento customizado, com dados separados para treino, validação e teste. O histórico, os pesos e as avaliações estão documentados. A água também usa um segmentador pré-treinado, que não foi treinado pela equipe. Os resultados e as limitações ficam registrados." |
| 4:35-4:50 | Página do projeto; créditos; encerramento | "O DrenaVision apoia a inspeção humana. Os próximos passos são ampliar os dados do território e validar o produto em campo. As fontes, licenças e evidências acompanham a entrega." |

## Evidência técnica legível

Abrir estes arquivos pelo editor ou pelo repositório. Mostrar trechos curtos,
com letra grande, e não rolar relatórios enormes durante a narração.

| Evidência | Arquivo | O que comprova |
|---|---|---|
| Treinamento concluído | `resultados/metrics/v3/training_manifest.json` | 63 épocas executadas; 1.107 imagens de treino, 60 de validação e 60 de teste; pesos alterados e hashes registrados |
| Código do produto | `drenavision/`, `ferramentas/scripts/` e `README.md` | Inferência, API, interface e instruções de execução |
| Avaliação do detector | `resultados/metrics/v3/master_map_comparison.json` | YOLO isolado: mAP50 de 0,3349 para 0,4845 nas mesmas 60 imagens; não é mAP do produto híbrido |
| Avaliação do produto | `resultados/metrics/v3/independent_comparison.json` | Conjunto adicional de 119 fotos: água em 33 de 40 positivas; 2 falsos alertas nas 10 negativas revisadas; animal em 46 de 69, contra 47 antes |
| Revisão dos vídeos | `resultados/metrics/v3/product_validation_v3_confirmed.json` | Resultados nos exemplos conhecidos; não é validação independente em campo |
| Fontes das entradas | `dados/examples/sources.json` | Autores, URLs, licenças, mudanças e participação no treino |

Não chamar mAP de "porcentagem geral de acerto". Não apresentar uma detecção
de presença como prova de caixa perfeita. Nos vídeos, IDs de rastreamento
podem se repetir ou se renovar: quatro IDs não significam quatro ratos.
Os clipes de Hanói, close, chuva e canal forneceram quadros ou fundos para
adaptação; são demonstrações de fontes conhecidas, não testes independentes.
O conjunto adicional é uma avaliação com fontes públicas do mesmo domínio,
não substitui validação futura no território de uso.

## Créditos das mídias

Exibir créditos legíveis no encerramento ou em cartelas entre os exemplos,
e acompanhar a publicação com esta lista completa. As entradas foram
redimensionadas/convertidas; os clipes receberam recorte de até 12 segundos,
20 FPS, H.264 e remoção do áudio. O resultado demonstrado acrescenta
marcações do modelo sobre a mídia original. Conferir a atribuição e as
condições de compartilhamento aplicáveis antes de publicar uma versão editada.

| Entrada | Autor e licença | Fonte |
|---|---|---|
| Rato atrás da grade | Ykmyks - [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [Rat.ogv](https://commons.wikimedia.org/wiki/File:Rat.ogv) |
| Roedor parcialmente oculto | NaIzletuSi ™ - [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | [Rodentia, Ljubljana](https://commons.wikimedia.org/wiki/File:2022-08-02_LJUBLJANA_Grudnovo_nabre%C5%BEje_RODENTIA_Rattus_PODGANA.webm) |
| Roedor junto a resíduos | RG72 - [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [Manĝanta rato en Hanojo](https://commons.wikimedia.org/wiki/File:Man%C4%9Danta_rato_en_Hanojo.webm) |
| Água na drenagem | Xyxyzyz - [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) | [Overflowing water drain Werrington March 2022](https://commons.wikimedia.org/wiki/File:Overflowing_water_drain_Werrington_March_2022.webm) |
| Canal com água | Ibnali1 - [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [Alajo Storm Drain 011](https://commons.wikimedia.org/wiki/File:Alajo_Storm_Drain_011.webm) |
| Água na sarjeta | Ibnali1 - [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [Gutter to the Alajo Storm Drain](https://commons.wikimedia.org/wiki/File:Gutter_to_the_Alajo_Storm_Drain.webm) |
| Lixo junto ao bueiro | Tdorante10 - [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | [Sutphin Bl Jamaica Av, Queens](https://commons.wikimedia.org/wiki/File:Sutphin_Bl_Jamaica_Av_td_(2019-08-18)_14.jpg) |
| Embalagens e água | Bart Everson - [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | [Storm Drain Clogged, New Orleans](https://commons.wikimedia.org/wiki/File:Storm_Drain_Clogged_-_Mid-City_New_Orleans.jpg) |

O treinamento YOLO e o segmentador complementar têm fontes e licenças
documentadas no projeto. O SegFormer NVIDIA é pré-treinado e tem licença de
pesquisa/avaliação não comercial; não apresentá-lo como treinamento da equipe
nem declarar que todo o projeto pode ser comercializado sob MIT.

## Conferência da gravação

- O vídeo mostra o envio real, o clique, a espera e o resultado de cada caso.
- Os resultados reproduzem, as caixas são legíveis e o alerta aparece na tela.
- A demonstração inclui uma falha real e explica o papel da revisão humana.
- O histórico contém as ocorrências criadas durante a gravação.
- O MP4 abre e reproduz até o final, com tempo e tamanho de imagem corretos.
- Cortes ou acelerações estão identificados; não há resultados fabricados.
- Legendas e narração correspondem ao que a execução mostrou.
- Créditos e links acompanham o arquivo. Publicação e envio são etapas
  separadas; registrar os links reais no README depois de concluídas.
