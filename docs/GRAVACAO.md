# Gravação local da demonstração

## Estado da entrega

A captura real da interface está concluída: seis vídeos e duas fotografias
foram enviados, analisados e apresentados. Os seis vídeos foram reproduzidos
até o final. O histórico foi conferido com oito registros pelo filtro
`Apresentação FIAP`, e uma consulta real `GET /health` recebeu HTTP 200.

**O MP4 final está concluído, narrado, legendado, exportado e revisado.** Tem
**9min12,46s** (552,458333s), 1280 × 720, H.264/AAC a 24 FPS e faststart,
com **14.722.602 bytes**. Vídeo e áudio foram decodificados integralmente sem
erros e com diferença de duração de **0,000s**. O pico de áudio foi
**−6,76 dBFS**, sem amostras em clipping. As folhas de contato dos oito casos,
cartelas, créditos e legendas passaram por revisão visual; a caixa compacta
do rato de Hanói também foi conferida em captura adicional.

As verificações estão em `resultados/local/gravacao/qa_final.json` e
`verificacao_ui.json`. O vídeo permanece local. O PDF técnico, os links
públicos e o envio no Teams ainda estão pendentes.

O roteiro completo e a proposta de resumo estão em
[ROTEIRO_GRAVACAO.md](ROTEIRO_GRAVACAO.md). Computer Vision exige vídeo da
solução funcionando, sem definir sua duração nas páginas 43-44 do enunciado.
A versão completa preserva as esperas reais para atender ao pedido de mostrar
todos os exemplos; ela pode ultrapassar cinco minutos.

## Arquivos locais

Os arquivos estão em `resultados/local/gravacao/`.

| Arquivo | Conteúdo | Estado |
|---|---|---|
| `drenavision_demonstracao_completa.mp4` | Captura da interface, narração e resultados reais | Concluído e revisado |
| `drenavision_demonstracao_completa.srt` | Legendas com tempos | Concluídas |
| `drenavision_demonstracao_completa.txt` | Transcrição da narração | Concluída |
| `drenavision_demonstracao_completa.manifest.json` | Método, duração, arquivos, capítulos e proveniência da gravação | Concluído |
| `CREDITOS_DAS_MIDIAS.md` | Cópia autocontida das fontes, autores, licenças e alterações das oito entradas | Preparada para acompanhar o MP4 |
| `qa_final.json` | Decodificação, sincronismo de duração, áudio, formato, tamanho e hash | Verificações concluídas |
| `verificacao_ui.json` | Resultados observados nas oito execuções, reprodução, histórico e API | Verificação concluída |

O pacote de compartilhamento é `pacote_gravacao_drenavision.zip`, na mesma
pasta, com o MP4, legendas, transcrição, manifesto, créditos e verificações.
O hash SHA-256 do MP4 é
`6d3c19eaa819f281ff5e7910cbff21863e701370de3b5b2aab12a97b1b2d355a`.

A pasta `resultados/local/` é ignorada pelo Git. O MP4 e os arquivos auxiliares
não serão baixados ao clonar o repositório. Este documento e o roteiro ficam
versionados em `docs/`, para que o avaliador entenda o conteúdo e o método.
Um link público real deverá ser registrado no README após a publicação.

## Capítulos

Tempos obtidos de `output_start_seconds` no manifesto final e arredondados
para baixo ao segundo. Os oito trechos de envio são seguidos pela espera e
pelo resultado da análise. Treinamento, avaliação e créditos são cartelas
explicativas; os uploads, resultados, histórico e API são capturas reais.

| Tempo | Trecho |
|---|---|
| 00:00 | DrenaVision: inspeção visual assistida |
| 00:23 | Tela inicial |
| 00:28 | 1 de 8 · Roedor junto a resíduos - envio e espera |
| 01:00 | 1 de 8 · Animal e resíduos - reprodução completa |
| 01:16 | 2 de 8 · Água na drenagem - envio e espera |
| 01:49 | 2 de 8 · Água na drenagem - resultado e reprodução |
| 02:04 | 3 de 8 · Canal com água - envio e espera |
| 02:31 | 3 de 8 · Canal com água - resultado e reprodução |
| 02:46 | 4 de 8 · Sarjeta e revisão de alerta - envio e espera |
| 03:15 | 4 de 8 · Sarjeta e revisão de alerta - resultado e reprodução |
| 03:30 | 5 de 8 · Roedor atrás da grade - envio e espera |
| 03:58 | 5 de 8 · Roedor atrás da grade - resultado e reprodução |
| 04:13 | 6 de 8 · Roedor parcialmente oculto - envio e espera |
| 04:43 | 6 de 8 · Roedor parcialmente oculto - resultado e reprodução |
| 04:58 | 7 de 8 · Foto de resíduos - envio e espera |
| 05:04 | 7 de 8 · Falha em resíduos visíveis |
| 05:17 | 8 de 8 · Foto com água e embalagens - envio |
| 05:23 | 8 de 8 · Quatro resíduos e duas regiões de água |
| 05:43 | Registros anteriores |
| 05:58 | Como funciona: três passos |
| 06:10 | As oito análises da apresentação |
| 06:20 | API: documentação para integração |
| 06:32 | API: resposta real HTTP 200 |
| 06:40 | Modelo e treinamento customizado |
| 07:09 | Avaliação e limites observados |
| 07:41 | Uso responsável e próximos passos |
| 08:07 | Créditos das mídias · 1 de 2 |
| 08:27 | Créditos das mídias · 2 de 2 |
| 08:51 | DrenaVision · demonstração concluída |

Duração final: **09:12,46**. As esperas e as reproduções capturadas não
foram aceleradas; a taxa de distribuição a 24 FPS não cria movimento novo.

## Método de captura e edição

- A demonstração usa a aplicação local em `http://127.0.0.1:8501`, por meio
  dos controles reais do navegador: upload, botão de análise, espera,
  reprodução, consulta ao histórico e acesso à API.
- A captura da tela foi feita em **1280 x 720**, com aproximadamente **8
  quadros por segundo** nos blocos gravados; o intervalo mediano foi
  0,127847s, equivalente a **7,82 FPS**, com **3.423 imagens JPEG** na captura
  bruta. A média global do índice inclui intervalos sem captura entre blocos;
  ela não representa a cadência da gravação. Essa é a taxa do screencast,
  não o desempenho de inferência nem a taxa dos clipes enviados.
- A distribuição final é MP4 H.264 a **24 quadros por segundo**,
  mantendo a duração real. A repetição de quadros ajusta a taxa de saída;
  não acelera a execução, a espera ou a reprodução.
- As esperas pelo processamento foram preservadas, **sem aceleração**.
  Foram retirados apenas intervalos não capturados de orquestração entre
  blocos. A execução e a reprodução permanecem em velocidade 1×. As
  cartelas explicativas são identificadas como tais; as telas do produto
  são capturas reais, sem substituir a análise por resultados fabricados.
- A narração usa a voz **Maria**, sintética e executada offline.
  Ela não é a voz de um integrante da equipe. O texto explica o que a
  aplicação realmente mostrou; legendas e transcrição acompanham o MP4.
- A montagem passou por decodificação completa de áudio/vídeo e inspeção
  visual de folhas de contato. A duração dos dois fluxos é a mesma, sem
  erro de decodificação. Não se trata de transmissão ao vivo da câmera.

O resultado persistido na sessão pode reaparecer ao remover e reenviar o
mesmo arquivo sem iniciar outra análise. Para repetir uma tomada com prova
clara de nova execução, mudar o nome do local, confirmar a edição e clicar
novamente em **Analisar foto ou vídeo**. O botão inicia uma análise nova;
a troca de local também impede que o resultado da tomada anterior apareça
antes do clique.

## O que os exemplos demonstram

| Entrada | Assunto e ponto de atenção |
|---|---|
| `bueiro_agua.jpg` | Resíduos e água junto à grelha; fotografia sem rato |
| `rato_residuos.mp4` | Roedor junto a resíduos, em Hanói; o v3 marca o animal com caixa compacta |
| `drenagem_chuva.mp4` | Água em movimento na drenagem; não mede profundidade ou risco |
| `canal_residuos.mp4` | Água em canal, em vídeo vertical; não é exemplo de rato |
| `sarjeta_residuos.mp4` | Água na sarjeta; possíveis falsos resíduos no tubo/objeto da lateral precisam de revisão |
| `rato_close.mp4` | Roedor atrás da grade; fonte conhecida da adaptação |
| `rato_rua.mp4` | Caso difícil: animal pequeno/oculto, que permanece sem marcação confirmada |
| `ralo_lixo.jpg` | Falha: resíduos visíveis que o modelo não marcou; fotografia sem rato |

O close, Hanói, chuva e canal forneceram quadros ou fundos para adaptação.
São demonstrações de fontes conhecidas, não testes independentes. A
qualidade do produto não é medida apenas por esses exemplos: os relatórios
completos estão em `resultados/metrics/v3/`. A ausência de marcação não
comprova ausência de problema. Uma pessoa deve conferir o registro e o local.
IDs de rastreamento não correspondem necessariamente a animais diferentes.

## Créditos completos das oito entradas

Origem dos metadados: `dados/examples/sources.json`. A lista identifica o
arquivo usado, o autor, o título original, a URL de origem, a licença e as
alterações. Os clipes são filmagens reais; as duas imagens são fotografias.
Os arquivos originais de entrada não contêm previsões do modelo. As caixas,
regiões coloridas e alertas são acrescentados pelo produto no resultado.

| Arquivo usado | Autor | Título original e URL de origem | Licença | Alterações |
|---|---|---|---|---|
| `rato_close.mp4` | Ykmyks | [Rat.ogv](https://commons.wikimedia.org/wiki/File:Rat.ogv) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | Trecho de até 12 s a partir de 0 s; H.264; máximo 960x720; 20 FPS; áudio removido. Exibição na interface e anotações do modelo no resultado. |
| `rato_rua.mp4` | NaIzletuSi ™ | [2022-08-02 LJUBLJANA Grudnovo nabrežje RODENTIA Rattus PODGANA.webm](https://commons.wikimedia.org/wiki/File:2022-08-02_LJUBLJANA_Grudnovo_nabre%C5%BEje_RODENTIA_Rattus_PODGANA.webm) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | Trecho de até 12 s a partir de 0 s; H.264; máximo 960x720; 20 FPS; áudio removido. Exibição na interface e anotações do modelo no resultado. |
| `rato_residuos.mp4` | RG72 | [Manĝanta rato en Hanojo.webm](https://commons.wikimedia.org/wiki/File:Man%C4%9Danta_rato_en_Hanojo.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | Trecho de até 12 s a partir de 6 s; H.264; máximo 960x720; 20 FPS; áudio removido. Exibição na interface e anotações do modelo no resultado. |
| `drenagem_chuva.mp4` | Xyxyzyz Author is Wikimedia user Xyxyzyz. | [Overflowing water drain Werrington March 2022.webm](https://commons.wikimedia.org/wiki/File:Overflowing_water_drain_Werrington_March_2022.webm) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0) | Trecho de até 12 s a partir de 0 s; H.264; máximo 960x720; 20 FPS; áudio removido. Exibição na interface e anotações do modelo no resultado. |
| `canal_residuos.mp4` | Ibnali1 | [Alajo Storm Drain 011.webm](https://commons.wikimedia.org/wiki/File:Alajo_Storm_Drain_011.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | Trecho de até 12 s a partir de 0 s; H.264; máximo 960x720; 20 FPS; áudio removido. Exibição na interface e anotações do modelo no resultado. |
| `sarjeta_residuos.mp4` | Ibnali1 | [Gutter to the Alajo Storm Drain.webm](https://commons.wikimedia.org/wiki/File:Gutter_to_the_Alajo_Storm_Drain.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | Trecho de até 12 s a partir de 0 s; H.264; máximo 960x720; 20 FPS; áudio removido. Exibição na interface e anotações do modelo no resultado. |
| `ralo_lixo.jpg` | Tdorante10 | [Sutphin Bl Jamaica Av td (2019-08-18) 14.jpg](https://commons.wikimedia.org/wiki/File:Sutphin_Bl_Jamaica_Av_td_(2019-08-18)_14.jpg) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | Redimensionamento para até 1600x1200 e conversão JPEG; metadados removidos. Exibição na interface e anotações do modelo no resultado. |
| `bueiro_agua.jpg` | Bart Everson | [Storm Drain Clogged - Mid-City New Orleans.jpg](https://commons.wikimedia.org/wiki/File:Storm_Drain_Clogged_-_Mid-City_New_Orleans.jpg) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0) | Redimensionamento para até 1600x1200 e conversão JPEG; metadados removidos. Exibição na interface e anotações do modelo no resultado. |

Os créditos devem acompanhar a publicação, em tela legível e na descrição
com os links completos. A cópia local autocontida
`resultados/local/gravacao/CREDITOS_DAS_MIDIAS.md` acompanha o MP4 mesmo quando
ele é compartilhado fora do Git. Duas cartelas de créditos aparecem ao final
da montagem e tiveram presença e legibilidade conferidas.
Os termos das fontes continuam aplicáveis às mídias;
a licença do código não substitui as licenças de terceiros. Informações
adicionais sobre modelos e dados estão em
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

## Revisão final concluída

- [x] Confirmar que o MP4 e os arquivos auxiliares foram exportados.
- [x] Decodificar vídeo e áudio completos sem erros; confirmar duração e término.
- [x] Confirmar que o envio, a espera e o resultado de cada um dos oito casos
  aparecem com clareza, sem acelerar as esperas.
- [x] Conferir alertas, caixa compacta do rato, regiões de água e exemplos de falha nas capturas.
- [x] Conferir os oito registros do histórico e a resposta HTTP 200 da API.
- [x] Conferir voz Maria, transcrição, legendas e sincronismo de duração de áudio/vídeo.
- [x] Conferir créditos e descrição honesta do papel dos exemplos no treino.
- [x] Registrar duração, formato, verificações e hash no manifesto e em `qa_final.json`.

## Como publicar e enviar

1. Abrir o MP4 da pasta local. O arquivo revisado está pronto para publicação;
   conferir também a reprodução no dispositivo usado para a entrega.
2. Compartilhar também `CREDITOS_DAS_MIDIAS.md`, as legendas e a transcrição
   ao enviar o MP4 como arquivo. Publicar o vídeo em um serviço de hospedagem com link acessível
   ao avaliador, sem depender de login na conta do aluno. Incluir créditos,
   licenças e links de origem na descrição; disponibilizar as legendas.
3. Abrir o link em uma janela anônima ou em outro dispositivo e testar a
   reprodução até o final.
4. Adicionar a URL real na seção de gravação do README e atualizar o estado
   no checklist de [ENTREGA.md](ENTREGA.md). Não usar o caminho local como
   substituto do link público.
5. Publicar também repositório e aplicação/protótipo conforme as regras da
   entrega. Manter os links funcionando durante toda a correção.
6. Um integrante faz o envio no Teams da disciplina até **19/11/2026 às
   23h55**. Conferir integrantes, turma e os demais arquivos exigidos.

A publicação, a alteração de visibilidade do repositório e o envio acadêmico
não foram realizados automaticamente nesta etapa.
