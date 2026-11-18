# Entrega acadêmica — FIAP

## Identificação e fonte dos requisitos

FIAP — Engenharia de Software. Global Solution 2026.2 — CitizenTech.
Disciplina: **Applied Computer Vision**. Atividade: **Enxergando o Mundo Real**.
Professores: Arnaldo Viana e Paulo Rodrigues. Projeto: DrenaHub / DrenaVision.

| Integrante | RM |
|---|---|
| Gabriel Machado Carrara Pimentel | RM99880 |
| Lourenzo Ramos | RM99951 |
| Vítor Hugo Rodrigues | RM97758 |

A referência é o desafio fornecido à equipe, páginas 43–44 e regras gerais.
O enunciado é fonte de requisitos; escolhas adicionais de arquitetura são do
projeto. PDF e vídeo valem **até 5 pontos cada**. Applied Computer Vision não
fixa duração mínima ou máxima do vídeo. O limite de cinco minutos nas páginas
41–42 pertence ao pitch de Business Intelligence & Startup Development.

## Documentos da entrega

- [Relatório técnico em PDF](relatorio/DrenaVision_Applied_Computer_Vision_FIAP.pdf).
- [Fonte editável do relatório](relatorio/relatorio_fonte.md).
- [Checklist de requisitos e páginas](relatorio/checklist_requisitos.md).
- [Reprodução do projeto](REPRODUZIR.md), [arquitetura e API](TECNICO.md) e
  [ficha de modelo](MODEL_CARD.md).

O relatório apresenta a explicação do código e das ferramentas, dataset,
pré-processamento, treinamento, avaliação, inferência e integração. A
validação do resumo de 250 palavras e do objetivo de 50 palavras acompanha
os artefatos do documento. A parte técnica pode ser lida no PDF e no GitHub;
o vídeo pode concentrar-se na demonstração simples da solução funcionando.

| Requisito oficial | Evidência |
|---|---|
| Detecção de objetos e/ou realidade aumentada | Aplicação de detecção em fotos e vídeos, com três classes técnicas e revisão humana |
| Treinamento customizado compatível com o problema | YOLO26n v3: 63 épocas executadas, melhor checkpoint na 43, manifesto, logs e hash; a revisão 0.5 mantém esse treino |
| Repositório GitHub com documentação e código | [Repositório real](https://github.com/Torugo0/drenahub-drenavision); publicação pública autorizada, em preparação, com confirmação de acesso após o envio |
| Vídeo da solução funcionando | `demo_drena_vision.mp4` fornecido pela equipe, 2min13,20s, enviado como arquivo MP4 anexado |
| PDF com título, integrantes, resumo, introdução, objetivo, ferramentas, metodologia, resultados e conclusão | Relatório oficial em `docs/relatorio/`, com fonte editável e checklist |

Grounding DINO Tiny, SigLIP 2 e SegFormer B1 são componentes pré-treinados,
com revisões, hashes e licenças registrados. Não foram treinados pela equipe
nesta revisão. Seus resultados não são apresentados como mAP do YOLO isolado.

## Vídeo simples fornecido pela equipe

Arquivo local fornecido: `C:/Users/Vitor/Downloads/demo_drena_vision.mp4`.
A revisão confirmou **2min13,20s**, Full HD, 30 FPS, H.264/AAC e decodificação
integral dos 3.996 quadros sem erro. A gravação mostra seleção/envio de
arquivos, processamento, marcações de animal, resíduos e água, além de alertas.
A trilha de áudio está silenciosa. Por orientação da equipe, o vídeo será
enviado **como arquivo MP4 anexado**, sem compactação do projeto.
A página 44 exige a demonstração funcionando, sem determinar plataforma de
publicação ou URL pública do vídeo.

## Três arquivos para enviar ao professor

1. `DrenaVision_Applied_Computer_Vision_FIAP.pdf`, como documento separado.
2. `demo_drena_vision.mp4`, como vídeo anexado.
3. Um TXT com a URL real `https://github.com/Torugo0/drenahub-drenavision`.

O PDF também fica visível no GitHub, em `docs/relatorio/`, com acesso destacado
no início do README. O repositório contém o código, a documentação, os testes,
as configurações e as evidências selecionadas. Modelos grandes, dados brutos,
banco de uso, logs, caches, saídas de trabalho e as duas fotos locais de
regressão sem revisão de origem/licença não são enviados ao histórico do Git.
As instruções para preparar o ambiente e obter os pesos estão em `REPRODUZIR.md`.

Mostrar código linha por linha não é uma exigência expressa do vídeo na
página 44. Uma explicação curta, narrada ou legendada, pode apresentar o
problema, o envio de foto/vídeo e o significado das marcações. As esperas de
processamento podem ser encurtadas ou aceleradas, desde que isso seja indicado.
O PDF deve sustentar a explicação técnica, inclusive o treinamento customizado.

A gravação narrada de **9min12,46s** permanece como histórico da **versão 0.4.0**.
Método, créditos e capítulos estão em [GRAVACAO.md](GRAVACAO.md) e
[ROTEIRO_GRAVACAO.md](ROTEIRO_GRAVACAO.md). Seus resultados não substituem a
avaliação do produto 0.5 nem comprovam o comportamento atual.

## Evidências atuais e seus limites

O perfil de imagem 0.5 é `pipeline-4c5c993223219929`; o de vídeo é
`pipeline-380203274bf9c28c`. A configuração congelada, as provas e as limitações
estão em [VALIDACAO_V4.md](VALIDACAO_V4.md) e `resultados/metrics/v4/`.

Há avaliação separada de 40 fotos reservadas e 179 regressões conhecidas;
não são reunidas numa taxa geral de acerto. A presença de animais melhora,
mas o F1 das caixas cai em ambas as coortes conhecidas. Resíduos apresentam
maior cobertura e mais falsos positivos. Os resultados de água permanecem
iguais nesse comparativo; o sistema não mede profundidade ou risco hidráulico.

A verificação funcional cobriu seis vídeos e duas fotos, com 1.388 quadros
inteiramente decodificados, persistência, H.264 e HTTP Range 206 verificados.
Isso comprova execução e transporte, não acerto universal. Rua continua sem
acerto confirmado; Hanoi tem menor cobertura temporal de animal que na 0.4.
Persistem caixas amplas/duplicadas, falsos resíduos em objetos instalados e
perdas de resíduos e água rasa. O relatório mantém esses erros.

O treino, pesos, métricas e configurações históricos permanecem para permitir
reprodução e comparação. Não são apresentados como execução da revisão 0.5.
Dados completos, checkpoints, histórico de uso e capturas locais permanecem
fora do Git. As fontes e licenças estão registradas nos manifestos e em
`THIRD_PARTY_NOTICES.md`; uso acadêmico não implica autorização comercial.

## Checklist de envio

- [x] Integrantes e RM confirmados pelo responsável pelo projeto.
- [x] Código, preparação dos dados, treino, avaliação, API e interface disponíveis localmente.
- [x] Treinamento customizado e checkpoint sustentados por logs e hashes.
- [x] Métricas, classes com baixo desempenho e erros preservados.
- [x] Vídeo simples fornecido e verificado para envio como MP4 anexado.
- [ ] Conferir os três arquivos: PDF separado, MP4 e TXT com link do GitHub.
- [ ] Concluir a publicação pública do repositório e conferir acesso sem login.
- [ ] Manter os arquivos e o acesso ao repositório disponíveis durante a correção.
- [ ] Conferir o PDF final, os três integrantes/RM e os arquivos exigidos.
- [ ] Confirmar a turma e um integrante concluir o envio no Teams.

Prazo geral do enunciado: **19/11/2026 às 23h55**, Teams da disciplina.
A entrega é realizada por um integrante, identificando toda a equipe. O
responsável enviará os três arquivos separados. A publicação pública do
repositório foi autorizada e está em preparação; sua conclusão será registrada
após a verificação de acesso. O envio acadêmico ainda deve ser feito pela equipe.

## Contribuição aos ODS

O projeto pode apoiar inspeção preventiva e manutenção relacionadas aos ODS
3, 6, 11 e 13. Essas contribuições são potenciais: o detector não comprovou
redução de alagamentos, impacto sanitário ou benefício climático em operação.
