# DrenaVision — DrenaHub

## [Abrir documentação técnica (PDF)](docs/relatorio/DrenaVision_Applied_Computer_Vision_FIAP.pdf)

Versão **0.5.0**: inferência ampliada com propostas Grounding DINO e verificação
de recortes SigLIP 2, mantendo o YOLO customizado v3 e a análise de água B1.
Não houve novo treinamento nesta revisão. **A API 0.5.0 está ativa e a
verificação funcional de fotos/vídeos está concluída.** A suíte automatizada
passou em 239 testes. Isso não elimina os erros de detecção descritos abaixo.

## Integrantes

| Aluno | RM |
|---|---|
| Gabriel Machado Carrara Pimentel | RM99880 |
| Lourenzo Ramos | RM99951 |
| Vítor Hugo Rodrigues | RM97758 |

## Relatório da entrega

O [relatório técnico em PDF](docs/relatorio/DrenaVision_Applied_Computer_Vision_FIAP.pdf)
explica o código, os dados, o treinamento customizado, a inferência composta,
a integração e os resultados reais. A [fonte editável](docs/relatorio/relatorio_fonte.md)
e o [checklist dos requisitos](docs/relatorio/checklist_requisitos.md) acompanham
cada seção exigida na disciplina. O documento distingue o YOLO treinado dos
componentes pré-treinados e mantém as falhas observadas.

O vídeo de demonstração pode se concentrar no uso do produto. O enunciado de
Applied Computer Vision não exige percorrer o código na gravação nem fixa
duração mínima ou máxima; a explicação técnica está no relatório e neste repositório.

## Ficha técnica do MVP

| Campo | Descrição |
|---|---|
| Produto | DrenaVision, módulo de inspeção visual da DrenaHub |
| Público | Síndicos, zeladores e equipes de manutenção de condomínios, garagens e áreas logísticas |
| Problema | Resíduos, água acumulada e animais podem passar despercebidos entre inspeções |
| Solução | Analisar fotos e vídeos, marcar os objetos encontrados e orientar a revisão do local |
| Tecnologia | YOLO26n customizado v3, Grounding DINO Tiny, verificador SigLIP 2, SegFormer B1 para água, FastAPI, Streamlit e histórico SQLite |
| ODS | 3, 6, 11 e 13: apoio à inspeção preventiva e à manutenção da drenagem |
| Entrega | Três arquivos para o professor: PDF separado, vídeo MP4 anexado e TXT com o link do GitHub; código, documentação e PDF publicados neste repositório público |

## Abrir e usar

No Windows, execute na pasta do projeto:

```powershell
.\iniciar.ps1
```

O iniciador verifica o ambiente e evita iniciar outra cópia dos serviços nas
mesmas portas. Acesse a [interface](http://127.0.0.1:8501).
Após atualizar o código, use `.\iniciar.ps1 -Reiniciar` para recarregar os
serviços que já estavam abertos.

Se a porta 8501 estiver em uso por outro aplicativo, execute
`.\iniciar.ps1 -PortaInterface 8502` e abra `http://127.0.0.1:8502`.
O iniciador não encerra serviços de outros projetos.

1. Envie uma **foto ou vídeo**, ou escolha um exemplo disponível na tela.
2. Informe o nome do local, se desejar, e inicie a análise.
3. Veja o resultado com marcações e os avisos do que merece inspeção.
4. Baixe o resultado ou consulte as análises anteriores no histórico.

Há **seis filmagens reais e duas fotografias** de roedores, resíduos e drenagem
em [dados/examples/](dados/examples/README.md), com origem e licença por arquivo.
Os vídeos usam MP4 H.264. O painel organiza envio e resultado lado a lado no
desktop e limita a altura da mídia; informações técnicas ficam recolhidas.
Fotos, vídeos e histórico passam pela API. As redes grandes são compartilhadas
nesse serviço; a interface não inicia outro processo com uma cópia dos modelos.
Os seis vídeos e duas fotos do catálogo concluíram a análise; os **1.388 quadros**
dos MP4 anotados foram decodificados integralmente, com H.264, início rápido e
entrega parcial HTTP verificados. Na tela, as duas fotos relatadas pelo usuário
foram enviadas e o vídeo Hanoi foi reproduzido até 12/12 segundos. Os registros
estão em [product_validation.json](resultados/metrics/v4/product_validation.json)
e [ui_verification.json](resultados/metrics/v4/ui_verification.json).

Os alertas são indicações visuais para uma pessoa verificar. A ausência de
alerta não comprova que o local está seguro. A revisão de animais concentra-se
em roedores; pode perder objetos ou marcar algo errado. Os componentes
externos são pré-treinados, sem ajuste dos seus pesos nesta revisão.

Na primeira instalação, prepare o ambiente e obtenha o modelo conforme
[REPRODUZIR.md](docs/REPRODUZIR.md#instalar-o-ambiente). A
[documentação da API](http://127.0.0.1:8000/docs) é opcional para quem deseja
integrar outros sistemas.

## Vídeo demonstrativo

A equipe forneceu `demo_drena_vision.mp4`, com **2min13,20s**, em Full HD a
30 FPS. A revisão local confirmou a demonstração do envio de fotos/vídeos,
marcações de animal, resíduos e água, além dos alertas. O arquivo foi
integralmente decodificado sem erros, mas sua trilha de áudio é silenciosa.
O vídeo será enviado **como arquivo MP4 anexado**, junto ao PDF separado e
a um TXT com o link do GitHub. A página 44 de Applied Computer Vision pede
um vídeo da solução funcionando e não determina plataforma de hospedagem ou
URL pública para ele. O relatório documenta a parte técnica; uma narração ou
legenda curta pode contextualizar a demonstração sem percorrer o código.

### Arquivos para enviar ao professor

| Arquivo | Conteúdo |
|---|---|
| `DrenaVision_Applied_Computer_Vision_FIAP.pdf` | Relatório técnico, também disponível em `docs/relatorio/` no GitHub. |
| `demo_drena_vision.mp4` | Demonstração da aplicação gravada pela equipe. |
| TXT com o link do GitHub | URL `https://github.com/Torugo0/drenahub-drenavision`, para consultar código e documentação. |

A gravação narrada anterior de **9min12,46s** registra a **versão 0.4.0** e
permanece preservada como histórico em `resultados/local/gravacao/`.
[GRAVACAO.md](docs/GRAVACAO.md) mantém método, créditos, arquivos e capítulos;
[ROTEIRO_GRAVACAO.md](docs/ROTEIRO_GRAVACAO.md) preserva o roteiro correspondente.
Essa gravação antiga não comprova o comportamento da versão 0.5.

Antes do envio, a equipe deve abrir o PDF, reproduzir o MP4, conferir o TXT e
concluir a entrega na tarefa da disciplina. O repositório, o PDF e os assets
da release foram conferidos sem autenticação. O PDF baixado do GitHub tem
o mesmo hash da cópia anexada. O envio acadêmico continua a cargo da equipe.

## Organização das pastas

```text
drena_hub_avc/
├── iniciar.ps1             abre o produto e verifica os serviços locais
├── drenavision/            código da aplicação
│   ├── api/                comunicação com outros sistemas
│   ├── core/               configuração, contratos e histórico
│   ├── data/               preparação dos dados de treinamento
│   ├── inference/          modelo, análise e processamento dos vídeos
│   └── ui/                 telas usadas pelo operador
├── dados/                  entradas e origem dos dados
│   ├── examples/           exemplos de foto/vídeo e seus créditos
│   ├── manifests/          fontes, licenças, divisões e hashes
│   ├── raw/                downloads originais (local; fora do Git)
│   └── processed/          imagens e rótulos do modelo (local; fora do Git)
├── resultados/             tudo que o sistema produz
│   ├── metrics/            evidências oficiais de treino e avaliação
│   ├── demo/               exemplos anotados da avaliação
│   ├── dataset_preview/    amostras para conferir as anotações
│   └── local/              modelo, vídeos e histórico (fora do Git)
├── ferramentas/            recursos para desenvolver e reproduzir
│   ├── scripts/            comandos de dados, treino e avaliação
│   ├── configs/            parâmetros do treinamento e da avaliação
│   └── tests/              testes automatizados
└── docs/                   documentação e evidências para o avaliador
    └── relatorio/          PDF final, fonte editável e checklist da disciplina
```

Para usar o produto, comece por `iniciar.ps1`. Para avaliar o código, comece por
`drenavision/`. Para conferir o trabalho acadêmico, abra `resultados/metrics/`
e `docs/`. Dados completos, pesos, histórico e saídas grandes ficam locais;
fontes, configurações, resultados selecionados e documentação ficam no Git.
Veja [dados/README.md](dados/README.md) e
[resultados/README.md](resultados/README.md) para os detalhes internos.

A revisão preservou código, testes, configurações e métricas que permitem
reproduzir o projeto. Datasets completos, modelos, banco de eventos, logs,
caches e saídas de trabalho ficam fora do Git. As fotos pessoais de regressão
`teste.jpg` e `teste1.jpg` também ficam locais: não integram o catálogo com
origem/licença registrada nem a entrega pública.

### O que cada pasta contém no Git

Esta tabela descreve os arquivos publicados no repositório. As pastas marcadas
como locais na árvore acima aparecem depois da preparação ou do uso e não são
enviadas ao GitHub.

| Pasta | Conteúdo versionado no Git | Conteúdo que fica somente no computador |
|---|---|---|
| `drenavision/` | Código Python: `api/` define os endpoints; `core/` contém configuração, contratos e acesso ao histórico; `data/` prepara os datasets; `inference/` executa treino, avaliação, detecção e conversão de vídeo; `ui/` monta a interface. | Caches do Python. O banco e o modelo usados pelo código ficam em `resultados/local/`. |
| `dados/` | `dataset.yaml` define as classes; `dataset_sources.csv` registra as fontes; `manifests/` guarda origem, divisão treino/validação/teste, hashes e relatórios; `examples/` contém seis vídeos reais, duas fotos, capas e seus créditos em `sources.json`. | `raw/`, `processed/`, `cache/` e `interim/`: downloads completos, dados preparados e arquivos de trabalho. |
| `resultados/` | `metrics/` preserva métricas reais, curvas, matrizes e exemplos de acertos/erros; `demo/` contém imagens anotadas, a captura da interface e a validação dos exemplos; `dataset_preview/` mostra uma montagem das anotações. | `local/`: pesos do modelo, vídeos analisados, banco SQLite, evidências de uso, logs, caches e execuções completas; `local/gravacao/` recebe a gravação, legendas, transcrição e manifesto. |
| `ferramentas/` | `scripts/` prepara exemplos/dados/modelos, audita anotações, treina, calibra e avalia; `configs/` contém `inference_v4.json`, o treino/avaliação YOLO v3 e os protocolos históricos; `tests/` reúne os testes automatizados. | Dados e resultados são gravados nos destinos de `dados/` ou `resultados/`, conforme a tarefa; não ficam misturados aos scripts. |
| `docs/` | `relatorio/` reúne o PDF da disciplina, a fonte editável e o checklist; `TECNICO.md` explica arquitetura e API; `REPRODUZIR.md` ensina a instalar e repetir o pipeline; `MODEL_CARD.md` registra resultados e limites do modelo; `ENTREGA.md` organiza a avaliação FIAP; `ROTEIRO_GRAVACAO.md` orienta a demonstração e `GRAVACAO.md` registra método, créditos e publicação; o notebook permite executar o pipeline no Colab. | Nenhum modelo, dataset completo, gravação ou histórico de uso é armazenado aqui. |

Na raiz também ficam versionados `iniciar.ps1` (inicialização), `requirements*.txt`
e `pyproject.toml` (dependências e configuração Python), `.env.example` (modelo
de configuração), `.streamlit/config.toml` (aparência e limite de upload),
`.gitignore`/`.gitattributes` (regras do Git), `LICENSE`, `CITATION.cff` e
`THIRD_PARTY_NOTICES.md` (licença, citação e créditos). O ambiente `.venv/`, o
arquivo pessoal `.env` e os caches não são versionados. O checkpoint `best.pt`
é distribuído pela release, separadamente do histórico de código do Git.

## Guia rápido para o avaliador

| O que verificar | Onde |
|---|---|
| Relatório oficial, código explicado e requisitos | [PDF](docs/relatorio/DrenaVision_Applied_Computer_Vision_FIAP.pdf), [fonte Markdown](docs/relatorio/relatorio_fonte.md) e [checklist](docs/relatorio/checklist_requisitos.md) |
| Produto funcionando | `iniciar.ps1` → foto/vídeo → resultado marcado → histórico |
| Código da interface e API | `drenavision/ui/` e `drenavision/api/` |
| Preparação e origem dos dados | `drenavision/data/` e `dados/manifests/` |
| Treinamento customizado mantido | `ferramentas/configs/train_v3.yaml` e `resultados/metrics/v3/`; a revisão 0.5 muda a inferência, não repete o treino |
| Perfil de inferência 0.5 | `ferramentas/configs/inference_v4.json`, `resultados/metrics/v4/frozen_candidate.json` e `active_profile.json` |
| Avaliação em 40 imagens reservadas e 179 regressões conhecidas | `resultados/metrics/v4/evaluation_holdout40.json` e `evaluation_regression179.json`; coortes separadas |
| Verificação funcional de catálogo e interface 0.5 | `resultados/metrics/v4/product_validation.json` e `ui_verification.json` |
| Escolha dos parâmetros e separação dos dados | [VALIDACAO_V4.md](docs/VALIDACAO_V4.md); o protocolo de água v3 permanece preservado |
| Histórico v1/v2 | arquivos originais em `resultados/metrics/` e `metrics/v2/`; as referências de água e animal mudaram entre protocolos |
| Requisitos e pendências FIAP | [ENTREGA.md](docs/ENTREGA.md) |
| Vídeo local histórico 0.4, créditos e publicação | [GRAVACAO.md](docs/GRAVACAO.md); MP4 em `resultados/local/gravacao/` |

O YOLO customizado v3, o modelo de água e os exemplos continuam distribuídos na
[release v0.4.0-pipeline-v3](https://github.com/Torugo0/drenahub-drenavision/releases/tag/v0.4.0-pipeline-v3).
Os dois componentes adicionais vêm de preparadores com origem oficial, revisão
e hashes fixados: `prepare_object_model.py` e `prepare_verifier_model.py`.
Eles já estão presentes neste computador; em uma instalação nova, o download
adicional é de aproximadamente **2,2 GB**. A API não baixa pesos durante a análise.
O repositório é público. Código, PDF e assets da release foram conferidos
sem login; os modelos auxiliares permanecem disponíveis por seus preparadores.

## Resultados reais e alcance

O perfil 0.5 congelado é `pipeline-4c5c993223219929`. A coorte de **40 imagens
novas CLICK-test** e a revisão das referências foram seladas antes da seleção
final dos parâmetros. O perfil foi congelado antes da inferência nessa coorte.
A presença de resíduos passou de **38/40 para 39/40**. Nas 60 caixas nativas,
com casamento IoU 0,50, as perdas caíram de **17 para 12** e o recall passou de
0,7167 para 0,8000; a precisão caiu de **0,7288 para 0,7059**. Houve mais
cobertura e também mais previsões sem correspondência, de 16 para 20.

Não houve falso alerta de animal nas 38 imagens negativas claras, nos dois
perfis; duas incertas foram excluídas. A água manteve o mesmo resultado nas
40 imagens, que não são uma nova avaliação de acurácia de água. A latência
mediana por foto na RTX 4060 subiu de **0,615 s para 0,965 s** nesse teste.
Esse conjunto não tem negativos de resíduos nem positivos de animais e vem
da mesma fonte/câmera conhecida: seus números não demonstram acerto universal
ou generalização para qualquer local.

Nas **179 regressões conhecidas**, a presença de animal subiu de **14/20 para
19/20** no teste v3 e de **46/69 para 63/69** no conjunto adicional, com zero
alertas nas negativas por proxy de rótulo. Porém, o F1 de caixas de animal com
IoU 0,50 **caiu de 0,4706 para 0,3913 e de 0,6207 para 0,6014**, respectivamente.
Há maior cobertura de presença e pior acordo de localização, incluindo caixas
duplicadas ou amplas. Referências de pose parciais limitam a medida, mas não
justificam atribuir todos os erros aos rótulos. A água ficou igual nas 179.
O benchmark reúne **40 imagens reservadas + 179 regressões conhecidas**;
seus resultados permanecem separados.

A verificação funcional dos vídeos concluiu sem falha de reprodução ou
persistência, mas **não comprova melhora temporal da detecção**. Em Hanoi,
animal foi marcado em 144/240 quadros, contra 187/240 na 0.4. Em Rua houve
uma marcação ambígua em 240 quadros; esse caso continua sem acerto confirmado.
A revisão encontrou falsos resíduos na lixeira fixa de Chuva e no PVC instalado
de Sarjeta, além de ambiguidades em vegetação. `ralo_lixo` ainda perde outros
resíduos e água rasa; em Canal, a região de água inclui concreto. As filmagens
são regressões conhecidas e algumas forneceram quadros de adaptação ao treino;
suas contagens não são recall ou número de animais físicos. O
[perfil ativo](resultados/metrics/v4/active_profile.json) reúne as provas e limites.

Na validação usada para escolher os parâmetros, os animais sinalizados
passaram de 15/20 para 20/20, com zero falsos alertas em 40 negativas. Isso é
**calibração**, não uma taxa geral de detecção. Nas duas fotos conhecidas do
usuário, a reexecução congelada produziu uma marcação de animal e cinco de
resíduos; elas continuam regressões de desenvolvimento, fora do treino e da
calibração. [VALIDACAO_V4.md](docs/VALIDACAO_V4.md) explica os denominadores,
a revisão assistida dos rótulos e o custo da maior cobertura.

### Histórico do treino e do produto 0.4.0

O YOLO v3 concluiu **63 épocas**, com melhor checkpoint na época **43**, usando
**1.107/60/60 imagens** de treino/validação/teste. O treino incorporou contexto
real do rato, 400 imagens positivas adicionais de água e 150 de resíduos.
Caixas fragmentadas da mesma poça foram agrupadas; negativos que continham
água foram excluídos após revisão dos originais. As bases anteriores permanecem
preservadas.

A água também recebe análise de superfície por um SegFormer B1 pré-treinado.
As áreas azuis vêm dos pixels previstos pelo modelo. O nome técnico
`agua_acumulada` é mantido na API, mas representa **água visível, parada ou em
movimento**; não mede profundidade nem comprova risco ou estagnação.

O produto **0.4.0** foi comparado ao v2 em **60 imagens de regressão e 119 imagens
adicionais que não entraram no ajuste local**. Nas 40 imagens adicionais com
água, a identificação por presença passou de **2 para 33**. Houve **dois falsos
alertas em dez negativos revisados**. A presença de animal ficou em **46/69**,
contra 47/69 do v2: os acertos nos vídeos conhecidos não comprovam melhora
universal para animais. O relatório mantém essa regressão e os erros.

Na avaliação acadêmica do **YOLO isolado**, o mAP@50 passou de 0,3349 para
0,4845 nas mesmas caixas v3. Esse valor não é mAP do produto composto.
[MODEL_CARD.md](docs/MODEL_CARD.md) separa treinamento, calibração, testes e
reprodução de vídeo. [VALIDACAO_V3.md](docs/VALIDACAO_V3.md) explica as referências
parciais e os limites das conclusões.

## Documentação e licença

| Documento | Finalidade |
|---|---|
| [Relatório oficial](docs/relatorio/DrenaVision_Applied_Computer_Vision_FIAP.pdf) | documento acadêmico com identificação, metodologia, métricas, exemplos e conclusão |
| [Fonte do relatório](docs/relatorio/relatorio_fonte.md) | conteúdo editável, referências e evidências da entrega |
| [Checklist do relatório](docs/relatorio/checklist_requisitos.md) | correspondência entre requisitos e páginas do PDF |
| [TECNICO.md](docs/TECNICO.md) | arquitetura, endpoints, configuração e limites |
| [REPRODUZIR.md](docs/REPRODUZIR.md) | instalação, dados, treinamento, avaliação e testes |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | ficha do modelo e evidências experimentais |
| [VALIDACAO_V3.md](docs/VALIDACAO_V3.md) | revisão das anotações, calibração e teste adicional |
| [VALIDACAO_V4.md](docs/VALIDACAO_V4.md) | fusão 0.5, perfis congelados, novos testes e limites dos dados |
| [ENTREGA.md](docs/ENTREGA.md) | requisitos FIAP, roteiro e checklist de entrega |
| [ROTEIRO_GRAVACAO.md](docs/ROTEIRO_GRAVACAO.md) | sequência completa dos exemplos, falas e resumo opcional |
| [GRAVACAO.md](docs/GRAVACAO.md) | estado da gravação local, método, créditos e como publicar |
| [Notebook Colab](docs/drenavision_training.ipynb) | execução do pipeline em notebook |

Código: [AGPL-3.0-only](LICENSE). As fontes de dados têm licenças próprias;
iRodent é tratado conservadoramente como CC BY-NC; o segmentador NVIDIA tem
licença de pesquisa/avaliação não comercial. A entrega é um protótipo acadêmico,
sem autorização implícita de uso comercial. Créditos e restrições estão
em [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
