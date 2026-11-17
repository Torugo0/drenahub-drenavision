# DrenaVision — DrenaHub

Versão **0.4.0**: detector v3 retreinado, análise própria de superfícies de água e painel compacto em azul.

## Ficha técnica do MVP

| Campo | Descrição |
|---|---|
| Produto | DrenaVision, módulo de inspeção visual da DrenaHub |
| Público | Síndicos, zeladores e equipes de manutenção de condomínios, garagens e áreas logísticas |
| Problema | Resíduos, água acumulada e animais podem passar despercebidos entre inspeções |
| Solução | Analisar fotos e vídeos, marcar os objetos encontrados e orientar a revisão do local |
| Tecnologia | YOLO26n customizado, SegFormer B1 para água, FastAPI, Streamlit e histórico SQLite |
| ODS | 3, 6, 11 e 13: apoio à inspeção preventiva e à manutenção da drenagem |
| Entrega | Código, modelo treinado, métricas reais e exemplos; vídeo narrado, PDF e publicação pública pendentes |

## Abrir e usar

No Windows, execute na pasta do projeto:

```powershell
.\iniciar.ps1
```

O iniciador verifica o ambiente e evita iniciar outra cópia dos serviços nas
mesmas portas. Acesse a [interface](http://127.0.0.1:8501).
Após atualizar o código, use `.\iniciar.ps1 -Reiniciar` para recarregar os
serviços que já estavam abertos.

1. Envie uma **foto ou vídeo**, ou escolha um exemplo disponível na tela.
2. Informe o nome do local, se desejar, e inicie a análise.
3. Veja o resultado com marcações e os avisos do que merece inspeção.
4. Baixe o resultado ou consulte as análises anteriores no histórico.

Há **seis filmagens reais e duas fotografias** de roedores, resíduos e drenagem
em [dados/examples/](dados/examples/README.md), com origem e licença por arquivo.
Os vídeos usam MP4 H.264. O painel organiza envio e resultado lado a lado no
desktop e limita a altura da mídia; informações técnicas ficam recolhidas.

Os alertas são indicações visuais para uma pessoa verificar. A ausência de
alerta não comprova que o local está seguro. A classe de animais foi treinada
principalmente com roedores; o modelo pode perder objetos ou marcar algo errado.

Na primeira instalação, prepare o ambiente e obtenha o modelo conforme
[REPRODUZIR.md](docs/REPRODUZIR.md#instalação). A
[documentação da API](http://127.0.0.1:8000/docs) é opcional para quem deseja
integrar outros sistemas.

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
└── docs/                   guias técnicos, protocolo de validação e notebook
```

Para usar o produto, comece por `iniciar.ps1`. Para avaliar o código, comece por
`drenavision/`. Para conferir o trabalho acadêmico, abra `resultados/metrics/`
e `docs/`. Dados completos, pesos, histórico e saídas grandes ficam locais;
fontes, configurações, resultados selecionados e documentação ficam no Git.
Veja [dados/README.md](dados/README.md) e
[resultados/README.md](resultados/README.md) para os detalhes internos.

### O que cada pasta contém no Git

Esta tabela descreve os arquivos publicados no repositório. As pastas marcadas
como locais na árvore acima aparecem depois da preparação ou do uso e não são
enviadas ao GitHub.

| Pasta | Conteúdo versionado no Git | Conteúdo que fica somente no computador |
|---|---|---|
| `drenavision/` | Código Python: `api/` define os endpoints; `core/` contém configuração, contratos e acesso ao histórico; `data/` prepara os datasets; `inference/` executa treino, avaliação, detecção e conversão de vídeo; `ui/` monta a interface. | Caches do Python. O banco e o modelo usados pelo código ficam em `resultados/local/`. |
| `dados/` | `dataset.yaml` define as classes; `dataset_sources.csv` registra as fontes; `manifests/` guarda origem, divisão treino/validação/teste, hashes e relatórios; `examples/` contém seis vídeos reais, duas fotos, capas e seus créditos em `sources.json`. | `raw/`, `processed/`, `cache/` e `interim/`: downloads completos, dados preparados e arquivos de trabalho. |
| `resultados/` | `metrics/` preserva métricas reais, curvas, matrizes e exemplos de acertos/erros; `demo/` contém imagens anotadas, a captura da interface e a validação dos exemplos; `dataset_preview/` mostra uma montagem das anotações. | `local/`: pesos do modelo, vídeos analisados, banco SQLite, evidências de uso, logs, caches e execuções completas. |
| `ferramentas/` | `scripts/` prepara exemplos/dados/modelos, audita anotações, treina, calibra e avalia; `configs/` preserva os protocolos antigos e os atuais `train_v3.yaml`/`evaluate_v3.yaml`; `tests/` reúne os testes automatizados. | Dados e resultados são gravados nos destinos de `dados/` ou `resultados/`, conforme a tarefa; não ficam misturados aos scripts. |
| `docs/` | `TECNICO.md` explica arquitetura e API; `REPRODUZIR.md` ensina a instalar e repetir o pipeline; `MODEL_CARD.md` registra resultados e limites do modelo; `ENTREGA.md` organiza a avaliação FIAP; o notebook permite executar o pipeline no Colab. | Nenhum modelo, dataset completo ou histórico de uso é armazenado aqui. |

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
| Produto funcionando | `iniciar.ps1` → foto/vídeo → resultado marcado → histórico |
| Código da interface e API | `drenavision/ui/` e `drenavision/api/` |
| Preparação e origem dos dados | `drenavision/data/` e `dados/manifests/` |
| Treinamento customizado atual | `ferramentas/configs/train_v3.yaml` e `resultados/metrics/v3/` |
| Produto completo em imagens novas | `resultados/metrics/v3/independent_comparison.json` |
| Escolha dos parâmetros e separação dos dados | [VALIDACAO_V3.md](docs/VALIDACAO_V3.md) e `resultados/metrics/v3/active_profile.json` |
| Histórico v1/v2 | arquivos originais em `resultados/metrics/` e `metrics/v2/`; as referências de água e animal mudaram entre protocolos |
| Requisitos e pendências FIAP | [ENTREGA.md](docs/ENTREGA.md) |

O modelo e os vídeos de demonstração são distribuídos na
[release v0.4.0-pipeline-v3](https://github.com/Torugo0/drenahub-drenavision/releases/tag/v0.4.0-pipeline-v3).
O repositório permanece privado nesta etapa; o acesso exige autorização na conta.

## Resultados reais e alcance

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

O produto foi comparado ao v2 em **60 imagens de regressão e 119 imagens
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
| [TECNICO.md](docs/TECNICO.md) | arquitetura, endpoints, configuração e limites |
| [REPRODUZIR.md](docs/REPRODUZIR.md) | instalação, dados, treinamento, avaliação e testes |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | ficha do modelo e evidências experimentais |
| [VALIDACAO_V3.md](docs/VALIDACAO_V3.md) | revisão das anotações, calibração e teste adicional |
| [ENTREGA.md](docs/ENTREGA.md) | requisitos FIAP, roteiro e checklist de entrega |
| [Notebook Colab](docs/drenavision_training.ipynb) | execução do pipeline em notebook |

Código: [AGPL-3.0-only](LICENSE). As fontes de dados têm licenças próprias;
iRodent é tratado conservadoramente como CC BY-NC; o segmentador NVIDIA tem
licença de pesquisa/avaliação não comercial. A entrega é um protótipo acadêmico,
sem autorização implícita de uso comercial. Créditos e restrições estão
em [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
