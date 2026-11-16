# DrenaVision — DrenaHub

Versão **0.3.0**: painel compacto em azul e detector v2 com anotações de roedores corrigidas.

## Ficha técnica do MVP

| Campo | Descrição |
|---|---|
| Produto | DrenaVision, módulo de inspeção visual da DrenaHub |
| Público | Síndicos, zeladores e equipes de manutenção de condomínios, garagens e áreas logísticas |
| Problema | Resíduos, água acumulada e animais podem passar despercebidos entre inspeções |
| Solução | Analisar fotos e vídeos, marcar os objetos encontrados e orientar a revisão do local |
| Tecnologia | YOLO26n com treinamento customizado, FastAPI, Streamlit e histórico SQLite |
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
└── docs/                   quatro guias e notebook
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
| `ferramentas/` | `scripts/` prepara exemplos/dados, audita/corrige anotações, treina, avalia e processa mídia; `configs/` contém parâmetros históricos v1 e os atuais `train_v2.yaml`/`evaluate_v2.yaml`; `tests/` reúne os testes automatizados. | Dados e resultados são gravados nos destinos de `dados/` ou `resultados/`, conforme a tarefa; não ficam misturados aos scripts. |
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
| Treinamento customizado atual | `ferramentas/configs/train_v2.yaml` e `resultados/metrics/v2/training_manifest.json` |
| Comparação no teste com rótulos corrigidos | `resultados/metrics/v2/comparison.json`, `candidate_test_metrics.json` e `baseline_test_metrics.json` |
| Histórico v1 | arquivos originais em `resultados/metrics/`; não usar suas caixas de animal como referência atual |
| Requisitos e pendências FIAP | [ENTREGA.md](docs/ENTREGA.md) |

O modelo e os vídeos de demonstração são distribuídos na
[release v0.3.0-model-v2](https://github.com/Torugo0/drenahub-drenavision/releases/tag/v0.3.0-model-v2).
O repositório permanece privado nesta etapa; o acesso exige autorização na conta.

## Resultados reais e alcance

O novo treinamento usou **557/62/62 imagens** de treino/validação/teste e
concluiu 60 épocas, com melhor checkpoint na época 35. A auditoria encontrou
305/447 caixas iRodent incorporando pontos ausentes `(0,0,0)`; a nova anotação
usa o envelope dos pontos visíveis com margem, preservando os dados antigos.

Comparação dos dois modelos nas **mesmas 62 imagens e caixas corrigidas v2**:

| Resultado no teste | Modelo v1 reavaliado | Modelo v2 ativo |
|---|---:|---:|
| Precisão | 0,2906 | 0,3828 |
| Recall | 0,2842 | 0,4251 |
| mAP@50 | 0,2563 | 0,3581 |
| mAP@50–95 | 0,1490 | 0,1810 |
| mAP@50 de animal | 0,0183 | 0,3102 |

Os números originais v1 continuam no histórico, mas usavam caixas de animal
defeituosas e não são comparáveis diretamente aos números acima. A caixa v2
pode representar apenas parte visível, sem garantia de corpo inteiro.
[MODEL_CARD.md](docs/MODEL_CARD.md) registra protocolo, hashes e resultados.

A [revisão dos oito exemplos no v2](resultados/metrics/v2/product_validation_v2.json)
processou os 1.388 quadros sem perda. O rato em close passou a ser marcado,
mas esse clipe forneceu frames de adaptação e **não é teste independente**.
Os falsos alertas sobre telha, carro e espuma não se repetiram; o rato na cena
com resíduos e o clipe difícil `rato_rua` ainda ficaram sem marcação. A foto de
lixo em Queens regrediu, enquanto a outra foto de bueiro manteve três embalagens
marcadas. Água e generalização continuam limitadas. Esses erros permanecem
documentados; a melhora não significa que toda filmagem terá detecção correta.

## Documentação e licença

| Documento | Finalidade |
|---|---|
| [TECNICO.md](docs/TECNICO.md) | arquitetura, endpoints, configuração e limites |
| [REPRODUZIR.md](docs/REPRODUZIR.md) | instalação, dados, treinamento, avaliação e testes |
| [MODEL_CARD.md](docs/MODEL_CARD.md) | ficha do modelo e evidências experimentais |
| [ENTREGA.md](docs/ENTREGA.md) | requisitos FIAP, roteiro e checklist de entrega |
| [Notebook Colab](docs/drenavision_training.ipynb) | execução do pipeline em notebook |

Código: [AGPL-3.0-only](LICENSE). As fontes de dados têm licenças próprias;
iRodent é tratado conservadoramente como CC BY-NC. Créditos e restrições estão
em [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
