# DrenaVision — DrenaHub

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
Os vídeos de entrada e os resultados usam MP4 H.264 para reprodução no navegador;
o resultado mostra também um quadro de destaque com as marcações.

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
│   ├── raw/                downloads originais locais
│   └── processed/          imagens e rótulos prontos para o modelo
├── resultados/             tudo que o sistema produz
│   ├── metrics/            evidências oficiais de treino e avaliação
│   ├── demo/               exemplos anotados da avaliação
│   ├── dataset_preview/    amostras para conferir as anotações
│   └── local/              modelo, vídeos, histórico e saídas locais
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

## Guia rápido para o avaliador

| O que verificar | Onde |
|---|---|
| Produto funcionando | `iniciar.ps1` → foto/vídeo → resultado marcado → histórico |
| Código da interface e API | `drenavision/ui/` e `drenavision/api/` |
| Preparação e origem dos dados | `drenavision/data/` e `dados/manifests/` |
| Treinamento customizado | `ferramentas/configs/train.yaml` e `resultados/metrics/training_manifest.json` |
| Avaliação independente | `resultados/metrics/evaluation_metrics.json` e exemplos de erro |
| Requisitos e pendências FIAP | [ENTREGA.md](docs/ENTREGA.md) |

O modelo e os vídeos de demonstração são distribuídos na
[release v0.2.0-product](https://github.com/Torugo0/drenahub-drenavision/releases/tag/v0.2.0-product).
O repositório permanece privado nesta etapa; o acesso exige autorização na conta.

## Resultados reais e alcance

O treinamento usou 432 imagens: 308 para treino, 62 para validação e 62 para
teste. Foram concluídas 70 épocas, com seleção do melhor modelo na época 50.

| Resultado no teste separado | Valor |
|---|---:|
| Precisão | 0,4447 |
| Recall | 0,4796 |
| mAP@50 | 0,3781 |
| mAP@50–95 | 0,2196 |

Água é a principal limitação medida: recall 0,1159. As fontes públicas ainda
representam pouco os ralos, grelhas e garagens do uso pretendido. Os exemplos
facilitam o teste do produto; as métricas vêm exclusivamente da avaliação do
conjunto de teste. Detalhes, hashes e resultados por classe estão no
[MODEL_CARD.md](docs/MODEL_CARD.md).

A [validação dos oito exemplos externos](resultados/demo/product_validation.json)
processou e decodificou 1.388 quadros sem perda na exportação. A revisão visual
também mostrou falhas: dois clipes de roedores ficaram sem marcação, a marcação
animal de maior confiança em outro clipe apontou para uma telha e houve falso
alerta em espuma/água. As fotos de bueiros tiveram marcações válidas de resíduos,
mas parte dos resíduos e a água ficaram sem marcação. Portanto, os exemplos
demonstram a cadeia de análise e os erros do modelo; não garantem detecção em
qualquer filmagem.

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
