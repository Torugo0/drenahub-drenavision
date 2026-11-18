# Avisos de terceiros

Este arquivo resume dependências e dados diretamente adotados pelo DrenaVision.
Ele não substitui os textos de licença incluídos nas distribuições originais
nem uma análise jurídica. Antes de redistribuir binários, pesos ou datasets,
gere um inventário completo de dependências transitivas e preserve avisos.

## Detector, runtime e aplicação

| Componente | Versão de referência | Licença | Fonte |
|---|---:|---|---|
| Ultralytics | 8.4.215 | AGPL-3.0 por padrão ou licença Enterprise | [licenciamento](https://www.ultralytics.com/license) |
| Pesos YOLO distribuídos pela Ultralytics | conforme config | seguem a opção de licença aplicável da Ultralytics | [modelos](https://docs.ultralytics.com/models/) |
| PyTorch | 2.14.1 | BSD-style | [repositório](https://github.com/pytorch/pytorch) |
| torchvision | 0.29.1 | BSD-3-Clause | [repositório](https://github.com/pytorch/vision) |
| OpenCV | 4.14.0.94 | Apache-2.0 | [licença](https://github.com/opencv/opencv/blob/4.x/LICENSE) |
| FastAPI | 0.163.1 | MIT | [repositório](https://github.com/fastapi/fastapi) |
| faster-coco-eval | 1.8.0 | Apache-2.0 | [repositório](https://github.com/MiXaiLL76/faster_coco_eval) |
| lap | 0.5.14 | BSD-2-Clause | [repositório e licença](https://github.com/gatagat/lap/blob/v0.5.14/LICENSE) |
| Pydantic / pydantic-settings | 2.14.3 / 2.16.3 | MIT | [Pydantic](https://github.com/pydantic/pydantic) |
| Streamlit | 1.67.0 | Apache-2.0 | [repositório](https://github.com/streamlit/streamlit) |
| NumPy | 2.2.6 | BSD-3-Clause | [licença](https://github.com/numpy/numpy/blob/main/LICENSE.txt) |
| Pillow | 12.0.0 | HPND | [licença](https://github.com/python-pillow/Pillow/blob/main/LICENSE) |
| HTTPX | 0.28.1 | BSD-3-Clause | [repositório](https://github.com/encode/httpx) |
| Uvicorn | 0.54.0 | BSD-3-Clause | [repositório](https://github.com/encode/uvicorn) |
| PyYAML | 6.0.3 | MIT | [repositório](https://github.com/yaml/pyyaml) |
| python-multipart | 0.0.34 | Apache-2.0 | [repositório](https://github.com/Kludex/python-multipart) |
| remotezip | 0.12.3 | MIT | [repositório](https://github.com/gtsystem/python-remotezip) |
| Transformers | 4.57.3 | Apache-2.0 | [repositório](https://github.com/huggingface/transformers) |
| Safetensors | 0.7.0 | Apache-2.0 | [repositório](https://github.com/huggingface/safetensors) |

Dependências transitivas devem ser conferidas no ambiente efetivamente
distribuído. Execute `python -m pip freeze` e use uma ferramenta de SBOM/licença
antes de publicação produtiva.

## Implicação da Ultralytics

Este repositório adota AGPL-3.0-only para permanecer compatível com a opção
open-source padrão da Ultralytics. A AGPL contém obrigações para software
modificado oferecido por rede. Uma futura DrenaHub proprietária, interna ou
SaaS não deve presumir que o uso é liberado: deve obter aconselhamento jurídico,
avaliar a licença Enterprise ou substituir o componente por alternativa com
licença compatível.

Treinar pesos não elimina automaticamente obrigações associadas ao software e
pesos de partida. Consulte os termos vigentes no momento da distribuição.

## Datasets

As quantidades abaixo distinguem a seleção inicial histórica do dataset
customizado v3, mantido na aplicação 0.5.0. O dataset v3 preparado contém
**1.107/60/60 imagens em treino/validação/teste**: no treino são 250 CLICK,
500 água, 333 iRodent e 24 imagens/cortes de adaptação do domínio; validação
e teste têm 20 imagens de cada uma das três fontes principais.
Os cortes, quadros e imagens de uma fonte não são locais independentes.
As contagens e origens autoritativas são o manifesto v3 e a
[auditoria v4](resultados/metrics/v4/data_audit_summary.json).
O conjunto adicional de 119 imagens e as 40 fotos CLICK reservadas na revisão
v4 têm papéis de avaliação separados e não foram adicionados ao treino v3.

### CLICK Trash Dataset v2

- título: **CLICK Trash Dataset v2**;
- página: [Mendeley Data, versão 2](https://data.mendeley.com/datasets/htcvvz6fm2/2);
- licença declarada: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
- uso histórico inicial: amostra determinística de 100/20/20 imagens por
  train/val/test, obtida do ZIP remoto por HTTP Range;
- dataset v3 efetivamente treinado: 250/20/20 imagens; a revisão v4 avaliou
  separadamente 40 novas fotos do teste nativo, com manifesto e revisão selados;
- transformação: classe única de lixo mapeada para `obstrucao_visivel`;
- obrigação: atribuição, link da licença e indicação das modificações.

### Dataset of Stagnant Water and Wet Surface v4

- página: [Mendeley Data, versão 4](https://data.mendeley.com/datasets/y6zyrnxbfm/4);
- licença declarada: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
- uso no projeto: somente a classe source `water`, mapeada para
  `agua_acumulada`; superfícies apenas molhadas são excluídas quando separáveis;
- dataset v3 efetivamente treinado: 500/20/20 imagens preparadas por
  train/val/test, sem considerar esses arquivos como 540 locais independentes;
- obrigação: atribuição, link da licença e indicação das modificações.

### iRodent: rodents in the wild

- registro: [Zenodo 8250392](https://doi.org/10.5281/zenodo.8250392);
- anotação: publicação com 447 registros de imagem em COCO para roedores em
  ambiente real; a seleção inicial foi de 100/20/20 imagens utilizáveis por
  train/val/test, enquanto o v3 preparado contém 333/20/20;
- transformação: algumas caixas foram derivadas de pontos corporais visíveis;
  esses envelopes podem representar partes do corpo, sem garantir limites
  completos do animal. Essa limitação é preservada na avaliação;
- conflito de licença: o registro traz referência Apache-2.0, enquanto a
  documentação do conjunto atribui às imagens-fonte condição CC BY-NC;
- tratamento adotado: **CC BY-NC 4.0**, a interpretação mais restritiva;
- consequência: não presumir permissão de uso comercial e não redistribuir em
  produto comercial sem autorização/revisão adicional.

O manifesto gerado registra fonte, versão, URL, licença e hash por imagem. O
dataset bruto/completo não é versionado. Somente amostras transformadas para
auditoria e demonstração acadêmica aparecem em `resultados/dataset_preview/`,
`resultados/demo/` e `resultados/metrics/error_examples/`, com identificador,
atribuição, licença e alterações descritos nos respectivos arquivos `README.md`
e no manifesto.

## Atribuição em materiais de entrega

O relatório e o vídeo devem citar os três conjuntos, suas versões e licenças.
Se uma imagem específica for exibida, mantenha o identificador no manifesto e
inclua atribuição compatível. Transformações como conversão para YOLO, seleção
de classes, mudança de nome e divisão de split devem ser informadas.

## Segmentação de água — NVIDIA SegFormer B1

Os perfis 0.4.0 e 0.5.0 usam [nvidia/segformer-b1-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b1-finetuned-ade-512-512),
revisão `a775b4f5dcb2179e57006dca0e268281306412ef`, treinado em ADE20K.
Crédito: NVIDIA e autores de SegFormer/ADE20K, conforme o model card oficial.
Pesos e código de origem têm a [NVIDIA Source Code License](https://github.com/NVlabs/SegFormer/blob/master/LICENSE),
limitada à pesquisa e avaliação não comercial, com condições de redistribuição.
Não são MIT nem relicenciados pelo AGPL do aplicativo. A licença acompanha a
pasta/ZIP de cinco arquivos do modelo e está preservada com seus bytes originais
em `resultados/metrics/v3/water_model_source/`.

O binário oficial é verificado por SHA-256, lido com `weights_only=True` e
convertido para safetensors. A aplicação usa apenas safetensors local e
configuração verificados. A seleção de seis nomes de classes aquáticas,
limiares, componentes conectados e máscara azul são adaptações do DrenaVision.
O projeto foi validado como protótipo acadêmico; nenhum documento autoriza
comercialização sem resolver essa licença e as restrições dos demais dados.

## Propostas de objetos - Grounding DINO Tiny

A revisão de inferência 0.5.0 utiliza o checkpoint oficial
[IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny),
revisão `a2bb814dd30d776dcf7e30523b00659f4f141c71`, sob **Apache-2.0**.
Não houve ajuste dos pesos desse componente no projeto. O vocabulário fixo,
os limiares, o mapeamento de frases para as classes e a fusão são adaptações
do DrenaVision, registradas em
[frozen_candidate.json](resultados/metrics/v4/frozen_candidate.json).

Pesos safetensors, configuração, tokenizador, processador e licença são
obtidos separadamente por `prepare_object_model.py`, com revisão fixada,
lista permitida de arquivos e verificação SHA-256. A inferência usa somente
arquivos locais, sem código remoto executável. O texto Apache-2.0 é preservado
no pacote local do modelo; essa licença não substitui a licença dos outros
componentes ou das imagens analisadas.

## Verificação de recortes - SigLIP 2

A mesma revisão utiliza
[google/siglip2-base-patch16-224](https://huggingface.co/google/siglip2-base-patch16-224),
revisão `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`, sob **Apache-2.0**.
O checkpoint é pré-treinado e não foi ajustado neste projeto. O verificador
compara recortes com descrições positivas/negativas fixas; sua margem de
similaridade não é uma probabilidade de risco.

`prepare_verifier_model.py` preserva o texto da licença e verifica revisão,
arquivos permitidos e hashes antes do carregamento local de safetensors.
A configuração de resolução fixa usa a arquitetura SigLIP e o tokenizador
Gemma fornecidos pelo checkpoint, sem carregar código remoto. As origens,
arquivos, licenças e transformações estão em
[model_sources.json](resultados/metrics/v4/model_sources.json); o estado ativo
e a configuração efetivamente validada estão em
[active_profile.json](resultados/metrics/v4/active_profile.json).
Não foi possível auditar integralmente os dados de pré-treinamento externos;
a ausência de duplicatas no dataset do projeto não prova ausência nesses dados.

## Código original do projeto

Salvo indicação em contrário, o código original deste repositório é:

```text
SPDX-License-Identifier: AGPL-3.0-only
Copyright (c) 2026 Equipe DrenaHub
```

Consulte o arquivo `LICENSE` para os termos completos.

## Biblioteca de testes do produto

As filmagens reais e fotos de bueiros em `dados/examples/` vêm do Wikimedia
Commons, com licenças CC BY ou CC BY-SA indicadas por arquivo. Autores, URLs,
versões das licenças, hashes e modificações estão em `dados/examples/sources.json`
e no README dessa pasta. Os clipes foram recortados, redimensionados e
convertidos para H.264 sem som. Quatro fontes de vídeo forneceram quadros
completos e/ou recortes para adaptação de treino v2/v3: `rato_close`,
`rato_residuos`, `drenagem_chuva` e `canal_residuos`. Seus resultados são
regressões conhecidas, não testes independentes. `rato_rua`,
`sarjeta_residuos` e as duas fotos do catálogo têm papéis separados, descritos
em `sources.json`, e não possuem caixas de referência completas para alegar
acurácia por quadro. As categorias do catálogo descrevem o assunto, não
previsões nem rótulos de avaliação. As fotos adicionais recebidas do usuário
permanecem locais e não são redistribuídas como amostras licenciadas do catálogo.

