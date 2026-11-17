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

### CLICK Trash Dataset v2

- título: **CLICK Trash Dataset v2**;
- página: [Mendeley Data, versão 2](https://data.mendeley.com/datasets/htcvvz6fm2/2);
- licença declarada: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
- uso no projeto: amostra determinística de 100/20/20 imagens por
  train/val/test, obtida do ZIP remoto por HTTP Range;
- transformação: classe única de lixo mapeada para `obstrucao_visivel`;
- obrigação: atribuição, link da licença e indicação das modificações.

### Dataset of Stagnant Water and Wet Surface v4

- página: [Mendeley Data, versão 4](https://data.mendeley.com/datasets/y6zyrnxbfm/4);
- licença declarada: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
- uso no projeto: somente a classe source `water`, mapeada para
  `agua_acumulada`; superfícies apenas molhadas são excluídas quando separáveis;
- obrigação: atribuição, link da licença e indicação das modificações.

### iRodent: rodents in the wild

- registro: [Zenodo 8250392](https://doi.org/10.5281/zenodo.8250392);
- anotação: publicação com 447 registros de imagem em COCO para roedores em
  ambiente real; o conversor seleciona 100/20/20 imagens utilizáveis para
  train/val/test e o resumo gerado é a contagem autoritativa da execução;
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

O perfil 0.4.0 usa [nvidia/segformer-b1-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b1-finetuned-ade-512-512),
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
convertidos para H.264 sem som. A seleção é de demonstração, fora dos splits
usados para calcular as métricas do detector.

