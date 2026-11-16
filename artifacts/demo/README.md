# Evidências demonstrativas

Esta pasta contém somente três imagens transformadas, escolhidas de forma
determinística **antes** da inferência: uma imagem do split de teste por classe.
As imagens receberam as caixas previstas pelo checkpoint DrenaVision; ausência
de caixa significa ausência de detecção no limiar 0,35, e não ausência do objeto
anotado.

| Arquivo-base | Seleção | Fonte/licença | Resultado do modelo |
|---|---|---|---|
| `click_trash_00c5aac3d4cf1d57` | determinística, antes da inferência | [CLICK Trash v2](https://data.mendeley.com/datasets/htcvvz6fm2/2), CC BY 4.0 | 2 detecções de `obstrucao_visivel` |
| `stagnant_water_04213a44e43bc933` | determinística, antes da inferência | [Stagnant Water v4](https://data.mendeley.com/datasets/y6zyrnxbfm/4), CC BY 4.0 | nenhuma detecção; falso negativo documentado |
| `irodent_02af19356995e059` | determinística, antes da inferência | [iRodent](https://zenodo.org/records/8250392), tratada como CC BY-NC 4.0 | nenhuma detecção; falso negativo documentado |
| `stagnant_water_78c5a3feac2bba2c` | primeiro TP estratificado da avaliação | Stagnant Water v4, CC BY 4.0 | 2 detecções; o caso ainda contém erros |
| `irodent_311988d4ce1851a6` | primeiro TP estratificado da avaliação | iRodent, tratada como CC BY-NC 4.0 | 2 detecções; o caso ainda contém erro |

Os hashes das entradas, o nome da imagem e o JSON integral da predição estão
nos arquivos `*_prediction.json`. O manifesto do dataset registra URL, versão,
licença, transformação e proveniência por imagem. O arquivo
`video_input_manifest.json` deixa explícito que a entrada demonstrativa é uma
sequência dessas três imagens estáticas, não uma filmagem capturada. O vídeo
anotado e seu manifesto completo são distribuídos como artefatos da release.

Os dois exemplos adicionais foram escolhidos **depois** da avaliação somente
para demonstrar que há verdadeiros positivos de água e animal. “TP” significa
ao menos uma correspondência classe-a-classe com IoU ≥ 0,50 no limiar 0,35;
não significa que todas as caixas da imagem foram encontradas nem que não haja
falso positivo. As métricas continuam sendo calculadas nas 62 imagens do teste.

Estas amostras são evidência acadêmica e não concedem direitos adicionais
sobre os datasets. O material iRodent não deve ser reutilizado comercialmente
sem autorização e revisão da licença.
