# Exemplos reais para experimentar

Esta biblioteca reúne seis filmagens reais e duas fotografias reais de roedores,
resíduos e drenagem. São **entradas sem caixas desenhadas**. A interface executa
o modelo e produz as marcações quando encontra algo.

Os clipes têm até 12 segundos, H.264, 20 FPS e áudio removido para abrir nos
navegadores. As fotos foram reduzidas. Todas as modificações, hashes e links dos
originais estão em `sources.json`. Os arquivos com `_capa` são quadros reais
completos extraídos dos próprios vídeos, sem caixas ou recortes. O instante de
cada capa está no campo `thumbnail_seconds` do catálogo.

| Exemplo | Autor | Origem | Licença da mídia |
|---|---|---|---|
| `rato_close.mp4` | Ykmyks | [Rat.ogv](https://commons.wikimedia.org/wiki/File:Rat.ogv) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `rato_rua.mp4` | NaIzletuSi ™ | [Rato em Ljubljana](https://commons.wikimedia.org/wiki/Special:Redirect/page/122315232) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| `rato_residuos.mp4` | RG72 | [Rato em Hanói](https://commons.wikimedia.org/wiki/Special:Redirect/page/98651680) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `drenagem_chuva.mp4` | Xyxyzyz | [Drenagem em Werrington](https://commons.wikimedia.org/wiki/Special:Redirect/page/116160402) | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |
| `canal_residuos.mp4` | Ibnali1 | [Canal de Alajo](https://commons.wikimedia.org/wiki/File:Alajo_Storm_Drain_011.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `sarjeta_residuos.mp4` | Ibnali1 | [Sarjeta de Alajo](https://commons.wikimedia.org/wiki/File:Gutter_to_the_Alajo_Storm_Drain.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `ralo_lixo.jpg` | Tdorante10 | [Bueiro em Queens](https://commons.wikimedia.org/wiki/Special:Redirect/page/92592505) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `bueiro_agua.jpg` | Bart Everson | [Drenagem em Nova Orleans](https://commons.wikimedia.org/wiki/Special:Redirect/page/86283786) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) |

As licenças das mídias são próprias e não mudam para a licença do código do
projeto. Ao redistribuir os derivados, preserve a atribuição e a licença indicada
para cada arquivo, inclusive a condição de compartilhamento pela mesma licença
dos arquivos CC BY-SA.

## Como interpretar

Os exemplos têm assuntos diferentes. Somente `rato_close`, `rato_rua` e
`rato_residuos` são filmagens de roedores conforme a fonte. Os outros três vídeos
mostram drenagem/água; as duas fotos mostram bueiros com resíduos e não contêm
rato visível. A classe `animal_visivel` do modelo representa roedores de espécies
variadas dos dados de treino; uma caixa dessa classe não identifica a espécie.

As categorias descrevem o assunto, sem garantir detecção ou servir como caixas
de referência. A capa de `rato_rua`, aos 9 s, mantém a cena completa: a ocorrência
é pequena, parcialmente escondida e pouco legível. Ela permanece como exemplo
difícil, sem alegar acerto ou criar uma caixa de referência incerta. As capas de
`rato_close` (0,5 s) e `rato_residuos` (0 s) mostram o assunto mais claramente.

No protocolo de adaptação v2, quadros de `rato_close` e `rato_residuos` foram
anotados para treino. Recortes sem alvos de `canal_residuos` e `drenagem_chuva`
viraram negativos de treino. Esses quatro clipes são fontes de adaptação e não
podem ser apresentados como teste independente do candidato. `rato_rua` continua
fora do treino como regressão separada; os demais exemplos servem à regressão
manual. Os splits oficiais de validação/teste usam imagens próprias do dataset.
O campo `evaluation_role` registra essa distinção por arquivo.

Não há filmagem de um objeto colocado artificialmente no ralo nesta biblioteca.
Para testar um bueiro com lixo de perto, use as fotografias reais `ralo_lixo.jpg`
e `bueiro_agua.jpg`; para observar água em movimento, use `sarjeta_residuos.mp4`.
O objeto junto à parede da sarjeta é de natureza incerta; não é ground truth de
lixo nem prova de presença de animal. Uma falha ou falso alerta deve permanecer
visível na validação.

## Recriar os arquivos

Na raiz do repositório, com o ambiente ativado:

```powershell
python ferramentas/scripts/prepare_examples.py
```

Para atualizar somente descrições e capas dos arquivos já existentes, sem
reencodar nem mudar os hashes dos vídeos/fotos:

```powershell
python ferramentas/scripts/prepare_examples.py --refresh-catalog
```

O comando consulta os créditos/licenças no Wikimedia Commons, verifica o SHA-1
fixado de cada original, baixa os arquivos e prepara os clipes. Se a origem ou a
licença mudar, o processo para para revisão. Os originais ficam somente no cache
local; `sources.json` contém o SHA-256 de cada arquivo pronto.
