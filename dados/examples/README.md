# Exemplos reais para experimentar

Esta biblioteca reúne seis filmagens reais e duas fotografias reais de roedores,
resíduos e drenagem. São **entradas sem caixas desenhadas**. A interface executa
o modelo e produz as marcações quando encontra algo.

Os clipes têm até 12 segundos, H.264, 20 FPS e áudio removido para abrir nos
navegadores. As fotos foram reduzidas. Todas as modificações, hashes e links dos
originais estão em `sources.json`. Os arquivos com `_capa` são quadros reais
extraídos dos próprios vídeos.

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

As categorias do catálogo descrevem o conteúdo dos arquivos. Elas não são
previsões nem rótulos de avaliação. Estes exemplos externos não foram adicionados
ao treino, validação ou teste do modelo. O rato parcialmente escondido e o
ambiente doméstico ajudam a observar limites de generalização; uma falha de
detecção deve permanecer visível.

Não há filmagem de um objeto colocado artificialmente no ralo nesta biblioteca.
Para testar um bueiro com lixo de perto, use as fotografias reais `ralo_lixo.jpg`
e `bueiro_agua.jpg`; para movimento com resíduos, use `sarjeta_residuos.mp4`.

## Recriar os arquivos

Na raiz do repositório, com o ambiente ativado:

```powershell
python ferramentas/scripts/prepare_examples.py
```

O comando consulta os créditos/licenças no Wikimedia Commons, verifica o SHA-1
fixado de cada original, baixa os arquivos e prepara os clipes. Se a origem ou a
licença mudar, o processo para para revisão. Os originais ficam somente no cache
local; `sources.json` contém o SHA-256 de cada arquivo pronto.
