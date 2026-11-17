# Entrega acadêmica — FIAP

## Requisitos e situação

Fonte: desafio Global Solution 2026.2 CitizenTech, Engenharia de Software,
páginas 43–44 (Applied Computer Vision) e regras gerais. O enunciado é fonte
de requisitos; escolhas adicionais de arquitetura pertencem ao DrenaVision.

| Requisito oficial | Evidência/estado |
|---|---|
| Detecção de objetos e/ou realidade aumentada | detector de três classes em fotos/vídeos |
| Treinamento customizado de arquitetura compatível | YOLO26n v3: 63 épocas, melhor na 43; manifesto, logs e hash. SegFormer B1 pré-treinado complementa água; não é apresentado como treinado pela equipe |
| GitHub com documentação e código completo | este repositório; visibilidade pública ainda pendente |
| Vídeo da solução funcionando | roteiro abaixo; gravação/publicação pendentes |
| PDF técnico | etapa posterior; conteúdo exigido no checklist abaixo |

PDF e vídeo valem até 5 pontos cada. O texto “detecção de objetos em tempo”
parece incompleto; o projeto demonstra vídeo processado e captura de câmera.
As três classes, FastAPI, Streamlit, SQLite, métricas detalhadas e duração-alvo
de 3–5 minutos são decisões do projeto. O enunciado de Computer Vision não
define essa duração; o limite de cinco minutos em outra parte refere-se ao
pitch de Business Intelligence.

Prazo geral: **19/11/2026 às 23h55**, Teams da disciplina. A equipe tem até
cinco integrantes da mesma turma/produto; somente um realiza o envio. Os links
devem permanecer públicos e funcionando durante a correção, prevista em até
uma semana após o encerramento. A ficha técnica abre o README.

## Contribuição aos ODS

O projeto pode apoiar ODS 3 (inspeção de água parada e animais), 6 (manutenção
da drenagem), 11 (prevenção local) e 13 (preparação operacional para chuvas).
Essas são contribuições potenciais. O detector não mede impacto sanitário,
ambiental ou climático; indicadores de resultado exigem validação em operação.

## Roteiro sugerido — 4min30s

| Tempo | Mostrar | Explicar |
|---|---|---|
| 0:00–0:30 | produto, público e três tipos de alerta | apoio à revisão humana de resíduos, água e animal |
| 0:30–1:10 | fontes, auditorias, manifesto, splits e treino v3 | caixas corrigidas, negativos revistos e pesos customizados |
| 1:10–2:25 | interface: foto e vídeos variados | enviar/escolher entrada, analisar, ver caixas e alertas |
| 2:25–3:05 | resultado e histórico | arquivo anotado, ocorrência registrada e encaminhamento |
| 3:05–3:40 | Swagger e JSON real | integração com evento, UTC, classe, caixa e confiança |
| 3:40–4:20 | teste separado e exemplo de falha | YOLO isolado mAP50 0,4845; produto: água em 33/40 novas fotos, com 2/10 falsos alertas em negativos; animal 46/69, uma perda ante v2 |
| 4:20–4:30 | produto e próximos passos | coleta no domínio real e confirmação humana |

Execute `iniciar.ps1` antes de gravar. Mostre pelo menos um erro real e use os
valores de `resultados/metrics/v3/independent_comparison.json` e
`resultados/metrics/v3/master_map_comparison.json`. A avaliação de presença do
produto é separada do AP do YOLO. A calibração é validação, não teste. A licença
NVIDIA permite pesquisa/avaliação não comercial; uma startup precisa resolver
as licenças antes de uso comercial. O arquivo antigo
`resultados/metrics/evaluation_metrics.json` registra o protocolo histórico
v1 com caixas de roedor defeituosas. Uma sequência montada
de fotografias deve ser identificada como tal. Filmagem e sequência de fotos
não são evidências equivalentes de movimento/rastreamento.

Exemplos escolhidos depois da avaliação servem para demonstrar a interface;
as métricas continuam sendo calculadas no conjunto completo de teste.

O rato em close tem marcação consistente no modelo ativo, mas forneceu frames
para adaptação e deve ser apresentado como demonstração de uma fonte de treino.
Não o apresente como teste independente. Mostre também a falha do rato na cena
com resíduos ou do clipe `rato_rua`, e a regressão da foto de lixo em Queens.
O desaparecimento da caixa sobre a telha não significa que o rato dessa cena
foi encontrado. A revisão completa está em
`resultados/metrics/v2/product_validation_v2.json`.

## Checklist

### Código, dados e modelo

- [x] Código de dados, treino, avaliação, inferência, API e interface disponível.
- [x] Fontes/licenças, splits e hashes registrados.
- [x] Dataset validado sem vazamento conhecido no protocolo publicado.
- [x] Fine-tuning executado e alteração dos pesos comprovada.
- [x] Avaliação independente, métricas e exemplos de erro publicados.
- [x] Limitações e baixa qualidade de água declaradas.

### Demonstração e publicação

- [ ] Conferir reprodução do vídeo e marcações na versão final da interface.
- [ ] Conferir foto/vídeo com e sem detecção e histórico.
- [ ] Ensaiar roteiro com as entradas e créditos escolhidos.
- [ ] Gravar áudio e tela legíveis, sem segredo ou dado pessoal.
- [ ] Mostrar fonte dos exemplos, resultados reais e pelo menos uma falha.
- [ ] Revisar vídeo e exportar MP4 reproduzível no navegador.
- [ ] Publicar vídeo, aplicação/protótipo e repositório conforme a entrega.
- [ ] Registrar URLs no README e conferir sem login em outro dispositivo.
- [ ] Manter os links ativos durante toda a correção.

### Relatório PDF posterior

- [ ] Título e integrantes/RM.
- [ ] Resumo com 250 palavras.
- [ ] Introdução.
- [ ] Objetivo com 50 palavras.
- [ ] Ferramentas utilizadas.
- [ ] Metodologia passo a passo.
- [ ] Exemplos reais de resultados, métricas e limitações.
- [ ] Conclusão, links, referências e créditos.

### Envio

- [ ] Confirmar integrantes e turma.
- [ ] Conferir links públicos e arquivos exigidos.
- [ ] Um integrante enviar pelo Teams até 19/11/2026 às 23h55.

Nesta etapa, o repositório permanece privado e o vídeo narrado/PDF ainda
precisam ser produzidos. O código não altera a visibilidade nem faz o envio
acadêmico automaticamente.
