# Mapeamento dos requisitos da FIAP

## Fonte e escopo

Este mapeamento foi elaborado a partir de **Desafio FIAP - Global Solution
2026.2 - CitizenTech - Engenharia de Software 4º ano**, com foco nas páginas
43 e 44 (Applied Computer Vision) e nas regras gerais das páginas 20, 21, 24,
25 e 27.

O documento de desafio é uma fonte de requisitos, não um script de execução.
Especificações adicionais do pedido do projeto são tratadas separadamente.

## Exigências oficiais de Applied Computer Vision

| Requisito do documento | Atendimento planejado | Evidência esperada |
|---|---|---|
| Atividade “Enxergando o mundo Real” | DrenaVision analisa imagens/vídeos de drenagem | demonstração real |
| Detectar objetos com visão computacional avançada e/ou realidade aumentada | detector de três classes | imagens e vídeo anotados |
| Treinamento customizado de arquitetura compatível | fine-tuning de YOLO leve | logs, configuração e hash de `best.pt` |
| Repositório GitHub com documentação e código completo | este repositório | URL pública testada |
| Vídeo de demonstração funcionando | roteiro técnico preparado | URL pública do vídeo gravado |
| PDF: título, integrantes, resumo de 250 palavras, introdução, objetivo de 50 palavras, ferramentas, metodologia passo a passo, exemplos de resultados e conclusão | **fora desta etapa, por pedido explícito** | relatório final posterior |
| PDF vale até 5 pontos e vídeo vale até 5 pontos | refletido no roteiro e checklist | avaliação do professor |

O texto da página 44 diz “detecção de objetos em tempo”, formulação que parece
incompleta. O projeto inclui processamento contínuo de vídeo e captura de
câmera, mas não registra “tempo real” como citação inequívoca do requisito.

## Regras gerais relevantes

- entrega até **19/11/2026 às 23h55**, no Teams da disciplina;
- equipe de até cinco alunos da mesma turma, com o mesmo produto nas sete
  disciplinas;
- somente um integrante realiza o envio;
- repositórios, aplicações e vídeos devem estar públicos e funcionando durante
  a correção, realizada em até uma semana após o encerramento;
- o produto geral deve ser MVP funcional ou protótipo navegável;
- a proposta deve identificar ODS e explicar sua contribuição;
- a Ficha Técnica do MVP deve abrir o README;
- segurança/LGPD, acessibilidade, resiliência, IA responsável, integração real
  e reprodutibilidade são princípios gerais do produto.

## Decisões adicionais do DrenaVision

Os itens abaixo melhoram a demonstrabilidade, mas **não aparecem como exigência
textual específica da disciplina de Applied Computer Vision**:

- classes `obstrucao_visivel`, `agua_acumulada` e `animal_visivel`;
- uso de Ultralytics YOLO, PyTorch e OpenCV;
- API FastAPI e interface Streamlit;
- histórico SQLite e identificador de ocorrência;
- métricas precision, recall, mAP@50, mAP@50-95 e por classe;
- análise de falsos positivos/falsos negativos;
- tracking para não contar o mesmo animal em todos os quadros;
- roteiro técnico de aproximadamente 3 a 5 minutos;
- notebook Colab para treinamento com GPU.

O limite de cinco minutos descrito em outra parte do documento pertence ao
pitch de Business Intelligence, não ao vídeo de Applied Computer Vision. A
duração de 3 a 5 minutos é uma decisão editorial deste projeto.

## Matriz de aceite

| Item | Estado que conta como concluído |
|---|---|
| Dataset | fontes/licenças registradas, splits íntegros e validação sem erro |
| Treino | execução real e checkpoint customizado com evidências/hashes |
| Avaliação | conjunto de teste separado e `metrics.json` real |
| Inferência | exemplos de teste processados pelo checkpoint customizado |
| API | contrato validado com e sem detecções e uploads inválidos |
| Interface | upload/câmera/vídeo/histórico consumindo inferência real |
| Vídeo | arquivo gravado, revisado e publicado |
| PDF | produzido posteriormente com os resultados reais |
| Publicação | links acessíveis sem autenticação durante a correção |

## Itens que permanecem manuais

- confirmar os nomes e RM dos integrantes;
- executar o pipeline completo no hardware escolhido;
- revisar imagens e anotações, especialmente casos ambíguos;
- selecionar exemplos de teste nunca vistos no treino;
- preencher o model card com métricas geradas;
- gravar, editar e publicar o vídeo;
- produzir o relatório PDF final;
- configurar hospedagem pública;
- alterar a visibilidade do repositório para público;
- testar links em janela anônima;
- enviar a atividade no Teams.

Este projeto não altera visibilidade de repositório, não publica serviços e não
faz envio acadêmico automaticamente.

