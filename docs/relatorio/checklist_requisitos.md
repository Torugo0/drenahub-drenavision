# Checklist da documentação e da entrega

Disciplina: **Applied Computer Vision - Enxergando o Mundo Real**.
Fonte dos requisitos: desafio FIAP Global Solution 2026.2 - CitizenTech,
páginas 43-44. As páginas abaixo são as páginas físicas do PDF, incluindo a capa.

## Requisitos oficiais do PDF

| Requisito | Onde foi atendido | Conferência |
|---|---|---|
| Título e identificação da disciplina | p. 1, capa | FIAP, Engenharia de Software, CitizenTech, DrenaHub e professores. |
| Integrantes e RM | p. 1 | Gabriel Machado Carrara Pimentel - RM99880; Lourenzo Ramos - RM99951; Vítor Hugo Rodrigues - RM97758. |
| Resumo com 250 palavras | p. 2, seção 1 | **250 palavras**, contadas programaticamente na fonte e no texto extraído do PDF. |
| Introdução | p. 2, seção 2 | Problema, inspeção assistida e alcance acadêmico. |
| Objetivo com 50 palavras | p. 3, seção 3 | **50 palavras**, contadas programaticamente na fonte e no PDF. |
| Explicação técnica das ferramentas | p. 3, seção 4 | Tecnologias, versões, funções e ambiente; especialistas pré-treinados separados do treinamento customizado. |
| Metodologia passo a passo | p. 4-7, seção 5 | Dados, licenças, rótulos, preparação, treinamento, avaliação, inferência e integração. |
| Exemplos de resultados | p. 8-14, seção 6 | Métricas, curvas, matriz de confusão e imagens reais com marcações, perdas e falso positivo. |
| Conclusão | p. 17, seção 9 | Realizações comprovadas, limites e próximos passos. |

O documento tem **20 páginas A4**, com margens, cabeçalho, rodapé e paginação.
Resumo e objetivo são contados por tokens Unicode separados por espaço;
títulos e palavras-chave não entram nesses denominadores.

## Conteúdo adicional solicitado e rastreabilidade

| Conteúdo | Localização |
|---|---|
| Dataset, fontes, licenças, divisões e quantidade de caixas | p. 4-5. |
| Treinamento executado: 63 épocas, melhor na 43, batch 12, seed 42, checkpoint | p. 5 e p. 20. |
| Precisão, recall, AP por classe e mAP do YOLO isolado | p. 8; números conferidos contra os JSON v3. |
| Curvas de treino/validação | p. 8, Figura 1, gerada do CSV de 63 épocas. |
| Curva precisão-recall e matriz do teste YOLO correto | p. 9, Figuras 2-3; não são gráficos do composto 0.5. |
| Holdout40 e regressões conhecidas179 separados | p. 6 e p. 10; perdas de localização e precisão explícitas. |
| Seis vídeos, duas fotos, 1.388 quadros e latência | p. 11. |
| Resíduos e água | p. 12, Figura 4, Bart Everson, CC BY 2.0. |
| Animal com predição real | p. 12, Figura 5, Ykmyks, CC BY-SA 4.0; fonte conhecida de adaptação. |
| Resíduos e água rasa não marcados | p. 13, Figura 6, Tdorante10, CC BY-SA 4.0. |
| Falso positivo no PVC instalado | p. 13, Figura 7, Ibnali1, CC BY-SA 4.0. |
| Entrada sem marcações | p. 14, Figura 8, Patil/Bhutad, CC BY 4.0; negativa revisada somente de água livre, sem certificação de ausência das três classes. |
| Diagrama da arquitetura real | p. 15, Figura 9. |
| Oito endpoints públicos e excerto JSON real | p. 15-16. |
| Discussão e limites operacionais | p. 17. |
| Referências, créditos e links | p. 18-20. |
| Proveniência e hashes das imagens reproduzidas | `assets/manifesto_figuras.json`; não há detecções sintéticas nem caixas corrigidas manualmente. |
| Fonte editável | `relatorio_fonte.md`. |
| Evidência da geração/validação | `validacao_relatorio.json`. |

## Validações executadas

- [x] Identificação e RMs conferidos contra os dados fornecidos pela equipe.
- [x] Resumo com 250 palavras e objetivo com 50 palavras.
- [x] Presença das onze seções e dos elementos exigidos no enunciado.
- [x] Treinamento, dataset, checkpoint e métricas confrontados com os artefatos originais.
- [x] Resultados do detector treinado separados da composição de modelos e da calibração.
- [x] Figuras reais com origem, autor, licença, modificações e papel de avaliação.
- [x] Referências HTTP verificadas; repositório existente conferido pelo conector GitHub.
- [x] Suíte completa: **239 testes aprovados em 32,21 segundos**, com um aviso de depreciação de dependência de teste.
- [x] Ruff, análise sintática Python e PowerShell; iniciador verificado com porta alternativa quando 8501 pertence a outro projeto.
- [x] PDF aberto, texto extraído e todas as páginas renderizadas para inspeção visual.
- [x] Tabelas, figuras, cabeçalhos, legendas, paginação e caracteres revisados; quebras que deixavam páginas quase vazias corrigidas.
- [x] Estrutura do código e histórico Git auditados; caches, `.env`, banco, datasets completos e pesos fora do versionamento de código.
- [x] Guias e avisos de terceiros atualizados; configurações e métricas históricas úteis preservadas.
- [x] Vídeo original copiado sem alteração, com hash e decodificação integral conferidos.
- [x] Código e PDF publicados no GitHub público; dez verificações anônimas HTTP 200, incluindo página/PDF, README, configuração, manifesto selado e quatro assets da release. PDF e arquivos selados baixados com hashes idênticos aos locais.

## Formato final da entrega

1. **`demo_drena_vision.mp4`**: vídeo original anexado, 2min13,20s, 1920 × 1080, 30 FPS; trilha silenciosa preservada.
2. **`LINK_REPOSITORIO.txt`**: URL do repositório e acesso direto ao PDF, com os integrantes.
3. **`DrenaVision_Applied_Computer_Vision_FIAP.pdf`**: documentação separada, também disponível no início do README do GitHub.

Os três arquivos ficam em `resultados/local/entrega/`, fora do Git. O código e
o relatório são entregues pelo repositório; o vídeo vai como anexo, sem exigir
um serviço externo de hospedagem. A equipe decidiu não usar ZIP nesta entrega.

Repositório: https://github.com/Torugo0/drenahub-drenavision

PDF no GitHub: https://github.com/Torugo0/drenahub-drenavision/blob/main/docs/relatorio/DrenaVision_Applied_Computer_Vision_FIAP.pdf

## Limites e pendências que permanecem

- A equipe realiza o envio acadêmico dos três anexos e mantém o repositório acessível durante a correção; esse envio não foi automatizado.
- A detecção tem perdas, duplicidades e falsos positivos; não é validação operacional, previsão de alagamentos ou diagnóstico sanitário.
- O protótipo não produz contagem física garantida de objetos, profundidade da água ou probabilidade calibrada de risco.
- Cinco negativos de treino são claramente revisados; referências de pose e grupos de resíduos limitam medidas de localização.
- Os resultados históricos são mantidos para comparação, sem serem apresentados como testes inéditos da versão atual.
- Nenhum código-fonte sem função comprovada foi identificado. Caches `.pyc` locais não foram removidos: o controle automático rejeitou as tentativas, e esses arquivos já são ignorados pelo Git.

## Regerar o PDF a partir da fonte

Com o ambiente do projeto instalado:

```powershell
python -m pip install -e ".[report]"
python ferramentas/scripts/gerar_relatorio.py --render
```

O gerador valida contagens, seções, identidade, métricas e hashes antes de
renderizar. As páginas de QA ficam em `resultados/local/report_audit/pdf_qa/`.
Após qualquer mudança de conteúdo, repetir a inspeção das páginas; um PDF
gerado sem erro de execução ainda pode exigir ajuste visual.
