# Checklist de entrega FIAP

## Código e dados

- [x] repositório contém código completo e documentação;
- [x] fontes, versões e licenças constam no manifesto;
- [x] dataset tem train/val/test e imagens negativas;
- [x] validação com hash/pHash passa sem vazamento;
- [x] amostras versionadas têm licença e atribuição documentadas;
- [x] pesos, datasets completos, segredos e dados pessoais estão fora do Git.

## Treino e avaliação

- [x] fine-tuning foi realmente executado;
- [x] `training_manifest.json` comprova mudança de pesos;
- [x] `best.pt` tem hash e mapa correto das três classes;
- [x] teste separado gerou `metrics.json`;
- [x] precision, recall, mAP e métricas por classe foram conferidas;
- [x] matriz e exemplos estratificados de TP/FP/FN foram preservados;
- [x] limitações, inclusive classe fraca, estão declaradas.

## Demonstração

- [x] API inicia e `/health` carrega o modelo;
- [x] imagem com detecção real funciona;
- [x] imagem sem detecção retorna lista vazia;
- [x] upload inválido é rejeitado;
- [x] vídeo anotado real é exportado;
- [x] tracking evita soma ingênua por quadro;
- [x] histórico registra ponto, UTC, versão, frame e ocorrência;
- [ ] disclaimer aparece na UI e no vídeo.

## Vídeo técnico

- [ ] roteiro de `VIDEO_ROTEIRO.md` ensaiado;
- [ ] dataset, treino e hashes aparecem;
- [ ] uma cena por classe usa exemplo de teste;
- [ ] JSON de integração aparece;
- [ ] métricas reais e um erro aparecem;
- [ ] vídeo não mostra segredo/dado pessoal;
- [ ] créditos/licenças foram incluídos;
- [ ] link público foi testado sem login.

## PDF posterior

- [ ] título e integrantes;
- [ ] resumo com 250 palavras;
- [ ] introdução;
- [ ] objetivo com 50 palavras;
- [ ] ferramentas;
- [ ] metodologia passo a passo;
- [ ] exemplos de resultados reais;
- [ ] conclusão;
- [ ] links e referências conferidos.

## Publicação e envio

- [ ] preencher URLs públicas no README;
- [ ] tornar o repositório público antes da correção;
- [ ] publicar aplicação/protótipo e vídeo;
- [ ] testar tudo em janela anônima e outro dispositivo;
- [ ] manter links ativos durante a correção;
- [ ] confirmar integrantes (até cinco, mesma turma/produto);
- [ ] apenas um integrante realizar o envio;
- [ ] enviar no Teams até **19/11/2026 às 23h55**.

O mapeamento entre este checklist e o documento oficial está em
[FIAP_REQUIREMENTS.md](FIAP_REQUIREMENTS.md).

