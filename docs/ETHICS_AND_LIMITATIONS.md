# Ética, uso responsável e limitações

## Declaração de finalidade

O DrenaVision é um protótipo acadêmico de **triagem visual assistida**. Ele
prioriza imagens que merecem revisão humana; não decide sozinho se um local
está seguro, interditado, alagado ou entupido.

Mensagens permitidas:

- “há uma região visualmente compatível com água acumulada”;
- “foi detectado material visível próximo ao ponto monitorado”;
- “há um animal visível; recomenda-se inspeção”.

Mensagens não permitidas:

- “a água tem X centímetros” sem referência e calibração;
- “vai alagar” ou “há X% de risco” a partir da imagem;
- “o ralo está entupido” apenas por uma caixa de detecção;
- “o animal bloqueou a drenagem” apenas por sua presença;
- identificação de espécie quando o modelo não foi treinado/validado para isso;
- “nenhuma detecção” como sinônimo de “local seguro”.

## Limitações técnicas

### Escopo de classe

`obstrucao_visivel` agrega resíduos, lixo, folhas e outros materiais com formas
muito diferentes. `agua_acumulada` pode ser confundida com reflexo, piso
molhado, sombra ou brilho. `animal_visivel` é amplo, enquanto a fonte iRodent é
concentrada em roedores; desempenho em cães, gatos, aves, répteis ou outros
animais não pode ser inferido sem dados próprios.

### Profundidade e geometria

Uma única imagem sem referência métrica, pose de câmera e calibração não
permite obter profundidade confiável da água. A área de uma caixa em pixels
também não mede volume, vazão ou severidade hidráulica.

### Mudança de domínio

O desempenho pode cair com:

- pouca luz, contraluz, reflexo, chuva, neblina ou sujeira na lente;
- câmeras, lentes, compressão e resoluções diferentes das fontes de treino;
- ângulos rasantes, oclusão e objetos pequenos;
- pisos, grelhas, ralos e ambientes não representados;
- regiões geográficas e espécies não presentes no dataset;
- mudanças sazonais ou operacionais.

Uma avaliação aleatória no dataset não substitui teste por local/câmera e
monitoramento contínuo em uso real.

### Contagem em vídeo

Rastreamento reduz dupla contagem, mas IDs podem mudar após oclusão, saída da
cena ou cruzamento de objetos. A contagem de IDs únicos é estimativa de
ocorrências visuais, não censo de animais. Sem tracking, o sistema usa pico
conservador e declara essa limitação.

### Calibração de confiança

A confiança do detector não é probabilidade calibrada de risco nem certeza
causal. Alterar o limiar modifica precision e recall; o valor operacional deve
ser escolhido com conjunto de validação e custo de falsos positivos/negativos.

## Limitações dos dados

### Cobertura

As fontes públicas ajudam a iniciar o treinamento, mas não reproduzem
necessariamente condomínios, garagens e centros logísticos brasileiros. Antes
de uso real, a DrenaHub deve coletar imagens próprias consentidas, documentar
câmeras/cenários e manter amostras negativas de ralos limpos, piso seco e áreas
sem animais.

### Licença do iRodent

O registro Zenodo menciona Apache-2.0, enquanto a documentação do conjunto
indica imagens-fonte CC BY-NC. O projeto adota o tratamento conservador
**CC BY-NC**. Isso impede presumir que o material pode compor um produto
comercial. A equipe deve obter autorização adicional, remover a fonte ou
substituí-la por imagens próprias/compatíveis antes de comercialização.

### Vazamento entre splits

Frames vizinhos do mesmo vídeo, versões redimensionadas e fotos quase idênticas
podem inflar métricas. O pipeline usa hash exato, hash perceptual e
`source_group`; ainda assim, casos sinalizados precisam de revisão humana.

### Anotação

Caixas são interpretações humanas. Água sem contorno claro, lixo parcialmente
oculto e animal distante podem gerar desacordo. Recomenda-se guia escrito,
dupla revisão de amostra e registro de ambiguidades. Corrigir labels depois de
olhar o conjunto de teste invalida a independência do teste; nesse caso, crie
uma nova versão e reavalie do início.

### Dados sintéticos

Imagens sintéticas podem testar interface, formato e desempenho técnico, desde
que rotuladas como sintéticas. Elas não sustentam alegações de precisão em
imagens reais e não devem ser misturadas silenciosamente ao conjunto de teste.

## Privacidade e LGPD

Imagens de infraestrutura podem capturar pessoas, placas, unidades, endereços,
rotinas ou metadados de localização. Para uso além da demonstração:

1. defina finalidade, base legal, controlador/operador e canal de atendimento;
2. enquadre a câmera para minimizar dados pessoais;
3. remova EXIF e metadados desnecessários;
4. aplique anonimização quando pertinente;
5. limite acesso e mantenha logs auditáveis;
6. defina retenção curta e descarte verificável;
7. documente compartilhamentos e transferência internacional;
8. realize avaliação de impacto quando o contexto exigir.

Não coloque nome, apartamento, telefone ou endereço no `point_id`. O protótipo
local não fornece, sozinho, conformidade com LGPD.

## Riscos por erro do modelo

| Erro | Possível efeito | Mitigação |
|---|---|---|
| falso negativo | inspeção deixa de ser priorizada | rotinas periódicas independentes, limiar validado e amostragem manual |
| falso positivo | deslocamento e custo desnecessários | confirmação humana e feedback estruturado |
| erro de classe | encaminhamento incorreto | mostrar imagem/caixa/confiança e permitir correção |
| dupla contagem | superestimação de ocorrências | tracking, janela temporal e revisão |
| viés de domínio | desempenho desigual por local | teste estratificado e coleta local |

O DrenaVision não deve ser a única barreira de segurança em um sistema de
prevenção de danos.

## Supervisão humana

Uma ocorrência deve exibir a imagem, caixas, classes, confiança, versão do
modelo, horário e ponto técnico. O operador confirma, rejeita ou encaminha a
ocorrência. Feedback usado em novo treinamento precisa passar por curadoria e
versionamento; não se recomenda retreino automático em dados não revisados.

## Transparência de resultados

Toda afirmação quantitativa precisa indicar:

- versão/hash do checkpoint;
- versão e composição do dataset;
- split usado e ausência de vazamento conhecido;
- limiar, tamanho de imagem e dispositivo;
- intervalo ou número de amostras quando aplicável;
- métricas por classe e exemplos de erro;
- data da execução.

Não selecionar apenas exemplos favoráveis para o vídeo. Inclua pelo menos uma
limitação observável e, se `animal_visivel` tiver desempenho ruim, mostre o
resultado e explique cobertura de dados em vez de ocultá-lo.

## Licença e comercialização

Ultralytics usa AGPL-3.0 por padrão e oferece opção Enterprise. A AGPL e as
licenças dos datasets são questões independentes. Um produto fechado, interno
ou oferecido por rede requer análise jurídica e decisão explícita de licença.
Este documento é orientação técnica, não parecer jurídico.

## Condições mínimas antes de piloto real

- métricas reais e revisão de erros em dados representativos;
- critérios de aceitação por classe e local;
- teste de carga, segurança e recuperação;
- autenticação, autorização e criptografia;
- política de retenção e resposta a incidentes;
- treinamento de operadores e processo de contestação;
- monitoramento de drift e revalidação;
- validação de licenças de código, pesos e cada imagem;
- aprovação do responsável técnico e jurídico.

